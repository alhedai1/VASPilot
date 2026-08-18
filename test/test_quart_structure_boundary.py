from __future__ import annotations

import asyncio
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from vaspilot.server.quart_server.quart_server import (
    QuartCrewServer,
    _structure_resolver_from_environment,
)
from vaspilot.tools.structure_application_boundary import StructureApplicationBoundary
from vaspilot.tools.structure_request_applicability import (
    StructureRequestApplicabilityClassifier,
)


class FakeBoundary:
    def __init__(self, should_run_crewai: bool, rendered="deterministic result"):
        self.should_run_crewai = should_run_crewai
        self.rendered = rendered
        self.calls = []

    def handle(self, source_text, output_directory, invocation_key):
        self.calls.append((source_text, Path(output_directory), invocation_key))
        return SimpleNamespace(
            should_run_crewai=self.should_run_crewai,
            rendered_response=None if self.should_run_crewai else self.rendered,
        )


class FakeCrew:
    def __init__(self):
        self.fingerprint = SimpleNamespace(uuid_str="fingerprint")
        self.tasks = []
        self.kickoff_calls = 0

    def kickoff(self):
        self.kickoff_calls += 1
        return "crew result"


class FakeGenerator:
    def __init__(self):
        self.crew_calls = 0
        self.stop_calls = 0
        self.created_crew = FakeCrew()

    def crew(self, _local_dir):
        self.crew_calls += 1
        return self.created_crew

    def stop(self):
        self.stop_calls += 1


def bare_server(temp_dir: str, boundary: FakeBoundary) -> QuartCrewServer:
    server = object.__new__(QuartCrewServer)
    server.structure_boundary = boundary
    server.generator = FakeGenerator()
    server._crew_thread_ids = {}
    server._conversation_to_fingerprint = {}
    server._fingerprint_to_conversation = {}
    import threading

    server._mapping_lock = threading.Lock()
    server.system_log = lambda *_args, **_kwargs: None
    server.agent_output = lambda *_args, **_kwargs: None
    server.work_dir = temp_dir
    server.db_path = str(Path(temp_dir) / "tasks.db")
    server.task_semaphore = asyncio.Semaphore(1)
    server._running_threads = {}
    server._schedule_log_to_db = lambda *_args, **_kwargs: None
    return server


def initialize_db(path: str, conversation_id: str, text: str):
    with sqlite3.connect(path) as connection:
        connection.execute(
            """CREATE TABLE task_executions (
            conversation_id TEXT PRIMARY KEY, task_description TEXT, status TEXT,
            started_at TIMESTAMP, completed_at TIMESTAMP, result TEXT, error_message TEXT)"""
        )
        connection.execute(
            "INSERT INTO task_executions (conversation_id, task_description, status) VALUES (?, ?, ?)",
            (conversation_id, text, "queued"),
        )


class QuartBoundaryTests(unittest.TestCase):
    def test_missing_mp_key_has_clear_request_time_configuration_error(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "MP_API_KEY is required"):
                _structure_resolver_from_environment(Path("unused"))

    def test_pure_worker_bypasses_crew_construction_and_kickoff(self):
        with tempfile.TemporaryDirectory() as td:
            boundary = FakeBoundary(False, "search_results: mp-2815")
            server = bare_server(td, boundary)
            container = {}
            server._run_crew_kickoff_thread(td, "search MoS2", container, "conversation")
            self.assertEqual(container["result"], "search_results: mp-2815")
            self.assertFalse(container["crew_constructed"])
            self.assertFalse(container["crew_kickoff_called"])
            self.assertEqual(server.generator.crew_calls, 0)

    def test_ambiguous_retrieval_cannot_reach_crew_or_structure_creation(self):
        response = '{"status":"not_pure","evidence":null,"clarification":null}'
        classifier = StructureRequestApplicabilityClassifier(lambda _m: response)

        class ForbiddenCoordinator:
            calls = 0

            def handle_structure_request(self, *_args):
                self.calls += 1
                raise AssertionError("MP resolver path must not run")

        with tempfile.TemporaryDirectory() as td:
            coordinator = ForbiddenCoordinator()
            boundary = StructureApplicationBoundary(classifier, coordinator)
            server = bare_server(td, boundary)
            container = {}
            server._run_crew_kickoff_thread(
                td, "get the structure of water", container, "water"
            )
            self.assertTrue(container["result"].startswith("clarification_required:"))
            self.assertIn("isolated molecule", container["result"])
            self.assertFalse(container["crew_constructed"])
            self.assertFalse(container["crew_kickoff_called"])
            self.assertEqual(server.generator.crew_calls, 0)
            self.assertEqual(coordinator.calls, 0)

    def test_intercepted_scientific_outcomes_are_stored_completed(self):
        for rendered in (
            "ambiguous: multiple distinct structures",
            "no_matches: no records satisfy all explicit constraints",
            "unsupported_semantic: 3R is not supported",
        ):
            with self.subTest(rendered=rendered), tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
                conversation_id = "conversation"
                server = bare_server(td, FakeBoundary(False, rendered))
                initialize_db(server.db_path, conversation_id, "search")
                asyncio.run(server._execute_crew_task_async(conversation_id, "search"))
                with sqlite3.connect(server.db_path) as connection:
                    row = connection.execute(
                        "SELECT status, result, error_message FROM task_executions"
                    ).fetchone()
                self.assertEqual(row, ("completed", rendered, None))
                self.assertEqual(server.generator.crew_calls, 0)

    def test_non_pure_reaches_existing_crew_path(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            server = bare_server(td, FakeBoundary(True))
            container = {}
            server._run_crew_kickoff_thread(td, "calculate bands", container, "conversation")
            self.assertEqual(container["result"], "crew result")
            self.assertEqual(server.generator.crew_calls, 1)
            self.assertEqual(server.generator.created_crew.kickoff_calls, 1)

    def test_two_executions_use_separate_conversation_directories(self):
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
            boundary = FakeBoundary(False)
            server = bare_server(td, boundary)
            initialize_db(server.db_path, "first", "search")
            with sqlite3.connect(server.db_path) as connection:
                connection.execute(
                    "INSERT INTO task_executions (conversation_id, task_description, status) VALUES (?, ?, ?)",
                    ("second", "search", "queued"),
                )
            asyncio.run(server._execute_crew_task_async("first", "search"))
            asyncio.run(server._execute_crew_task_async("second", "search"))
            self.assertEqual(
                [call[1] for call in boundary.calls],
                [Path(td) / "first", Path(td) / "second"],
            )

    def test_later_identical_submissions_receive_different_ids(self):
        async def exercise():
            with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as td:
                fake_generator = FakeGenerator()
                with patch(
                    "vaspilot.server.quart_server.quart_server.VaspCrew",
                    return_value=fake_generator,
                ):
                    server = QuartCrewServer(
                        crew_config={},
                        work_dir=td,
                        structure_boundary=FakeBoundary(False),
                    )
                with patch("builtins.print"):
                    await server._init_db()
                server._process_queue = AsyncMock()
                client = server.app.test_client()
                first = await client.post("/submit", json={"task_description": "same"})
                second = await client.post("/submit", json={"task_description": "same"})
                first_id = (await first.get_json())["conversation_id"]
                second_id = (await second.get_json())["conversation_id"]
                self.assertNotEqual(first_id, second_id)

        asyncio.run(exercise())


if __name__ == "__main__":
    unittest.main()
