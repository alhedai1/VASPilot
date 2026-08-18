#!/usr/bin/env python3
"""Live, no-VASP generalization benchmark for structure retrieval."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
from typing import Any
import uuid

import yaml
from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from vaspilot.server.quart_server.quart_server import QuartCrewServer
from vaspilot.tools.structure_resolver import RESOLVER_POLICY_VERSION
from vaspilot.tools.structure_request_applicability import (
    APPLICABILITY_POLICY_VERSION,
)
from vaspilot.tools.structure_request_coordinator import COORDINATOR_POLICY_VERSION
from vaspilot.tools.structure_request_parser import PARSER_POLICY_VERSION
from vaspilot.tools.structure_semantics import SEMANTIC_POLICY_VERSION


SUITE = (
    ("A", "pure", "search for the structure of 2H phase of MoS2"),
    ("B", "pure", "get the most stable MoS2 structure"),
    ("C", "pure", "find rutile TiO2 structures"),
    ("D", "pure", "retrieve mp-2815"),
    ("E", "pure", "show hexagonal MoS2 structures in space group 194"),
    ("F", "mixed", "relax 2H-WS2 using VASP"),
    ("G", "mixed", "optimize rutile TiO2 using VASP"),
    ("H", "mixed", "calculate the band structure of the most stable Si structure"),
    ("I", "mixed", "run a DOS calculation for mp-2815"),
    ("J", "safe_ambiguity", "calculate the band structure of MoS2"),
    ("K", "safe_ambiguity", "relax crystalline H2O from Materials Project"),
    ("L", "clarification", "get the structure of water"),
    ("M", "clarification", "get the structure of sodium chloride"),
    ("N", "pass_through", "calculate the band structure from vasprun.xml"),
    ("O", "pass_through", "analyze POSCAR"),
    ("P", "pass_through", "explain the structure of water"),
)


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "not-installed"


class SpyCrew:
    def __init__(self) -> None:
        self.fingerprint = SimpleNamespace(uuid_str=f"benchmark-spy-{uuid.uuid4().hex}")
        self.tasks: list[Any] = []
        self.kickoff_calls = 0

    def kickoff(self) -> str:
        self.kickoff_calls += 1
        return "BENCHMARK_SPY: downstream calculation intentionally not run"


class SpyGenerator:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.calls: list[dict[str, Any]] = []

    def crew(self, local_dir: str, **kwargs: Any) -> SpyCrew:
        forbidden = set(kwargs.get("forbidden_tools", ()))
        configured = {
            tool
            for agent in self.config.get("agents", {}).values()
            if isinstance(agent, dict)
            for tool in (agent.get("tools") or [])
        }
        crew = SpyCrew()
        self.calls.append(
            {
                "directory": str(local_dir),
                "forbidden": forbidden,
                "available": configured - forbidden,
                "resolved_context": kwargs.get("resolved_structure_context"),
                "crew": crew,
            }
        )
        return crew

    def stop(self) -> None:
        return None


def build_server(config: dict[str, Any], captured_logs: list[str]) -> QuartCrewServer:
    server = object.__new__(QuartCrewServer)
    server.config = config
    server.structure_boundary = server._create_structure_boundary()
    server.generator = SpyGenerator(config)
    server._crew_thread_ids = {}
    server._conversation_to_fingerprint = {}
    server._fingerprint_to_conversation = {}
    server._mapping_lock = threading.Lock()
    server.system_log = lambda *args, **_kwargs: captured_logs.append(
        " ".join(map(str, args))
    )
    server.agent_output = lambda *args, **_kwargs: captured_logs.append(
        " ".join(map(str, args))
    )
    return server


def run_once(
    server: QuartCrewServer,
    case_id: str,
    category: str,
    prompt: str,
    run_number: int,
) -> tuple[dict[str, Any], str, str]:
    started = time.perf_counter()
    invocation_key = f"benchmark-{case_id}-{run_number}-{uuid.uuid4().hex}"
    before_calls = len(server.generator.calls)
    container: dict[str, Any] = {}
    rendered = ""
    cached = ""

    try:
        with tempfile.TemporaryDirectory(prefix=f"vaspilot-{case_id}-{run_number}-") as td:
            conversation_dir = Path(td).resolve()
            server._run_crew_kickoff_thread(
                str(conversation_dir), prompt, container, invocation_key
            )
            if "error" in container:
                raise container["error"]

            boundary = container["boundary_result"]
            coordinator = boundary.coordinator_result
            resolution = coordinator.resolution_result if coordinator else None
            new_calls = server.generator.calls[before_calls:]
            call = new_calls[0] if new_calls else None
            files = sorted(
                path.name for path in conversation_dir.rglob("*") if path.is_file()
            )
            request = (
                coordinator.request.model_dump(mode="json")
                if coordinator and coordinator.request
                else None
            )
            ordered_ids = (
                [candidate.material_id for candidate in coordinator.candidates]
                if coordinator
                else []
            )
            selected_id = (
                coordinator.selected.material_id
                if coordinator and coordinator.selected
                else None
            )
            selected_path_inside = bool(
                coordinator
                and coordinator.structure_path
                and Path(coordinator.structure_path)
                .resolve()
                .is_relative_to(conversation_dir)
            )
            diagnostics = None
            if resolution is not None:
                diagnostics = {
                    "error": resolution.error,
                    "requested_semantic_label": resolution.requested_semantic_label,
                    "normalized_semantic_label": resolution.normalized_semantic_label,
                    "equivalence_groups": [
                        group.model_dump(mode="json")
                        for group in resolution.equivalence_groups
                    ],
                    "semantic_matches": [
                        candidate.semantic_match.model_dump(mode="json")
                        if candidate.semantic_match
                        else None
                        for candidate in resolution.candidates
                    ],
                }
            rendered = str(container.get("result", ""))
            cached = boundary.model_dump_json()
            record = {
                "case_id": case_id,
                "category": category,
                "prompt": prompt,
                "run": run_number,
                "conversation_id": invocation_key,
                "classification": boundary.applicability.status.value,
                "request": request,
                "resolver_status": (
                    coordinator.resolver_status.value
                    if coordinator and coordinator.resolver_status
                    else None
                ),
                "ordered_ids": ordered_ids,
                "selected_id": selected_id,
                "classifier_retries": boundary.applicability.retry_count,
                "parser_retries": (
                    coordinator.parser_retry_count if coordinator else None
                ),
                "files_written": files,
                "file_count": len(files),
                "selected_path_inside_conversation": selected_path_inside,
                "crew_would_start": len(new_calls) == 1,
                "crew_start_count": len(new_calls),
                "crew_kickoff_count": (
                    call["crew"].kickoff_calls if call is not None else 0
                ),
                "search_materials_project_available": (
                    "search_materials_project" in call["available"]
                    if call is not None
                    else False
                ),
                "create_crystal_structure_available": (
                    "create_crystal_structure" in call["available"]
                    if call is not None
                    else False
                ),
                "forbidden_tools": sorted(call["forbidden"]) if call else [],
                "diagnostics": diagnostics,
                "elapsed_seconds": round(time.perf_counter() - started, 6),
                "failure": None,
            }
    except BaseException as exc:
        record = {
            "case_id": case_id,
            "category": category,
            "prompt": prompt,
            "run": run_number,
            "conversation_id": invocation_key,
            "classification": None,
            "request": None,
            "resolver_status": None,
            "ordered_ids": [],
            "selected_id": None,
            "classifier_retries": None,
            "parser_retries": None,
            "files_written": [],
            "file_count": 0,
            "selected_path_inside_conversation": False,
            "crew_would_start": False,
            "crew_start_count": 0,
            "crew_kickoff_count": 0,
            "search_materials_project_available": False,
            "create_crystal_structure_available": False,
            "forbidden_tools": [],
            "diagnostics": None,
            "elapsed_seconds": round(time.perf_counter() - started, 6),
            "failure": f"{type(exc).__name__}: {exc}",
        }
    return record, rendered, cached


def summarize(records: list[dict[str, Any]], run_count: int) -> list[dict[str, Any]]:
    summaries = []
    for case_id, category, prompt in SUITE:
        rows = [record for record in records if record["case_id"] == case_id]

        def consistency(field: str) -> bool:
            values = [json.dumps(row[field], sort_keys=True) for row in rows]
            return len(set(values)) <= 1

        summaries.append(
            {
                "case_id": case_id,
                "category": category,
                "prompt": prompt,
                "runs": len(rows),
                "classification_counts": dict(Counter(r["classification"] for r in rows)),
                "classification_consistent": consistency("classification"),
                "request_consistent": consistency("request"),
                "resolver_status_counts": dict(Counter(r["resolver_status"] for r in rows)),
                "resolver_status_consistent": consistency("resolver_status"),
                "ids_consistent": consistency("ordered_ids") and consistency("selected_id"),
                "selected_ids": dict(Counter(r["selected_id"] for r in rows)),
                "ordered_ids": rows[0]["ordered_ids"] if rows else [],
                "classifier_retry_runs": sum(bool(r["classifier_retries"]) for r in rows),
                "parser_retry_runs": sum(bool(r["parser_retries"]) for r in rows),
                "failure_count": sum(r["failure"] is not None for r in rows),
                "crew_start_counts": dict(Counter(r["crew_start_count"] for r in rows)),
                "file_count_counts": dict(Counter(r["file_count"] for r in rows)),
                "mean_elapsed_seconds": round(
                    sum(r["elapsed_seconds"] for r in rows) / max(len(rows), 1), 6
                ),
                "resolved_mixed_tool_policy_pass": all(
                    not r["search_materials_project_available"]
                    and not r["create_crystal_structure_available"]
                    for r in rows
                    if r["classification"] == "mixed_mp_structure"
                    and r["resolver_status"] == "selected"
                ),
                "complete": len(rows) == run_count,
            }
        )
    return summaries


def markdown_report(report: dict[str, Any]) -> str:
    lines = [
        "# Deterministic structure-retrieval generalization benchmark",
        "",
        f"- Generated: {report['generated_at']}",
        f"- Runs per prompt: {report['configuration']['runs_per_prompt']}",
        f"- Total runs: {report['overall']['total_runs']}",
        f"- Failures: {report['overall']['failures']}",
        f"- Fully consistent prompts: {report['overall']['fully_consistent_prompts']}/16",
        f"- MP_API_KEY leak detected: {report['secret_checks']['any_leak_detected']}",
        "",
        "| Case | Classification | Resolver | Selected ID | Consistent | Retries | Failures | Crew starts | Files |",
        "|---|---|---|---|---:|---:|---:|---|---|",
    ]
    for item in report["summaries"]:
        consistent = all(
            item[key]
            for key in (
                "classification_consistent",
                "request_consistent",
                "resolver_status_consistent",
                "ids_consistent",
            )
        )
        lines.append(
            "| {case_id} | {classification} | {resolver} | {selected} | {consistent} | "
            "{retries} | {failures} | {crew} | {files} |".format(
                case_id=item["case_id"],
                classification=json.dumps(item["classification_counts"], sort_keys=True),
                resolver=json.dumps(item["resolver_status_counts"], sort_keys=True),
                selected=json.dumps(item["selected_ids"], sort_keys=True),
                consistent="yes" if consistent else "no",
                retries=item["classifier_retry_runs"] + item["parser_retry_runs"],
                failures=item["failure_count"],
                crew=json.dumps(item["crew_start_counts"], sort_keys=True),
                files=json.dumps(item["file_count_counts"], sort_keys=True),
            )
        )
    lines.extend(("", "See the JSON report for requests, ordered IDs, diagnostics, and per-run timings."))
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "examples/1.Basic/configs/crew_config_en.yaml",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=REPOSITORY_ROOT / "benchmarks/results"
    )
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be positive")

    load_dotenv(REPOSITORY_ROOT / ".env", override=True)
    api_key = os.environ.get("MP_API_KEY", "").strip()
    if not api_key:
        parser.error("MP_API_KEY is required")
    with args.config.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    captured_logs: list[str] = []
    rendered_results: list[str] = []
    cached_results: list[str] = []
    server = build_server(config, captured_logs)
    records = []
    for case_id, category, prompt in SUITE:
        for run_number in range(1, args.runs + 1):
            record, rendered, cached = run_once(
                server, case_id, category, prompt, run_number
            )
            records.append(record)
            rendered_results.append(rendered)
            cached_results.append(cached)
            print(
                f"{case_id} {run_number}/{args.runs}: "
                f"{record['classification']} {record['resolver_status']} "
                f"{record['selected_id'] or '-'}"
            )

    summaries = summarize(records, args.runs)
    with sqlite3.connect(":memory:") as database:
        database.execute("CREATE TABLE results (payload TEXT)")
        database.executemany(
            "INSERT INTO results VALUES (?)",
            ((json.dumps(record, sort_keys=True),) for record in records),
        )
        database_text = "\n".join(
            row[0] for row in database.execute("SELECT payload FROM results")
        )
    git_diff = subprocess.run(
        ["git", "diff", "--no-ext-diff"],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    ).stdout
    repository_logs = "\n".join(
        path.read_text(encoding="utf-8", errors="replace")
        for path in (REPOSITORY_ROOT / "logs").glob("*")
        if path.is_file()
    )
    secret_checks = {
        "rendered_results": api_key in "\n".join(rendered_results),
        "captured_logs": api_key in "\n".join(captured_logs),
        "cached_results": api_key in "\n".join(cached_results),
        "database_rows": api_key in database_text,
        "repository_logs": api_key in repository_logs,
        "git_diff": api_key in git_diff,
    }
    secret_checks["any_leak_detected"] = any(secret_checks.values())
    fully_consistent = sum(
        all(
            summary[key]
            for key in (
                "classification_consistent",
                "request_consistent",
                "resolver_status_consistent",
                "ids_consistent",
            )
        )
        for summary in summaries
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "configuration": {
            "runs_per_prompt": args.runs,
            "config_path": str(args.config.resolve().relative_to(REPOSITORY_ROOT)),
            "live_materials_project": True,
            "downstream_execution": "spy_only_no_vasp",
            "fresh_conversation_id_per_run": True,
        },
        "versions": {
            "python": sys.version.split()[0],
            "vaspilot": package_version("vaspilot"),
            "mp-api": package_version("mp-api"),
            "pymatgen": package_version("pymatgen"),
            "pydantic": package_version("pydantic"),
            "quart": package_version("quart"),
            "resolver_policy": RESOLVER_POLICY_VERSION,
            "semantic_policy": SEMANTIC_POLICY_VERSION,
            "parser_policy": PARSER_POLICY_VERSION,
            "applicability_policy": APPLICABILITY_POLICY_VERSION,
            "coordinator_policy": COORDINATOR_POLICY_VERSION,
        },
        "overall": {
            "total_runs": len(records),
            "failures": sum(record["failure"] is not None for record in records),
            "fully_consistent_prompts": fully_consistent,
        },
        "secret_checks": secret_checks,
        "summaries": summaries,
        "records": records,
    }
    serialized = json.dumps(report, indent=2, sort_keys=True)
    if api_key in serialized:
        raise RuntimeError("refusing to write report containing MP_API_KEY")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = args.output_dir / f"structure_generalization_{stamp}.json"
    markdown_path = args.output_dir / f"structure_generalization_{stamp}.md"
    json_path.write_text(serialized + "\n", encoding="utf-8")
    markdown_path.write_text(markdown_report(report), encoding="utf-8")
    print(f"JSON_REPORT={json_path}")
    print(f"MARKDOWN_REPORT={markdown_path}")
    return 1 if report["overall"]["failures"] or secret_checks["any_leak_detected"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
