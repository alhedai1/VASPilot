from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from vaspilot.tools.structure_models import (
    CandidateResult,
    EquivalenceGroupResult,
    RequestOperation,
    ResolutionStatus,
    SelectionBehavior,
    StructureRequest,
    StructureResolutionResult,
    SymmetrySummary,
)
from vaspilot.tools.structure_request_coordinator import (
    InMemoryInvocationStore,
    StructureRequestCoordinator,
    render_structure_result,
)
from vaspilot.tools.structure_request_parser import (
    ParserStatus,
    StructureRequestParseResult,
)


REQUEST = StructureRequest(
    operation=RequestOperation.SEARCH,
    selection=SelectionBehavior.RETURN_ALL,
    formula="MoS2",
)


def candidate(material_id: str = "mp-2815") -> CandidateResult:
    return CandidateResult(
        material_id=material_id,
        formula="MoS2",
        chemical_system="Mo-S",
        nsites=6,
        energy_above_hull=0.0,
        formation_energy_per_atom=-1.5,
        is_stable=True,
        theoretical=False,
        deprecated=False,
        symmetry=SymmetrySummary(
            symbol="P6_3/mmc", number=194, crystal_system="hexagonal", point_group="6/mmm"
        ),
    )


def parsed(request: StructureRequest = REQUEST) -> StructureRequestParseResult:
    return StructureRequestParseResult(
        status=ParserStatus.PARSED, request=request, retry_count=1
    )


def resolution(status: ResolutionStatus, **kwargs) -> StructureResolutionResult:
    return StructureResolutionResult(
        status=status,
        request=REQUEST,
        resolver_policy_version="resolver-test-v1",
        **kwargs,
    )


class FakeParser:
    def __init__(self, result: StructureRequestParseResult, barrier=None):
        self.result = result
        self.calls = 0
        self.barrier = barrier
        self.lock = threading.Lock()

    def parse(self, _source_text: str) -> StructureRequestParseResult:
        with self.lock:
            self.calls += 1
        if self.barrier:
            self.barrier.wait(timeout=2)
        return self.result


class FakeResolver:
    def __init__(self, result=None, exception=None):
        self.result = result
        self.exception = exception
        self.calls = 0
        self.requests = []
        self.lock = threading.Lock()

    def resolve(self, request):
        with self.lock:
            self.calls += 1
            self.requests.append(request)
        if self.exception:
            raise self.exception
        return self.result


class CoordinatorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def coordinator(self, parse_result, resolve_result):
        parser = FakeParser(parse_result)
        resolver = FakeResolver(resolve_result)
        coordinator = StructureRequestCoordinator(parser, lambda _path: resolver)
        return coordinator, parser, resolver

    def test_parsed_calls_parser_once_and_resolver_once_without_altering_request(self):
        result = resolution(ResolutionStatus.SEARCH_RESULTS, candidates=(candidate(),))
        coordinator, parser, resolver = self.coordinator(parsed(), result)
        outcome = coordinator.handle_structure_request("search MoS2", self.temp.name, "one")
        self.assertEqual((parser.calls, resolver.calls), (1, 1))
        self.assertIs(resolver.requests[0], REQUEST)
        self.assertEqual(outcome.request, REQUEST)

    def test_parse_error_and_clarification_do_not_resolve(self):
        for parse_result in (
            StructureRequestParseResult(status="parse_error", error="bad JSON", retry_count=2),
            StructureRequestParseResult(
                status="clarification_required", clarification="Search or select?", retry_count=0
            ),
        ):
            coordinator, parser, resolver = self.coordinator(parse_result, None)
            outcome = coordinator.handle_structure_request("text", self.temp.name, str(parse_result.status))
            self.assertEqual(parser.calls, 1)
            self.assertEqual(resolver.calls, 0)
            self.assertEqual(outcome.parser_status, parse_result.status)

    def test_search_renders_ordered_candidates_and_writes_nothing(self):
        candidates = (candidate("mp-2815"), candidate("mp-9999"))
        coordinator, _, _ = self.coordinator(
            parsed(), resolution(ResolutionStatus.SEARCH_RESULTS, candidates=candidates)
        )
        outcome = coordinator.handle_structure_request("search", self.temp.name, "search")
        self.assertLess(outcome.rendered_response.index("mp-2815"), outcome.rendered_response.index("mp-9999"))
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_selected_preserves_id_and_path(self):
        selected = candidate()
        result = resolution(
            ResolutionStatus.SELECTED,
            selected=selected,
            candidates=(selected,),
            structure_path=str(Path(self.temp.name) / "mp-2815.cif"),
        )
        coordinator, _, _ = self.coordinator(parsed(), result)
        outcome = coordinator.handle_structure_request("get", self.temp.name, "selected")
        self.assertEqual(outcome.selected.material_id, "mp-2815")
        self.assertEqual(outcome.structure_path, result.structure_path)

    def test_ambiguity_is_authoritative_and_not_retried(self):
        groups = (
            EquivalenceGroupResult(group_index=0, representative_material_id="mp-1", member_material_ids=("mp-1",), energy_above_hull=0),
            EquivalenceGroupResult(group_index=1, representative_material_id="mp-2", member_material_ids=("mp-2",), energy_above_hull=0),
        )
        coordinator, parser, resolver = self.coordinator(
            parsed(), resolution(ResolutionStatus.AMBIGUOUS, equivalence_groups=groups)
        )
        outcome = coordinator.handle_structure_request("get", self.temp.name, "ambiguous")
        self.assertEqual(outcome.resolver_status, ResolutionStatus.AMBIGUOUS)
        self.assertEqual((parser.calls, resolver.calls), (1, 1))

    def test_unsupported_and_failure_statuses_remain_authoritative(self):
        for status in (
            ResolutionStatus.UNSUPPORTED_SEMANTIC,
            ResolutionStatus.NO_MATCHES,
            ResolutionStatus.INSUFFICIENT_DATA,
            ResolutionStatus.API_ERROR,
            ResolutionStatus.DOWNLOAD_ERROR,
        ):
            kwargs = {"error": "failure"} if status in {ResolutionStatus.API_ERROR, ResolutionStatus.DOWNLOAD_ERROR} else {}
            coordinator, _, resolver = self.coordinator(parsed(), resolution(status, **kwargs))
            outcome = coordinator.handle_structure_request("request", self.temp.name, status.value)
            self.assertEqual(outcome.resolver_status, status)
            self.assertEqual(resolver.calls, 1)

    def test_same_key_is_cached_and_different_keys_are_independent(self):
        coordinator, parser, resolver = self.coordinator(
            parsed(), resolution(ResolutionStatus.SEARCH_RESULTS, candidates=(candidate(),))
        )
        first = coordinator.handle_structure_request("same", self.temp.name, "key-1")
        second = coordinator.handle_structure_request("different text", self.temp.name, "key-1")
        third = coordinator.handle_structure_request("same", self.temp.name, "key-2")
        self.assertFalse(first.returned_from_store)
        self.assertTrue(second.returned_from_store)
        self.assertFalse(third.returned_from_store)
        self.assertEqual((parser.calls, resolver.calls), (2, 2))

    def test_concurrent_same_key_executes_once(self):
        parser = FakeParser(parsed())
        resolver = FakeResolver(resolution(ResolutionStatus.SEARCH_RESULTS, candidates=(candidate(),)))
        coordinator = StructureRequestCoordinator(parser, lambda _path: resolver)
        start = threading.Barrier(3)
        outcomes = []

        def run():
            start.wait()
            outcomes.append(coordinator.handle_structure_request("same", self.temp.name, "shared"))

        threads = [threading.Thread(target=run) for _ in range(2)]
        for thread in threads:
            thread.start()
        start.wait()
        for thread in threads:
            thread.join(timeout=3)
        self.assertEqual((parser.calls, resolver.calls), (1, 1))
        self.assertEqual(sorted(item.returned_from_store for item in outcomes), [False, True])

    def test_exception_is_propagated_and_not_cached(self):
        parser = FakeParser(parsed())
        resolver = FakeResolver(exception=RuntimeError("boom"))
        coordinator = StructureRequestCoordinator(parser, lambda _path: resolver)
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "boom"):
                coordinator.handle_structure_request("same", self.temp.name, "failed")
        self.assertEqual((parser.calls, resolver.calls), (2, 2))

    def test_rendering_is_byte_identical(self):
        parse = parsed()
        result = resolution(ResolutionStatus.SEARCH_RESULTS, candidates=(candidate(),))
        first = render_structure_result(parse, result).encode("utf-8")
        second = render_structure_result(parse, result).encode("utf-8")
        self.assertEqual(first, second)

    def test_resolver_factory_receives_output_directory(self):
        observed = []
        parser = FakeParser(parsed())
        resolver = FakeResolver(resolution(ResolutionStatus.SEARCH_RESULTS))
        coordinator = StructureRequestCoordinator(
            parser, lambda path: observed.append(path) or resolver
        )
        coordinator.handle_structure_request("search", self.temp.name, "path")
        self.assertEqual(observed, [Path(self.temp.name)])

    def test_store_is_bounded(self):
        store = InMemoryInvocationStore(capacity=1)
        self.assertEqual(store.execute_once("a", lambda: 1), (1, False))
        self.assertEqual(store.execute_once("b", lambda: 2), (2, False))
        self.assertEqual(store.execute_once("a", lambda: 3), (3, False))


if __name__ == "__main__":
    unittest.main()
