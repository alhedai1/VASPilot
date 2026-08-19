from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from vaspilot.tools.structure_application_boundary import StructureApplicationBoundary
from vaspilot.tools.structure_models import (
    CandidateResult,
    ResolutionStatus,
    StructureRequest,
    StructureResolutionResult,
    SymmetrySummary,
)
from vaspilot.tools.structure_request_applicability import StructureApplicabilityResult
from vaspilot.tools.structure_request_coordinator import StructureCoordinatorResult
from vaspilot.tools.structure_request_parser import StructureRequestParseResult


class FakeClassifier:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    def classify(self, _text):
        self.calls += 1
        return self.result


class FakeCoordinator:
    def __init__(self, result=None):
        self.result = result
        self.calls = 0
        self.arguments = []

    def handle_structure_request(self, text, output, key):
        self.calls += 1
        self.arguments.append((text, Path(output), key))
        return self.result


PARSE_ERROR_RESULT = StructureRequestParseResult(
    status="parse_error", error="test", retry_count=0
)
COORDINATOR_RESULT = StructureCoordinatorResult(
    invocation_key="id",
    parser_result=PARSE_ERROR_RESULT,
    parser_status="parse_error",
    parser_retry_count=0,
    rendered_response="search_results: 1 candidate(s)\n- mp-2815",
    coordinator_policy_version="test",
)


def selected_coordinator_result(path: Path) -> StructureCoordinatorResult:
    request = StructureRequest(
        operation="select", selection="require_unique", formula="MoS2"
    )
    parse = StructureRequestParseResult(status="parsed", request=request, retry_count=0)
    candidate = CandidateResult(
        material_id="mp-2815",
        formula="MoS2",
        chemical_system="Mo-S",
        nsites=6,
        energy_above_hull=0.0,
        formation_energy_per_atom=-1.5,
        is_stable=True,
        theoretical=False,
        deprecated=False,
        symmetry=SymmetrySummary(
            symbol="P6_3/mmc",
            number=194,
            crystal_system="hexagonal",
            point_group="6/mmm",
        ),
    )
    resolution = StructureResolutionResult(
        status=ResolutionStatus.SELECTED,
        request=request,
        candidates=(candidate,),
        selected=candidate,
        structure_path=str(path),
        resolver_policy_version="resolver-test-v1",
    )
    return StructureCoordinatorResult(
        invocation_key="mixed",
        parser_result=parse,
        resolution_result=resolution,
        request=request,
        parser_status="parsed",
        resolver_status="selected",
        selected=candidate,
        structure_path=str(path),
        candidates=(candidate,),
        parser_retry_count=0,
        resolver_policy_version="resolver-test-v1",
        rendered_response=f"selected: mp-2815\nstructure_path: {path}",
    )


def stopped_coordinator_result(status: str) -> StructureCoordinatorResult:
    request = StructureRequest(
        operation="select", selection="require_unique", formula="MoS2"
    )
    parse = StructureRequestParseResult(status="parsed", request=request, retry_count=0)
    resolution = StructureResolutionResult(
        status=status,
        request=request,
        resolver_policy_version="resolver-test-v1",
    )
    return StructureCoordinatorResult(
        invocation_key="mixed",
        parser_result=parse,
        resolution_result=resolution,
        request=request,
        parser_status="parsed",
        resolver_status=status,
        parser_retry_count=0,
        resolver_policy_version="resolver-test-v1",
        rendered_response=f"{status}: deterministic stop",
    )


class BoundaryTests(unittest.TestCase):
    def test_pure_calls_coordinator_once(self):
        classification = StructureApplicabilityResult(
            status="pure_mp_structure",
            evidence={"retrieval_intent": "search", "material_anchor": "MoS2"},
            retry_count=0,
        )
        coordinator = FakeCoordinator(COORDINATOR_RESULT)
        boundary = StructureApplicationBoundary(FakeClassifier(classification), coordinator)
        result = boundary.handle("search MoS2", ".", "id")
        self.assertFalse(result.should_run_crewai)
        self.assertEqual(coordinator.calls, 1)
        self.assertEqual(result.rendered_response, COORDINATOR_RESULT.rendered_response)

    def test_not_pure_does_not_call_coordinator(self):
        classification = StructureApplicabilityResult(status="not_pure", retry_count=0)
        coordinator = FakeCoordinator()
        result = StructureApplicationBoundary(
            FakeClassifier(classification), coordinator
        ).handle("calculate bands", ".", "id")
        self.assertTrue(result.should_run_crewai)
        self.assertEqual(coordinator.calls, 0)

    def test_same_invocation_is_fully_idempotent(self):
        classification = StructureApplicabilityResult(
            status="pure_mp_structure",
            evidence={"retrieval_intent": "search", "material_anchor": "MoS2"},
            retry_count=0,
        )
        classifier = FakeClassifier(classification)
        coordinator = FakeCoordinator(COORDINATOR_RESULT)
        boundary = StructureApplicationBoundary(classifier, coordinator)
        first = boundary.handle("search MoS2", ".", "same-id")
        second = boundary.handle("search MoS2", ".", "same-id")
        self.assertEqual(first, second)
        self.assertEqual(classifier.calls, 1)
        self.assertEqual(coordinator.calls, 1)

    def test_clarification_and_error_are_deterministic_and_do_not_continue(self):
        cases = (
            StructureApplicabilityResult(
                status="clarification_required", clarification="Search or calculate?", retry_count=0
            ),
            StructureApplicabilityResult(
                status="classification_error", error="invalid classifier output", retry_count=2
            ),
        )
        for classification in cases:
            coordinator = FakeCoordinator()
            result = StructureApplicationBoundary(
                FakeClassifier(classification), coordinator
            ).handle("text", ".", classification.status.value)
            self.assertFalse(result.should_run_crewai)
            self.assertEqual(coordinator.calls, 0)
            self.assertTrue(result.rendered_response.startswith(classification.status.value))

    def test_api_key_cannot_appear_without_a_dependency_leaking_it(self):
        classification = StructureApplicabilityResult(status="not_pure", retry_count=0)
        result = StructureApplicationBoundary(
            FakeClassifier(classification), FakeCoordinator()
        ).handle("ordinary task", ".", "id")
        self.assertNotIn("MP_API_KEY", result.model_dump_json())

    def test_mixed_selected_builds_frozen_context_and_runs_once(self):
        classification = StructureApplicabilityResult(
            status="mixed_mp_structure",
            evidence={"retrieval_intent": "relax", "material_anchor": "MoS2"},
            retry_count=0,
        )
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "mp-2815.cif"
            path.write_text("fixture", encoding="utf-8")
            mixed = FakeCoordinator(selected_coordinator_result(path))
            boundary = StructureApplicationBoundary(
                FakeClassifier(classification), FakeCoordinator(), mixed_coordinator=mixed
            )
            first = boundary.handle("relax MoS2", td, "mixed")
            second = boundary.handle("relax MoS2", td, "mixed")
            self.assertTrue(first.should_run_crewai)
            self.assertEqual(first, second)
            self.assertEqual(mixed.calls, 1)
            self.assertEqual(first.resolved_structure.material_id, "mp-2815")
            self.assertEqual(first.resolved_structure.structure_path, str(path.resolve()))
            with self.assertRaises(Exception):
                first.resolved_structure.material_id = "mp-other"

    def test_mixed_scientific_stop_never_runs_crewai(self):
        classification = StructureApplicabilityResult(
            status="mixed_mp_structure",
            evidence={"retrieval_intent": "calculate", "material_anchor": "MoS2"},
            retry_count=0,
        )
        for status in ("ambiguous", "unsupported_semantic", "insufficient_data"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as td:
                mixed = FakeCoordinator(stopped_coordinator_result(status))
                result = StructureApplicationBoundary(
                    FakeClassifier(classification),
                    FakeCoordinator(),
                    mixed_coordinator=mixed,
                ).handle("calculate bands of MoS2", td, "mixed")
                self.assertFalse(result.should_run_crewai)
                self.assertIsNone(result.resolved_structure)
                self.assertTrue(result.rendered_response.startswith(status))

    def test_selected_path_outside_conversation_directory_is_rejected(self):
        classification = StructureApplicabilityResult(
            status="mixed_mp_structure",
            evidence={"retrieval_intent": "relax", "material_anchor": "MoS2"},
            retry_count=0,
        )
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as outside:
            path = Path(outside) / "mp-2815.cif"
            path.write_text("fixture", encoding="utf-8")
            mixed = FakeCoordinator(selected_coordinator_result(path))
            boundary = StructureApplicationBoundary(
                FakeClassifier(classification), FakeCoordinator(), mixed_coordinator=mixed
            )
            with self.assertRaisesRegex(ValueError, "outside conversation directory"):
                boundary.handle("relax MoS2", root, "mixed")


if __name__ == "__main__":
    unittest.main()
