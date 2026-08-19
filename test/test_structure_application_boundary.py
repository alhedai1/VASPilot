from __future__ import annotations

import unittest
from pathlib import Path

from vaspilot.tools.structure_application_boundary import StructureApplicationBoundary
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


if __name__ == "__main__":
    unittest.main()
