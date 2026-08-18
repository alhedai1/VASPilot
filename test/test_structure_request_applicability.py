from __future__ import annotations

import json
import unittest

from vaspilot.tools.structure_request_applicability import (
    ApplicabilityStatus,
    StructureRequestApplicabilityClassifier,
)


PURE_EXAMPLES = (
    "search for the structure of 2H phase of MoS2",
    "list stable TiO2 structures from Materials Project",
    "get the most stable MoS2 structure",
    "retrieve mp-2815",
    "find rutile TiO2 structures",
    "show hexagonal MoS2 candidates in space group 194",
    "download the wurtzite ZnO structure",
)

NOT_PURE_EXAMPLES = (
    "calculate the band structure of MoS2",
    "calculate the band structure from vasprun.xml",
    "relax 2H-MoS2 using VASP",
    "run a geometry optimization for mp-2815",
    "analyze structure_file: C:/data/sample.cif",
    "make a 2x2x2 supercell from POSCAR",
    "create an FCC aluminum structure",
    "explain what a crystal structure is",
    "search the literature for MoS2",
    "plot the density of states",
)


def pure_json(text: str) -> str:
    words = text.split()
    intent = words[0]
    anchor = next(
        word for word in reversed(words) if word.lower() not in {intent.lower(), "structure", "structures"}
    )
    return json.dumps(
        {
            "status": "pure_mp_structure",
            "evidence": {"retrieval_intent": intent, "material_anchor": anchor},
            "clarification": None,
        }
    )


class SequenceLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def __call__(self, _messages):
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


class ApplicabilityTests(unittest.TestCase):
    def test_required_positive_examples(self):
        for text in PURE_EXAMPLES:
            with self.subTest(text=text):
                classifier = StructureRequestApplicabilityClassifier(lambda _m, t=text: pure_json(t))
                result = classifier.classify(text)
                self.assertEqual(result.status, ApplicabilityStatus.PURE_MP_STRUCTURE)

    def test_required_negative_examples(self):
        response = json.dumps(
            {"status": "not_pure", "evidence": None, "clarification": None}
        )
        for text in NOT_PURE_EXAMPLES:
            with self.subTest(text=text):
                result = StructureRequestApplicabilityClassifier(
                    lambda _messages: response
                ).classify(text)
                self.assertEqual(result.status, ApplicabilityStatus.NOT_PURE)

    def test_band_structure_word_and_local_path_are_not_intercepted(self):
        response = '{"status":"not_pure","evidence":null,"clarification":null}'
        classifier = StructureRequestApplicabilityClassifier(lambda _m: response)
        self.assertEqual(
            classifier.classify("calculate band structure").status,
            ApplicabilityStatus.NOT_PURE,
        )

    def test_ungroundable_retrieval_common_names_require_clarification(self):
        response = '{"status":"not_pure","evidence":null,"clarification":null}'
        classifier = StructureRequestApplicabilityClassifier(lambda _m: response)
        for text in (
            "get the structure of water",
            "get the structure of sodium chloride",
            "retrieve the crystal structure of an unsupported common material name",
        ):
            with self.subTest(text=text):
                result = classifier.classify(text)
                self.assertEqual(
                    result.status, ApplicabilityStatus.CLARIFICATION_REQUIRED
                )
                self.assertIn("isolated molecule", result.clarification)
                self.assertIn("crystalline", result.clarification)

    def test_erroneous_pure_common_name_is_defensively_changed_to_clarification(self):
        response = json.dumps(
            {
                "status": "pure_mp_structure",
                "evidence": {
                    "retrieval_intent": "get",
                    "material_anchor": "water",
                },
                "clarification": None,
            }
        )
        result = StructureRequestApplicabilityClassifier(lambda _m: response).classify(
            "get the structure of water"
        )
        self.assertEqual(result.status, ApplicabilityStatus.CLARIFICATION_REQUIRED)
        self.assertIn("isolated molecule", result.clarification)

    def test_weak_llm_clarification_is_replaced_for_ambiguous_retrieval(self):
        response = json.dumps(
            {
                "status": "clarification_required",
                "evidence": None,
                "clarification": "Please provide a formula.",
            }
        )
        result = StructureRequestApplicabilityClassifier(lambda _m: response).classify(
            "get the structure of water"
        )
        self.assertIn("isolated molecule", result.clarification)
        self.assertIn("crystalline material/phase", result.clarification)

    def test_formula_retrieval_is_not_changed_by_fallback(self):
        pure = pure_json("search Materials Project for H2O structures")
        result = StructureRequestApplicabilityClassifier(lambda _m: pure).classify(
            "search Materials Project for H2O structures"
        )
        self.assertEqual(result.status, ApplicabilityStatus.PURE_MP_STRUCTURE)

    def test_explicit_mp_formula_query_overrides_unnecessary_llm_clarification(self):
        response = json.dumps(
            {
                "status": "clarification_required",
                "evidence": None,
                "clarification": "Please provide a formula.",
            }
        )
        result = StructureRequestApplicabilityClassifier(lambda _m: response).classify(
            "search Materials Project for H2O structures"
        )
        self.assertEqual(result.status, ApplicabilityStatus.PURE_MP_STRUCTURE)
        self.assertEqual(result.evidence.material_anchor, "H2O")

    def test_creation_explanation_and_band_work_are_excluded_from_fallback(self):
        response = '{"status":"not_pure","evidence":null,"clarification":null}'
        classifier = StructureRequestApplicabilityClassifier(lambda _m: response)
        for text in (
            "create an isolated H2O molecule",
            "calculate the band structure from vasprun.xml",
            "explain the structure of water",
        ):
            with self.subTest(text=text):
                self.assertEqual(
                    classifier.classify(text).status, ApplicabilityStatus.NOT_PURE
                )
        self.assertEqual(
            classifier.classify("analyze C:/data/structure.cif").status,
            ApplicabilityStatus.NOT_PURE,
        )

    def test_malformed_output_retries_then_succeeds(self):
        llm = SequenceLLM(["not json", pure_json("retrieve mp-2815")])
        result = StructureRequestApplicabilityClassifier(llm).classify("retrieve mp-2815")
        self.assertEqual(result.status, ApplicabilityStatus.PURE_MP_STRUCTURE)
        self.assertEqual(result.retry_count, 1)
        self.assertEqual(llm.calls, 2)

    def test_retry_exhaustion_is_explicit(self):
        llm = SequenceLLM(["bad"])
        result = StructureRequestApplicabilityClassifier(llm).classify("retrieve mp-2815")
        self.assertEqual(result.status, ApplicabilityStatus.CLASSIFICATION_ERROR)
        self.assertEqual(result.retry_count, 2)
        self.assertEqual(llm.calls, 3)

    def test_inexact_or_missing_pure_evidence_is_rejected(self):
        response = json.dumps(
            {
                "status": "pure_mp_structure",
                "evidence": {
                    "retrieval_intent": "download",
                    "material_anchor": "invented-material",
                },
                "clarification": None,
            }
        )
        result = StructureRequestApplicabilityClassifier(
            lambda _m: response, max_corrective_retries=0
        ).classify("download MoS2")
        self.assertEqual(result.status, ApplicabilityStatus.CLASSIFICATION_ERROR)


if __name__ == "__main__":
    unittest.main()
