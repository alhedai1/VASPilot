from __future__ import annotations

import json
import unittest

from vaspilot.tools.structure_request_parser import ParserStatus, StructureRequestParser


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def parsed(request, evidence):
    return json.dumps({"status": "parsed", "request": request, "evidence": evidence})


def ev(field, quote):
    return {"field": field, "quote": quote}


class StructureRequestParserTests(unittest.TestCase):
    def parse_once(self, text, request, evidence):
        fake = FakeLLM([parsed(request, evidence)])
        result = StructureRequestParser(fake).parse(text)
        self.assertEqual(result.status, ParserStatus.PARSED, result.error)
        self.assertEqual(len(fake.calls), 1)
        return result

    def test_2h_mos2_only_formula_and_semantic_label(self):
        text = "search for the structure of 2H phase of MoS2"
        result = self.parse_once(
            text,
            {"operation": "search", "selection": "return_all", "formula": "MoS2", "semantic_label": "2H"},
            [ev("operation", "search"), ev("selection", "search"), ev("formula", "MoS2"), ev("semantic_label", "2H")],
        )
        self.assertEqual(result.request.formula, "MoS2")
        self.assertEqual(result.request.semantic_label, "2H")
        self.assertIsNone(result.request.spacegroup_number)
        self.assertIsNone(result.request.crystal_system)
        self.assertIsNone(result.request.num_sites)

    def test_generic_semantic_wrapper_is_normalized_without_inference(self):
        text = "search for the structure of 2H phase of MoS2"
        result = self.parse_once(
            text,
            {"operation": "search", "selection": "return_all", "formula": "MoS2", "semantic_label": "2H phase"},
            [ev("operation", "search"), ev("selection", "search"), ev("formula", "MoS2"), ev("semantic_label", "2H phase")],
        )
        self.assertEqual(result.request.semantic_label, "2H")
        self.assertIsNone(result.request.spacegroup_number)

    def test_most_stable_mos2(self):
        text = "get the most stable MoS2 structure"
        result = self.parse_once(
            text,
            {"operation": "select", "selection": "most_stable", "formula": "MoS2"},
            [ev("operation", "get"), ev("selection", "most stable"), ev("formula", "MoS2")],
        )
        self.assertEqual(result.request.selection.value, "most_stable")

    def test_underspecified_single_selection(self):
        text = "get an MoS2 structure"
        result = self.parse_once(
            text,
            {"operation": "select", "selection": "require_unique", "formula": "MoS2"},
            [ev("operation", "get"), ev("selection", "get"), ev("formula", "MoS2")],
        )
        self.assertEqual(result.request.selection.value, "require_unique")

    def test_explicit_space_group_and_site_limit(self):
        text = "Find hexagonal MoS2 in space group 194 with at most 12 sites"
        result = self.parse_once(
            text,
            {"operation": "search", "selection": "return_all", "formula": "MoS2", "crystal_system": "hexagonal", "spacegroup_number": 194, "num_sites": {"maximum": 12}},
            [ev("operation", "Find"), ev("selection", "Find"), ev("formula", "MoS2"), ev("crystal_system", "hexagonal"), ev("spacegroup_number", "space group 194"), ev("num_sites.maximum", "at most 12 sites")],
        )
        self.assertEqual(result.request.spacegroup_number, 194)
        self.assertEqual(result.request.num_sites.maximum, 12)

    def test_explicit_mp_id(self):
        text = "Retrieve mp-2815"
        result = self.parse_once(
            text,
            {"operation": "select", "selection": "require_unique", "material_ids": ["mp-2815"]},
            [ev("operation", "Retrieve"), ev("selection", "Retrieve"), ev("material_ids", "mp-2815")],
        )
        self.assertEqual(result.request.material_ids, ("mp-2815",))

    def test_formula_normalization(self):
        text = "search for Mo2S4 structures"
        result = self.parse_once(
            text,
            {"operation": "search", "selection": "return_all", "formula": "Mo2S4"},
            [ev("operation", "search"), ev("selection", "structures"), ev("formula", "Mo2S4")],
        )
        self.assertEqual(result.request.formula, "MoS2")

    def test_included_and_excluded_elements(self):
        text = "list Na-Cl structures containing Na but excluding K"
        result = self.parse_once(
            text,
            {"operation": "search", "selection": "return_all", "chemical_system": ["Na", "Cl"], "include_elements": ["Na"], "exclude_elements": ["K"]},
            [ev("operation", "list"), ev("selection", "structures"), ev("chemical_system", "Na-Cl"), ev("include_elements", "containing Na"), ev("exclude_elements", "excluding K")],
        )
        self.assertEqual(result.request.exclude_elements, ("K",))

    def test_stable_experimental_structures(self):
        text = "search stable experimental MoS2 structures"
        result = self.parse_once(
            text,
            {"operation": "search", "selection": "return_all", "formula": "MoS2", "stable_only": True, "theoretical": "only_experimental"},
            [ev("operation", "search"), ev("selection", "structures"), ev("formula", "MoS2"), ev("stable_only", "stable"), ev("theoretical", "experimental")],
        )
        self.assertTrue(result.request.stable_only)
        self.assertEqual(result.request.theoretical.value, "only_experimental")

    def test_explicit_energy_range(self):
        text = "search MoS2 with energy above hull from 0.01 to 0.05 eV/atom"
        result = self.parse_once(
            text,
            {"operation": "search", "selection": "return_all", "formula": "MoS2", "energy_above_hull": {"minimum": 0.01, "maximum": 0.05}},
            [ev("operation", "search"), ev("selection", "search"), ev("formula", "MoS2"), ev("energy_above_hull.minimum", "from 0.01"), ev("energy_above_hull.maximum", "to 0.05 eV/atom")],
        )
        self.assertEqual(result.request.energy_above_hull.maximum, 0.05)

    def test_explicit_result_limit_and_ordering(self):
        text = "list 5 MoS2 structures ordered by material ID"
        result = self.parse_once(
            text,
            {"operation": "search", "selection": "return_all", "formula": "MoS2", "result_limit": 5, "order_by": "material_id"},
            [ev("operation", "list"), ev("selection", "structures"), ev("formula", "MoS2"), ev("result_limit", "5"), ev("order_by", "ordered by material ID")],
        )
        self.assertEqual(result.request.result_limit, 5)
        self.assertEqual(result.request.order_by.value, "material_id")

    def test_conflicting_intent_returns_clarification(self):
        fake = FakeLLM([json.dumps({"status": "clarification_required", "request": None, "evidence": [], "clarification": "Do you want a list or one structure?"})])
        result = StructureRequestParser(fake).parse("list MoS2 structures but retrieve one")
        self.assertEqual(result.status, ParserStatus.CLARIFICATION_REQUIRED)
        self.assertEqual(result.retry_count, 0)
        self.assertEqual(len(fake.calls), 1)

    def test_invented_constraint_is_rejected_then_corrected(self):
        text = "search for the structure of 2H phase of MoS2"
        invented = parsed(
            {"operation": "search", "selection": "return_all", "formula": "MoS2", "semantic_label": "2H", "spacegroup_number": 194},
            [ev("operation", "search"), ev("selection", "search"), ev("formula", "MoS2"), ev("semantic_label", "2H"), ev("spacegroup_number", "2H")],
        )
        corrected = parsed(
            {"operation": "search", "selection": "return_all", "formula": "MoS2", "semantic_label": "2H"},
            [ev("operation", "search"), ev("selection", "search"), ev("formula", "MoS2"), ev("semantic_label", "2H")],
        )
        fake = FakeLLM([invented, corrected])
        result = StructureRequestParser(fake).parse(text)
        self.assertEqual(result.status, ParserStatus.PARSED)
        self.assertEqual(result.retry_count, 1)
        self.assertIsNone(result.request.spacegroup_number)

    def test_invented_stability_is_rejected_by_literal_evidence(self):
        text = "search MoS2 structures"
        invented = parsed(
            {"operation": "search", "selection": "return_all", "formula": "MoS2", "stable_only": True},
            [ev("operation", "search"), ev("selection", "structures"), ev("formula", "MoS2"), ev("stable_only", "MoS2")],
        )
        corrected = parsed(
            {"operation": "search", "selection": "return_all", "formula": "MoS2"},
            [ev("operation", "search"), ev("selection", "structures"), ev("formula", "MoS2")],
        )
        result = StructureRequestParser(FakeLLM([invented, corrected])).parse(text)
        self.assertEqual(result.status, ParserStatus.PARSED)
        self.assertEqual(result.retry_count, 1)
        self.assertFalse(result.request.stable_only)

    def test_malformed_json_retries(self):
        good = parsed(
            {"operation": "select", "selection": "require_unique", "formula": "MoS2"},
            [ev("operation", "get"), ev("selection", "get"), ev("formula", "MoS2")],
        )
        fake = FakeLLM(["not json", good])
        result = StructureRequestParser(fake).parse("get an MoS2 structure")
        self.assertEqual(result.status, ParserStatus.PARSED)
        self.assertEqual(result.retry_count, 1)

    def test_invalid_pydantic_output_retries(self):
        bad = parsed(
            {"operation": "search", "selection": "require_unique", "formula": "MoS2"},
            [ev("operation", "search"), ev("selection", "search"), ev("formula", "MoS2")],
        )
        good = parsed(
            {"operation": "search", "selection": "return_all", "formula": "MoS2"},
            [ev("operation", "search"), ev("selection", "search"), ev("formula", "MoS2")],
        )
        result = StructureRequestParser(FakeLLM([bad, good])).parse("search MoS2")
        self.assertEqual(result.status, ParserStatus.PARSED)
        self.assertEqual(result.retry_count, 1)

    def test_retry_exhaustion(self):
        fake = FakeLLM(["bad", "still bad", "also bad"])
        result = StructureRequestParser(fake).parse("search MoS2")
        self.assertEqual(result.status, ParserStatus.PARSE_ERROR)
        self.assertEqual(result.retry_count, 2)
        self.assertEqual(len(fake.calls), 3)

    def test_llm_exception_retries_and_returns_parse_error(self):
        fake = FakeLLM([RuntimeError("offline"), RuntimeError("offline"), RuntimeError("offline")])
        result = StructureRequestParser(fake).parse("search MoS2")
        self.assertEqual(result.status, ParserStatus.PARSE_ERROR)
        self.assertEqual(result.retry_count, 2)
        self.assertIn("offline", result.error)

    def test_parser_has_no_materials_project_dependency(self):
        fake = FakeLLM([parsed(
            {"operation": "select", "selection": "require_unique", "material_ids": ["mp-2815"]},
            [ev("operation", "Retrieve"), ev("selection", "Retrieve"), ev("material_ids", "mp-2815")],
        )])
        result = StructureRequestParser(fake).parse("Retrieve mp-2815")
        self.assertEqual(result.status, ParserStatus.PARSED)
        self.assertEqual(len(fake.calls), 1)


if __name__ == "__main__":
    unittest.main()
