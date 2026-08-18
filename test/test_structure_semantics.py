from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from pymatgen.analysis.prototypes import AFLOW_PROTOTYPE_LIBRARY
from pymatgen.core import Lattice, Structure

from vaspilot.tools.structure_models import ResolutionStatus, StructureRequest
from vaspilot.tools.structure_resolver import StructureResolver
from vaspilot.tools.structure_semantics import (
    MOLYBDENITE_AFLOW_TAG,
    SemanticCandidateStatus,
    StructureSemanticRecognizer,
)


def prototype(aflow_tag: str, substitutions=None) -> Structure:
    entry = next(
        item
        for item in AFLOW_PROTOTYPE_LIBRARY
        if item["tags"].get("aflow") == aflow_tag
    )
    structure = entry["snl"].structure.copy()
    if substitutions:
        structure.replace_species(substitutions)
    return structure


def document(material_id: str, structure: Structure, energy=0.0):
    return SimpleNamespace(
        material_id=material_id,
        structure=structure,
        nsites=len(structure),
        energy_above_hull=energy,
        formation_energy_per_atom=-1.0,
        deprecated=False,
        is_stable=energy == 0,
        theoretical=False,
    )


def relaxed_anatase() -> Structure:
    """A locally fixed mp-390-like conventional cell."""

    ti = [
        [0.5, 0.5, 0.5],
        [0.5, 0.0, 0.75],
        [0.0, 0.0, 0.0],
        [0.0, 0.5, 0.25],
    ]
    oxygen = [
        [0.0, 0.5, 0.457152125],
        [0.5, 0.5, 0.707152125],
        [0.5, 0.0, 0.542847875],
        [0.0, 0.0, 0.792847875],
        [0.5, 0.0, 0.957152125],
        [0.0, 0.0, 0.207152125],
        [0.0, 0.5, 0.042847875],
        [0.5, 0.5, 0.292847875],
    ]
    return Structure(
        Lattice.tetragonal(3.7825396799, 9.6150215748),
        ["Ti"] * 4 + ["O"] * 8,
        ti + oxygen,
    )


class FakeSummary:
    def __init__(self, documents):
        self.documents = documents
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return list(self.documents)


class FakeMPRester:
    def __init__(self, summary):
        self.materials = SimpleNamespace(summary=summary)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class StructureSemanticTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recognizer = StructureSemanticRecognizer()

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def resolver(self, documents):
        summary = FakeSummary(documents)
        resolver = StructureResolver(
            "unused",
            self.temp.name,
            mpr_factory=lambda _key: FakeMPRester(summary),
            semantic_recognizer=self.recognizer,
        )
        return resolver, summary

    @staticmethod
    def search_request(formula, label, limit=20):
        return StructureRequest(
            operation="search",
            selection="return_all",
            formula=formula,
            semantic_label=label,
            result_limit=limit,
        )

    @staticmethod
    def select_request(formula, label):
        return StructureRequest(
            operation="select",
            selection="most_stable",
            formula=formula,
            semantic_label=label,
        )

    def assert_matches(self, label, structure, expected=True):
        plan = self.recognizer.plan(label)
        self.assertIsNotNone(plan)
        result = self.recognizer.match(plan, structure)
        expected_status = (
            SemanticCandidateStatus.MATCH if expected else SemanticCandidateStatus.NO_MATCH
        )
        self.assertEqual(result.status, expected_status)

    def test_rocksalt_acceptance_and_rejection(self):
        rocksalt = prototype("AB_cF8_225_a_b", {"Na": "Zn", "Cl": "S"})
        wurtzite = prototype("AB_hP4_186_b_b")
        self.assert_matches("rocksalt", rocksalt)
        self.assert_matches("rock salt", wurtzite, False)

    def test_wurtzite_and_zincblende_are_distinct(self):
        wurtzite = prototype("AB_hP4_186_b_b")
        zincblende = prototype("AB_cF8_216_c_a")
        self.assert_matches("wurtzite", wurtzite)
        self.assert_matches("wurtzite", zincblende, False)
        self.assert_matches("zincblende", zincblende)
        self.assert_matches("zinc blende", wurtzite, False)

    def test_rutile_and_anatase_are_distinct(self):
        rutile = prototype("A2B_tP6_136_f_a")
        anatase = prototype("A2B_tI12_141_e_a")
        self.assert_matches("rutile", rutile)
        self.assert_matches("rutile", anatase, False)
        self.assert_matches("anatase", anatase)
        self.assert_matches("anatase", rutile, False)

    def test_relaxed_anatase_uses_exact_aflow_label_fallback(self):
        structure = relaxed_anatase()
        plan = self.recognizer.plan("anatase")
        result = self.recognizer.match(plan, structure)
        self.assertEqual(result.status, SemanticCandidateStatus.MATCH)
        self.assertEqual(result.match_method.value, "aflow_protostructure_label")

    def test_cubic_and_orthorhombic_perovskites_are_distinct(self):
        cubic = prototype("AB3C_cP5_221_a_c_b")
        orthorhombic = prototype("AB3C_oP20_62_c_cd_a")
        self.assert_matches("cubic perovskite", cubic)
        self.assert_matches("cubic perovskite", orthorhombic, False)
        self.assert_matches("orthorhombic perovskite", orthorhombic)
        self.assert_matches("orthorhombic perovskite", cubic, False)

    def test_alias_normalization(self):
        self.assertEqual(self.recognizer.plan(" Rock-salt ").prototype.aflow_tag, "AB_cF8_225_a_b")
        self.assertEqual(self.recognizer.plan("ZINCBLENDE").prototype.aflow_tag, "AB_cF8_216_c_a")
        self.assertEqual(self.recognizer.plan("halite").prototype.aflow_tag, "AB_cF8_225_a_b")

    def test_molybdenite_and_2h_family_matching(self):
        mos2 = prototype(MOLYBDENITE_AFLOW_TAG)
        ws2 = prototype(MOLYBDENITE_AFLOW_TAG, {"Mo": "W"})
        mose2 = prototype(MOLYBDENITE_AFLOW_TAG, {"S": "Se"})
        self.assert_matches("molybdenite", mos2)
        self.assert_matches("2H", mos2)
        self.assert_matches("2h", ws2)
        self.assert_matches("2H", mose2)

    def test_1t_like_and_wrong_spacegroup_194_are_rejected(self):
        one_t = prototype("AB2_hP3_164_a_d", {"Cd": "Mo", "I": "S"})
        wrong_194 = prototype("AB2_hP6_194_b_f", {"Ca": "Mo", "In": "S"})
        self.assert_matches("2H", one_t, False)
        self.assert_matches("2H", wrong_194, False)

    def test_non_tmd_molybdenite_geometry_is_incompatible(self):
        non_tmd = prototype(MOLYBDENITE_AFLOW_TAG, {"Mo": "Na", "S": "Cl"})
        plan = self.recognizer.plan("2H")
        result = self.recognizer.match(plan, non_tmd)
        self.assertEqual(result.status, SemanticCandidateStatus.INCOMPATIBLE_FAMILY)

    def test_unknown_1t_and_3r_are_unsupported_without_api(self):
        for label in ("unknown phase", "1T", "3R"):
            resolver, summary = self.resolver([document("mp-1", prototype(MOLYBDENITE_AFLOW_TAG))])
            result = resolver.resolve(self.search_request("MoS2", label))
            self.assertEqual(result.status, ResolutionStatus.UNSUPPORTED_SEMANTIC)
            self.assertEqual(summary.calls, [])

    def test_2h_outside_mx2_family_is_unsupported(self):
        resolver, summary = self.resolver([document("mp-1", prototype("A2B_tP6_136_f_a"))])
        result = resolver.resolve(self.search_request("TiO2", "2H"))
        self.assertEqual(result.status, ResolutionStatus.UNSUPPORTED_SEMANTIC)
        self.assertIn("MX2", result.error)
        self.assertEqual(summary.calls, [])

    def test_supported_semantic_with_no_match(self):
        anatase = prototype("A2B_tI12_141_e_a")
        resolver, _ = self.resolver([document("mp-1", anatase)])
        result = resolver.resolve(self.search_request("TiO2", "rutile"))
        self.assertEqual(result.status, ResolutionStatus.NO_MATCHES)

    def test_semantic_search_is_order_independent_and_writes_nothing(self):
        rutile1 = document("mp-20", prototype("A2B_tP6_136_f_a"), 0.2)
        rutile2 = document("mp-3", prototype("A2B_tP6_136_f_a"), 0.1)
        first, _ = self.resolver([rutile1, rutile2])
        result1 = first.resolve(self.search_request("TiO2", "rutile"))
        second, _ = self.resolver([rutile2, rutile1])
        result2 = second.resolve(self.search_request("TiO2", "rutile"))
        ids1 = [item.material_id for item in result1.candidates]
        ids2 = [item.material_id for item in result2.candidates]
        self.assertEqual(ids1, ["mp-3", "mp-20"])
        self.assertEqual(ids1, ids2)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])
        self.assertEqual(result1.candidates[0].semantic_match.matched_aflow_tag, "A2B_tP6_136_f_a")

    def test_semantic_most_stable_ranks_only_matches_and_writes_one(self):
        rutile = document("mp-10", prototype("A2B_tP6_136_f_a"), 0.2)
        anatase = document("mp-2", prototype("A2B_tI12_141_e_a"), 0.0)
        resolver, _ = self.resolver([anatase, rutile])
        result = resolver.resolve(self.select_request("TiO2", "rutile"))
        self.assertEqual(result.status, ResolutionStatus.SELECTED)
        self.assertEqual(result.selected.material_id, "mp-10")
        self.assertEqual(result.selected.semantic_match.normalized_label, "rutile")
        self.assertEqual(len(list(Path(self.temp.name).iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
