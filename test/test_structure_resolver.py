from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pydantic import ValidationError
from pymatgen.core import Lattice, Structure

from vaspilot.tools.structure_models import (
    FloatRange,
    RequestOperation,
    ResolutionStatus,
    SelectionBehavior,
    StructureRequest,
    TheoreticalFilter,
)
from vaspilot.tools.structure_resolver import StructureResolver


def simple_si(a: float = 3.0) -> Structure:
    return Structure(Lattice.cubic(a), ["Si"], [[0, 0, 0]])


def diamond_si(a: float = 5.4) -> Structure:
    return Structure(
        Lattice.cubic(a),
        ["Si", "Si"],
        [[0, 0, 0], [0.25, 0.25, 0.25]],
    )


def nacl(a: float = 5.6) -> Structure:
    return Structure(Lattice.cubic(a), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])


def document(
    material_id: str,
    structure: Structure,
    energy: float | None = 0.0,
    *,
    deprecated: bool = False,
    stable: bool = True,
    theoretical: bool | None = False,
):
    return SimpleNamespace(
        material_id=material_id,
        structure=structure,
        nsites=len(structure),
        energy_above_hull=energy,
        formation_energy_per_atom=-1.0,
        deprecated=deprecated,
        is_stable=stable,
        theoretical=theoretical,
    )


class FakeSummary:
    def __init__(self, pages):
        self.pages = pages
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        if kwargs["num_chunks"] is None:
            return [item for page in self.pages for item in page]
        return self.pages[0]


class FakeMPRester:
    def __init__(self, summary):
        self.materials = SimpleNamespace(summary=summary)

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


class ResolverTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp.cleanup()

    def resolver(self, pages):
        summary = FakeSummary(pages)
        resolver = StructureResolver(
            "unused",
            self.temp.name,
            mpr_factory=lambda _key: FakeMPRester(summary),
        )
        return resolver, summary

    @staticmethod
    def search_request(**kwargs):
        return StructureRequest(
            operation=RequestOperation.SEARCH,
            selection=SelectionBehavior.RETURN_ALL,
            formula="Si",
            **kwargs,
        )

    @staticmethod
    def select_request(selection=SelectionBehavior.REQUIRE_UNIQUE, **kwargs):
        return StructureRequest(
            operation=RequestOperation.SELECT,
            selection=selection,
            formula="Si",
            **kwargs,
        )

    def test_api_order_does_not_change_output_order(self):
        docs = [document("mp-20", simple_si(), 0.2), document("mp-3", simple_si(), 0.1)]
        first, _ = self.resolver([docs])
        second, _ = self.resolver([list(reversed(docs))])
        ids1 = [item.material_id for item in first.resolve(self.search_request()).candidates]
        ids2 = [item.material_id for item in second.resolve(self.search_request()).candidates]
        self.assertEqual(ids1, ["mp-3", "mp-20"])
        self.assertEqual(ids1, ids2)

    def test_all_pages_and_limit_after_sorting(self):
        resolver, summary = self.resolver(
            [[document("mp-10", simple_si(), 0.5)], [document("mp-2", simple_si(), 0.0)]]
        )
        result = resolver.resolve(self.search_request(result_limit=1))
        self.assertEqual([item.material_id for item in result.candidates], ["mp-2"])
        self.assertEqual(result.total_api_records, 2)
        self.assertIsNone(summary.calls[0]["num_chunks"])
        self.assertFalse(summary.calls[0]["deprecated"])
        self.assertFalse(summary.calls[0]["include_gnome"])

    def test_formula_and_symmetry_filters(self):
        resolver, _ = self.resolver([[document("mp-1", simple_si())]])
        result = resolver.resolve(self.search_request(crystal_system="cubic", spacegroup_number=221))
        self.assertEqual(result.status, ResolutionStatus.SEARCH_RESULTS)
        mismatch = resolver.resolve(self.search_request(spacegroup_number=225))
        self.assertEqual(mismatch.status, ResolutionStatus.NO_MATCHES)

    def test_deprecated_rejected_defensively(self):
        resolver, _ = self.resolver([[document("mp-1", simple_si(), deprecated=True)]])
        result = resolver.resolve(self.search_request())
        self.assertEqual(result.status, ResolutionStatus.NO_MATCHES)

    def test_included_and_excluded_elements(self):
        resolver, _ = self.resolver([[document("mp-1", nacl())]])
        request = StructureRequest(
            operation="search",
            selection="return_all",
            chemical_system=("Na", "Cl"),
            include_elements=("Na",),
            exclude_elements=("K",),
        )
        self.assertEqual(resolver.resolve(request).status, ResolutionStatus.SEARCH_RESULTS)
        rejected = request.model_copy(update={"exclude_elements": ("Cl",)})
        self.assertEqual(resolver.resolve(rejected).status, ResolutionStatus.NO_MATCHES)

    def test_stable_and_theoretical_filters(self):
        docs = [
            document("mp-1", simple_si(), stable=False, theoretical=False),
            document("mp-2", simple_si(), stable=True, theoretical=True),
        ]
        resolver, _ = self.resolver([docs])
        request = self.search_request(
            stable_only=True,
            theoretical=TheoreticalFilter.ONLY_THEORETICAL,
        )
        self.assertEqual([c.material_id for c in resolver.resolve(request).candidates], ["mp-2"])

    def test_search_writes_nothing(self):
        resolver, _ = self.resolver([[document("mp-1", simple_si())]])
        result = resolver.resolve(self.search_request())
        self.assertEqual(result.status, ResolutionStatus.SEARCH_RESULTS)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_equivalent_duplicates_select_canonical_id_and_write_once(self):
        resolver, _ = self.resolver(
            [[document("mp-9", simple_si(4.0)), document("mp-2", simple_si(3.0))]]
        )
        result = resolver.resolve(self.select_request())
        self.assertEqual(result.status, ResolutionStatus.SELECTED)
        self.assertEqual(result.selected.material_id, "mp-2")
        self.assertEqual(result.equivalence_groups[0].member_material_ids, ("mp-2", "mp-9"))
        self.assertEqual(len(list(Path(self.temp.name).iterdir())), 1)

    def test_distinct_structures_are_ambiguous_and_write_nothing(self):
        resolver, _ = self.resolver(
            [[document("mp-1", simple_si()), document("mp-2", diamond_si())]]
        )
        result = resolver.resolve(self.select_request())
        self.assertEqual(result.status, ResolutionStatus.AMBIGUOUS)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_unique_most_stable_group(self):
        resolver, _ = self.resolver(
            [[document("mp-1", simple_si(), 0.1), document("mp-2", diamond_si(), 0.0)]]
        )
        result = resolver.resolve(self.select_request(SelectionBehavior.MOST_STABLE))
        self.assertEqual(result.status, ResolutionStatus.SELECTED)
        self.assertEqual(result.selected.material_id, "mp-2")

    def test_distinct_energy_tie_is_ambiguous(self):
        resolver, _ = self.resolver(
            [[document("mp-1", simple_si(), 0.0), document("mp-2", diamond_si(), 5e-7)]]
        )
        result = resolver.resolve(self.select_request(SelectionBehavior.MOST_STABLE))
        self.assertEqual(result.status, ResolutionStatus.AMBIGUOUS)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_missing_energy_handling(self):
        resolver, _ = self.resolver(
            [[document("mp-1", simple_si(), None), document("mp-2", diamond_si(), None)]]
        )
        result = resolver.resolve(self.select_request(SelectionBehavior.MOST_STABLE))
        self.assertEqual(result.status, ResolutionStatus.INSUFFICIENT_DATA)
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

        resolver2, _ = self.resolver(
            [[document("mp-1", simple_si(), None), document("mp-2", diamond_si(), 0.2)]]
        )
        result2 = resolver2.resolve(self.select_request(SelectionBehavior.MOST_STABLE))
        self.assertEqual(result2.selected.material_id, "mp-2")

    def test_explicit_id_and_constraint_mismatch(self):
        resolver, summary = self.resolver([[document("mp-7", simple_si())]])
        request = StructureRequest(
            operation="select",
            selection="require_unique",
            material_ids=("mp-7",),
        )
        result = resolver.resolve(request)
        self.assertEqual(result.status, ResolutionStatus.SELECTED)
        self.assertEqual(summary.calls[0]["material_ids"], ["mp-7"])

        bad_resolver, _ = self.resolver([[document("mp-7", simple_si())]])
        mismatch = bad_resolver.resolve(request.model_copy(update={"formula": "Ge"}))
        self.assertEqual(mismatch.status, ResolutionStatus.CONSTRAINTS_NOT_SATISFIED)

    def test_semantic_label_is_unsupported_without_api_or_files(self):
        resolver, summary = self.resolver([[document("mp-1", simple_si())]])
        result = resolver.resolve(self.search_request(semantic_label="2H"))
        self.assertEqual(result.status, ResolutionStatus.UNSUPPORTED_SEMANTIC)
        self.assertEqual(summary.calls, [])
        self.assertEqual(list(Path(self.temp.name).iterdir()), [])

    def test_energy_range_is_revalidated(self):
        resolver, _ = self.resolver([[document("mp-1", simple_si(), 0.2)]])
        request = self.search_request(energy_above_hull=FloatRange(maximum=0.1))
        self.assertEqual(resolver.resolve(request).status, ResolutionStatus.NO_MATCHES)

    def test_element_only_and_non_finite_ranges_are_rejected(self):
        with self.assertRaises(ValidationError):
            StructureRequest(
                operation="search",
                selection="return_all",
                include_elements=("O",),
            )
        with self.assertRaises(ValidationError):
            FloatRange(maximum=float("nan"))
        with self.assertRaises(ValidationError):
            FloatRange(maximum=float("inf"))

    def test_malformed_document_is_rejected_without_losing_valid_records(self):
        malformed = document("mp-1", simple_si())
        malformed.nsites = "not-an-integer"
        resolver, _ = self.resolver([[malformed, document("mp-2", simple_si())]])
        result = resolver.resolve(self.search_request())
        self.assertEqual(result.status, ResolutionStatus.SEARCH_RESULTS)
        self.assertEqual([candidate.material_id for candidate in result.candidates], ["mp-2"])

    def test_selected_filename_is_sanitized_and_stays_in_output_directory(self):
        resolver, _ = self.resolver([[document("../unsafe/id", simple_si())]])
        result = resolver.resolve(self.select_request())
        self.assertEqual(result.status, ResolutionStatus.SELECTED)
        output_path = Path(result.structure_path).resolve()
        self.assertEqual(output_path.parent, Path(self.temp.name).resolve())
        self.assertEqual(output_path.name, "unsafe_id_Si.vasp")
        self.assertEqual(len(list(Path(self.temp.name).iterdir())), 1)

    def test_selected_structure_is_written_atomically_inside_output_directory(self):
        resolver, _ = self.resolver([[document("mp-2", simple_si())]])
        final_path = Path(self.temp.name).resolve() / "mp-2_Si.vasp"
        writer_paths = []
        original_to = Structure.to

        def recording_to(structure, *args, **kwargs):
            writer_paths.append(Path(kwargs["filename"]).resolve())
            return original_to(structure, *args, **kwargs)

        with patch.object(Structure, "to", new=recording_to):
            result = resolver.resolve(self.select_request())

        self.assertEqual(result.status, ResolutionStatus.SELECTED)
        self.assertEqual(Path(result.structure_path), final_path)
        self.assertEqual(len(writer_paths), 1)
        self.assertEqual(writer_paths[0].parent, final_path.parent)
        self.assertNotEqual(writer_paths[0], final_path)
        self.assertFalse(writer_paths[0].exists())
        self.assertTrue(final_path.is_file())
        self.assertEqual(len(list(final_path.parent.iterdir())), 1)


if __name__ == "__main__":
    unittest.main()
