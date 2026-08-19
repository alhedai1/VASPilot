"""Deterministic, agent-independent Materials Project structure resolver."""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Optional

from mp_api.client import MPRester
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Composition, Element, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup

from .structure_models import (
    CandidateResult,
    EquivalenceGroupResult,
    RequestOperation,
    ResolutionStatus,
    ResultOrder,
    SelectionBehavior,
    StructureRequest,
    StructureResolutionResult,
    SemanticMatchSummary,
    SymmetrySummary,
    TheoreticalFilter,
)
from .structure_semantics import (
    SEMANTIC_POLICY_VERSION,
    SemanticCandidateStatus,
    SemanticPlan,
    StructureSemanticRecognizer,
)


RESOLVER_POLICY_VERSION = "structure-resolver-v1"
INCLUDE_GNOME = False
SUMMARY_CHUNK_SIZE = 100
SYMPREC_ANGSTROM = 0.1
ANGLE_TOLERANCE_DEGREES = 5.0
MATCHER_LTOL = 0.20
MATCHER_STOL = 0.30
MATCHER_ANGLE_TOLERANCE = 5.0
ENERGY_TIE_TOLERANCE_EV_PER_ATOM = 1e-6

SUMMARY_FIELDS = [
    "material_id",
    "formula_pretty",
    "formula_anonymous",
    "chemsys",
    "elements",
    "composition",
    "composition_reduced",
    "structure",
    "symmetry",
    "nsites",
    "volume",
    "density",
    "energy_above_hull",
    "formation_energy_per_atom",
    "is_stable",
    "deprecated",
    "theoretical",
    "database_IDs",
    "origins",
    "warnings",
]

@dataclass(frozen=True)
class _ValidatedCandidate:
    result: CandidateResult
    structure: Structure


def _finite_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _canonical_id_key(material_id: str) -> tuple[int, int, str]:
    text = str(material_id).strip().lower()
    if text.startswith("mp-") and text[3:].isdigit():
        return (0, int(text[3:]), text)
    return (1, 0, text)


def _normalize_formula(formula: str) -> str:
    return Composition(formula).reduced_formula


def _normalize_elements(elements: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted({Element(str(value)).symbol for value in elements}))


def _normalize_chemsys(elements: Iterable[str]) -> str:
    return "-".join(_normalize_elements(elements))


def _in_range(value: float | int, bounds: Any) -> bool:
    if bounds.minimum is not None and value < bounds.minimum:
        return False
    return not (bounds.maximum is not None and value > bounds.maximum)


def _safe_filename_component(value: str) -> str:
    component = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._")
    return component or "unknown"


class StructureResolver:
    """Resolve a strict StructureRequest without any LLM or agent behavior."""

    def __init__(
        self,
        api_key: str,
        output_dir: str | Path,
        mpr_factory: Callable[[str], Any] = MPRester,
        semantic_recognizer: Optional[StructureSemanticRecognizer] = None,
    ) -> None:
        self.api_key = api_key
        self.output_dir = Path(output_dir)
        self.mpr_factory = mpr_factory
        self.semantic_recognizer = semantic_recognizer
        self.matcher = StructureMatcher(
            ltol=MATCHER_LTOL,
            stol=MATCHER_STOL,
            angle_tol=MATCHER_ANGLE_TOLERANCE,
            primitive_cell=True,
            scale=True,
            attempt_supercell=False,
            allow_subset=False,
        )

    def resolve(self, request: StructureRequest) -> StructureResolutionResult:
        semantic_plan: Optional[SemanticPlan] = None
        if request.semantic_label:
            recognizer = self._get_semantic_recognizer()
            semantic_plan = recognizer.plan(request.semantic_label)
            if semantic_plan is None:
                return self._semantic_result(
                    request,
                    ResolutionStatus.UNSUPPORTED_SEMANTIC,
                    None,
                    error=f"unsupported semantic label: {request.semantic_label}",
                )
            if not recognizer.request_family_is_compatible(
                semantic_plan, request.formula
            ):
                return self._semantic_result(
                    request,
                    ResolutionStatus.UNSUPPORTED_SEMANTIC,
                    semantic_plan,
                    error="2H v1 supports only transition-metal dichalcogenide MX2 compositions",
                )

        try:
            normalized = self._normalize_request(request)
        except (TypeError, ValueError) as exc:
            return self._result(request, ResolutionStatus.CONSTRAINTS_NOT_SATISFIED, error=str(exc))

        try:
            documents = self._query(normalized)
        except Exception as exc:
            return self._result(request, ResolutionStatus.API_ERROR, error=str(exc))

        validated = []
        for document in documents:
            try:
                candidate = self._validate_document(document, normalized)
            except (ArithmeticError, TypeError, ValueError):
                candidate = None
            if candidate is not None:
                validated.append(candidate)
        validated.sort(key=lambda item: self._sort_key(item.result, request.order_by))

        if semantic_plan is not None:
            semantic_validated = []
            compatible_candidates = 0
            for candidate in validated:
                semantic_match = recognizer.match(
                    semantic_plan, candidate.structure
                )
                if semantic_match.status != SemanticCandidateStatus.INCOMPATIBLE_FAMILY:
                    compatible_candidates += 1
                if semantic_match.status == SemanticCandidateStatus.MATCH:
                    diagnostic = SemanticMatchSummary(
                        requested_label=semantic_plan.requested_label,
                        normalized_label=semantic_plan.normalized_label,
                        match_method=semantic_match.match_method.value,
                        matched_aflow_tag=semantic_plan.prototype.aflow_tag,
                        matched_name=semantic_plan.prototype.mineral_name,
                        semantic_policy_version=SEMANTIC_POLICY_VERSION,
                    )
                    semantic_validated.append(
                        _ValidatedCandidate(
                            result=candidate.result.model_copy(
                                update={"semantic_match": diagnostic}
                            ),
                            structure=candidate.structure,
                        )
                    )
            validated = semantic_validated
            if not validated and compatible_candidates == 0:
                return self._semantic_result(
                    request,
                    ResolutionStatus.UNSUPPORTED_SEMANTIC,
                    semantic_plan,
                    total_api_records=len(documents),
                    error="no candidate belongs to the semantic label's supported material family",
                )

        if not validated:
            status = (
                ResolutionStatus.CONSTRAINTS_NOT_SATISFIED
                if documents and len(request.material_ids) == 1
                else ResolutionStatus.NO_MATCHES
            )
            return self._semantic_result(
                request,
                status,
                semantic_plan,
                total_api_records=len(documents),
            )

        if request.operation == RequestOperation.SEARCH:
            results = tuple(item.result for item in validated[: request.result_limit])
            return self._semantic_result(
                request,
                ResolutionStatus.SEARCH_RESULTS,
                semantic_plan,
                candidates=results,
                total_api_records=len(documents),
                validated_records=len(validated),
            )

        groups = self._group_equivalent(validated)
        group_results = self._group_results(groups)

        if len(request.material_ids) == 1:
            winner = validated[0]
        elif request.selection == SelectionBehavior.REQUIRE_UNIQUE:
            if len(groups) != 1:
                return self._selection_result(
                    request, ResolutionStatus.AMBIGUOUS, validated, documents, group_results
                )
            winner = self._representative(groups[0])
        else:
            finite_groups = [
                (group, self._group_energy(group))
                for group in groups
                if self._group_energy(group) is not None
            ]
            if not finite_groups:
                return self._selection_result(
                    request,
                    ResolutionStatus.INSUFFICIENT_DATA,
                    validated,
                    documents,
                    group_results,
                )
            minimum = min(energy for _, energy in finite_groups if energy is not None)
            winners = [
                group
                for group, energy in finite_groups
                if energy is not None
                and abs(energy - minimum) <= ENERGY_TIE_TOLERANCE_EV_PER_ATOM
            ]
            if len(winners) != 1:
                return self._selection_result(
                    request, ResolutionStatus.AMBIGUOUS, validated, documents, group_results
                )
            winner = self._representative(winners[0])

        try:
            path = self._write_selected(winner)
        except Exception as exc:
            return self._selection_result(
                request,
                ResolutionStatus.DOWNLOAD_ERROR,
                validated,
                documents,
                group_results,
                error=str(exc),
            )
        return self._semantic_result(
            request,
            ResolutionStatus.SELECTED,
            semantic_plan,
            candidates=tuple(item.result for item in validated),
            selected=winner.result,
            structure_path=str(path),
            equivalence_groups=group_results,
            total_api_records=len(documents),
            validated_records=len(validated),
        )

    def _normalize_request(self, request: StructureRequest) -> dict[str, Any]:
        crystal_system = request.crystal_system.value if request.crystal_system else None
        spacegroup_symbol = (
            SpaceGroup(request.spacegroup_symbol).symbol
            if request.spacegroup_symbol
            else None
        )
        return {
            "request": request,
            "material_ids": tuple(str(value).strip() for value in request.material_ids),
            "formula": _normalize_formula(request.formula) if request.formula else None,
            "chemsys": _normalize_chemsys(request.chemical_system) if request.chemical_system else None,
            "include_elements": set(_normalize_elements(request.include_elements)),
            "exclude_elements": set(_normalize_elements(request.exclude_elements)),
            "crystal_system": crystal_system,
            "spacegroup_symbol": spacegroup_symbol,
        }

    def _query(self, normalized: dict[str, Any]) -> list[Any]:
        request: StructureRequest = normalized["request"]
        params: dict[str, Any] = {
            "deprecated": False,
            "include_gnome": INCLUDE_GNOME,
            "num_chunks": None,
            "chunk_size": SUMMARY_CHUNK_SIZE,
            "all_fields": False,
            "fields": SUMMARY_FIELDS,
        }
        if normalized["material_ids"]:
            params["material_ids"] = list(normalized["material_ids"])
        if normalized["formula"]:
            params["formula"] = normalized["formula"]
        if normalized["chemsys"]:
            params["chemsys"] = normalized["chemsys"]
        if normalized["include_elements"]:
            params["elements"] = sorted(normalized["include_elements"])
        if normalized["exclude_elements"]:
            params["exclude_elements"] = sorted(normalized["exclude_elements"])
        if normalized["crystal_system"]:
            params["crystal_system"] = normalized["crystal_system"].capitalize()
        if normalized["spacegroup_symbol"]:
            params["spacegroup_symbol"] = normalized["spacegroup_symbol"]
        if request.spacegroup_number is not None:
            params["spacegroup_number"] = request.spacegroup_number
        if request.num_sites:
            params["num_sites"] = (request.num_sites.minimum, request.num_sites.maximum)
        if request.energy_above_hull:
            params["energy_above_hull"] = (
                request.energy_above_hull.minimum,
                request.energy_above_hull.maximum,
            )
        if request.stable_only:
            params["is_stable"] = True
        if request.theoretical == TheoreticalFilter.ONLY_THEORETICAL:
            params["theoretical"] = True
        elif request.theoretical == TheoreticalFilter.ONLY_EXPERIMENTAL:
            params["theoretical"] = False

        with self.mpr_factory(self.api_key) as mpr:
            return list(mpr.materials.summary.search(**params))

    def _validate_document(
        self, document: Any, normalized: dict[str, Any]
    ) -> Optional[_ValidatedCandidate]:
        request: StructureRequest = normalized["request"]
        if bool(getattr(document, "deprecated", True)):
            return None
        structure = getattr(document, "structure", None)
        if not isinstance(structure, Structure):
            return None

        material_id = str(getattr(document, "material_id", "")).strip()
        if not material_id:
            return None
        if normalized["material_ids"] and material_id not in normalized["material_ids"]:
            return None

        formula = structure.composition.reduced_formula
        if normalized["formula"] and formula != normalized["formula"]:
            return None
        elements = {element.symbol for element in structure.composition.elements}
        chemsys = "-".join(sorted(elements))
        if normalized["chemsys"] and chemsys != normalized["chemsys"]:
            return None
        if not normalized["include_elements"].issubset(elements):
            return None
        if normalized["exclude_elements"].intersection(elements):
            return None

        nsites = len(structure)
        documented_nsites = getattr(document, "nsites", None)
        if documented_nsites is not None and int(documented_nsites) != nsites:
            return None
        if request.num_sites and not _in_range(nsites, request.num_sites):
            return None

        energy = _finite_float(getattr(document, "energy_above_hull", None))
        if request.energy_above_hull:
            if energy is None or not _in_range(energy, request.energy_above_hull):
                return None

        stable = bool(getattr(document, "is_stable", False))
        if request.stable_only and not stable:
            return None
        theoretical = getattr(document, "theoretical", None)
        if request.theoretical == TheoreticalFilter.ONLY_THEORETICAL and theoretical is not True:
            return None
        if request.theoretical == TheoreticalFilter.ONLY_EXPERIMENTAL and theoretical is not False:
            return None

        analyzer = SpacegroupAnalyzer(
            structure,
            symprec=SYMPREC_ANGSTROM,
            angle_tolerance=ANGLE_TOLERANCE_DEGREES,
        )
        symbol = analyzer.get_space_group_symbol()
        number = analyzer.get_space_group_number()
        crystal_system = analyzer.get_crystal_system().lower()
        if normalized["spacegroup_symbol"] and symbol != normalized["spacegroup_symbol"]:
            return None
        if request.spacegroup_number is not None and number != request.spacegroup_number:
            return None
        if normalized["crystal_system"] and crystal_system != normalized["crystal_system"]:
            return None

        result = CandidateResult(
            material_id=material_id,
            formula=formula,
            chemical_system=chemsys,
            nsites=nsites,
            energy_above_hull=energy,
            formation_energy_per_atom=_finite_float(
                getattr(document, "formation_energy_per_atom", None)
            ),
            is_stable=stable,
            theoretical=theoretical if isinstance(theoretical, bool) else None,
            deprecated=False,
            symmetry=SymmetrySummary(
                symbol=symbol,
                number=number,
                crystal_system=crystal_system,
                point_group=analyzer.get_point_group_symbol(),
            ),
        )
        return _ValidatedCandidate(result=result, structure=structure)

    def _sort_key(self, candidate: CandidateResult, order: ResultOrder) -> tuple[Any, ...]:
        energy_key = (
            candidate.energy_above_hull is None,
            candidate.energy_above_hull if candidate.energy_above_hull is not None else math.inf,
        )
        id_key = _canonical_id_key(candidate.material_id)
        if order == ResultOrder.MATERIAL_ID:
            return id_key
        if order == ResultOrder.FORMULA_THEN_ENERGY:
            return (candidate.formula, *energy_key, *id_key)
        if order == ResultOrder.NSITES_THEN_ENERGY:
            return (candidate.nsites, *energy_key, *id_key)
        return (*energy_key, *id_key)

    def _group_equivalent(
        self, candidates: list[_ValidatedCandidate]
    ) -> list[list[_ValidatedCandidate]]:
        groups: list[list[_ValidatedCandidate]] = []
        for candidate in sorted(candidates, key=lambda item: _canonical_id_key(item.result.material_id)):
            for group in groups:
                if all(self.matcher.fit(candidate.structure, member.structure) for member in group):
                    group.append(candidate)
                    break
            else:
                groups.append([candidate])
        return groups

    def _representative(self, group: list[_ValidatedCandidate]) -> _ValidatedCandidate:
        return min(group, key=lambda item: _canonical_id_key(item.result.material_id))

    def _group_energy(self, group: list[_ValidatedCandidate]) -> Optional[float]:
        energies = [
            item.result.energy_above_hull
            for item in group
            if item.result.energy_above_hull is not None
        ]
        return min(energies) if energies else None

    def _group_results(
        self, groups: list[list[_ValidatedCandidate]]
    ) -> tuple[EquivalenceGroupResult, ...]:
        output = []
        for index, group in enumerate(groups):
            representative = self._representative(group)
            output.append(
                EquivalenceGroupResult(
                    group_index=index,
                    representative_material_id=representative.result.material_id,
                    member_material_ids=tuple(
                        item.result.material_id
                        for item in sorted(
                            group, key=lambda value: _canonical_id_key(value.result.material_id)
                        )
                    ),
                    energy_above_hull=self._group_energy(group),
                )
            )
        return tuple(output)

    def _write_selected(self, candidate: _ValidatedCandidate) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_dir = self.output_dir.resolve(strict=True)
        material_id = _safe_filename_component(candidate.result.material_id)
        formula = _safe_filename_component(candidate.result.formula)
        path = output_dir / f"{material_id}_{formula}.vasp"
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".vaspilot-structure-", suffix=".tmp", dir=output_dir
        )
        os.close(descriptor)
        temporary_path = Path(temporary_name)
        try:
            candidate.structure.to(filename=str(temporary_path), fmt="poscar")
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return path

    def _selection_result(
        self,
        request: StructureRequest,
        status: ResolutionStatus,
        validated: list[_ValidatedCandidate],
        documents: list[Any],
        groups: tuple[EquivalenceGroupResult, ...],
        error: Optional[str] = None,
    ) -> StructureResolutionResult:
        semantic_plan = (
            self._get_semantic_recognizer().plan(request.semantic_label)
            if request.semantic_label
            else None
        )
        return self._semantic_result(
            request,
            status,
            semantic_plan,
            candidates=tuple(item.result for item in validated),
            equivalence_groups=groups,
            total_api_records=len(documents),
            validated_records=len(validated),
            error=error,
        )

    def _get_semantic_recognizer(self) -> StructureSemanticRecognizer:
        if self.semantic_recognizer is None:
            self.semantic_recognizer = StructureSemanticRecognizer()
        return self.semantic_recognizer

    @staticmethod
    def _result(
        request: StructureRequest,
        status: ResolutionStatus,
        **kwargs: Any,
    ) -> StructureResolutionResult:
        return StructureResolutionResult(
            status=status,
            request=request,
            resolver_policy_version=RESOLVER_POLICY_VERSION,
            **kwargs,
        )

    @staticmethod
    def _semantic_result(
        request: StructureRequest,
        status: ResolutionStatus,
        plan: Optional[SemanticPlan],
        **kwargs: Any,
    ) -> StructureResolutionResult:
        return StructureResolutionResult(
            status=status,
            request=request,
            resolver_policy_version=RESOLVER_POLICY_VERSION,
            requested_semantic_label=request.semantic_label,
            normalized_semantic_label=plan.normalized_label if plan else None,
            semantic_policy_version=(
                SEMANTIC_POLICY_VERSION if request.semantic_label else None
            ),
            **kwargs,
        )
