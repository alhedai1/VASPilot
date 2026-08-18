"""Deterministic local recognition of named crystal prototypes and families."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from pymatgen.analysis.prototypes import AflowPrototypeMatcher, get_protostructure_label
from pymatgen.core import Composition, Element, Structure


SEMANTIC_POLICY_VERSION = "structure-semantics-v1"
AFLOW_MATCHER_LTOL = 0.20
AFLOW_MATCHER_STOL = 0.30
AFLOW_MATCHER_ANGLE_TOLERANCE = 5.0
AFLOW_LABEL_SYMPREC_ANGSTROM = 0.1
MOLYBDENITE_AFLOW_TAG = "AB2_hP6_194_c_f"
TMD_CHALCOGENS = frozenset({"S", "Se", "Te"})


class SemanticMatchMethod(str, Enum):
    AFLOW_PROTOTYPE = "aflow_prototype"
    AFLOW_PROTOSTRUCTURE_LABEL = "aflow_protostructure_label"
    AFLOW_MX2_2H_FAMILY = "aflow_mx2_2h_family"


class SemanticCandidateStatus(str, Enum):
    MATCH = "match"
    NO_MATCH = "no_match"
    INCOMPATIBLE_FAMILY = "incompatible_family"


@dataclass(frozen=True)
class PrototypeIdentity:
    canonical_label: str
    aflow_tag: str
    mineral_name: str
    strukturbericht: str


@dataclass(frozen=True)
class SemanticPlan:
    requested_label: str
    normalized_label: str
    method: SemanticMatchMethod
    prototype: PrototypeIdentity


@dataclass(frozen=True)
class SemanticCandidateMatch:
    status: SemanticCandidateStatus
    plan: SemanticPlan
    match_method: Optional[SemanticMatchMethod] = None
    explanation: Optional[str] = None


PROTOTYPES = {
    "rocksalt": PrototypeIdentity(
        "rocksalt", "AB_cF8_225_a_b", "Halite, Rock Salt", "B1"
    ),
    "wurtzite": PrototypeIdentity(
        "wurtzite", "AB_hP4_186_b_b", "Wurtzite", "B4"
    ),
    "zinc blende": PrototypeIdentity(
        "zinc blende", "AB_cF8_216_c_a", "Zincblende, Sphalerite", "B3"
    ),
    "rutile": PrototypeIdentity("rutile", "A2B_tP6_136_f_a", "Rutile", "C4"),
    "anatase": PrototypeIdentity("anatase", "A2B_tI12_141_e_a", "Anatase", "C5"),
    "cubic perovskite": PrototypeIdentity(
        "cubic perovskite", "AB3C_cP5_221_a_c_b", "(Cubic) Perovskite", "E2_1"
    ),
    "orthorhombic perovskite": PrototypeIdentity(
        "orthorhombic perovskite",
        "AB3C_oP20_62_c_cd_a",
        "Orthorhombic Perovskite",
        "None",
    ),
    "molybdenite": PrototypeIdentity(
        "molybdenite", MOLYBDENITE_AFLOW_TAG, "Molybdenite", "C7"
    ),
}

ALIASES = {
    "rocksalt": "rocksalt",
    "rock salt": "rocksalt",
    "halite": "rocksalt",
    "wurtzite": "wurtzite",
    "zinc blende": "zinc blende",
    "zincblende": "zinc blende",
    "sphalerite": "zinc blende",
    "rutile": "rutile",
    "anatase": "anatase",
    "cubic perovskite": "cubic perovskite",
    "orthorhombic perovskite": "orthorhombic perovskite",
    "molybdenite": "molybdenite",
}


def normalize_semantic_label(label: str) -> str:
    normalized = re.sub(r"[-_]+", " ", label.strip().casefold())
    return " ".join(normalized.split())


def is_mx2_tmd_composition(composition: Composition) -> bool:
    reduced = composition.reduced_composition
    amounts = reduced.get_el_amt_dict()
    if len(amounts) != 2 or sorted(amounts.values()) != [1.0, 2.0]:
        return False
    metal_symbols = [symbol for symbol, amount in amounts.items() if amount == 1]
    chalcogen_symbols = [symbol for symbol, amount in amounts.items() if amount == 2]
    if len(metal_symbols) != 1 or len(chalcogen_symbols) != 1:
        return False
    metal = Element(metal_symbols[0])
    return metal.is_transition_metal and chalcogen_symbols[0] in TMD_CHALCOGENS


class StructureSemanticRecognizer:
    """Resolve aliases and validate candidate structures against AFLOW."""

    def __init__(self) -> None:
        self.matcher = AflowPrototypeMatcher(
            initial_ltol=AFLOW_MATCHER_LTOL,
            initial_stol=AFLOW_MATCHER_STOL,
            initial_angle_tol=AFLOW_MATCHER_ANGLE_TOLERANCE,
        )

    def plan(self, label: str) -> Optional[SemanticPlan]:
        normalized = normalize_semantic_label(label)
        if normalized == "2h":
            return SemanticPlan(
                requested_label=label,
                normalized_label=normalized,
                method=SemanticMatchMethod.AFLOW_MX2_2H_FAMILY,
                prototype=PROTOTYPES["molybdenite"],
            )
        canonical = ALIASES.get(normalized)
        if canonical is None:
            return None
        return SemanticPlan(
            requested_label=label,
            normalized_label=canonical,
            method=SemanticMatchMethod.AFLOW_PROTOTYPE,
            prototype=PROTOTYPES[canonical],
        )

    def request_family_is_compatible(
        self, plan: SemanticPlan, formula: Optional[str]
    ) -> bool:
        if plan.method != SemanticMatchMethod.AFLOW_MX2_2H_FAMILY or formula is None:
            return True
        try:
            return is_mx2_tmd_composition(Composition(formula))
        except (TypeError, ValueError):
            return False

    def match(self, plan: SemanticPlan, structure: Structure) -> SemanticCandidateMatch:
        if plan.method == SemanticMatchMethod.AFLOW_MX2_2H_FAMILY:
            if not is_mx2_tmd_composition(structure.composition):
                return SemanticCandidateMatch(
                    SemanticCandidateStatus.INCOMPATIBLE_FAMILY,
                    plan,
                    explanation="2H v1 supports layered transition-metal dichalcogenides with MX2 composition",
                )
        matches = self.matcher.get_prototypes(structure) or []
        matched_tags = {item["tags"].get("aflow") for item in matches}
        if plan.prototype.aflow_tag in matched_tags:
            return SemanticCandidateMatch(
                SemanticCandidateStatus.MATCH,
                plan,
                match_method=plan.method,
            )

        # The bundled matcher compares coordinates and can reject a distorted
        # member of the same generic prototype. For named AFLOW prototypes only,
        # accept the exact structure-derived AFLOW identity (formula archetype,
        # Pearson symbol, space group, and occupied Wyckoff positions). This is
        # deliberately not used by the stricter MX2/2H family classifier.
        if plan.method == SemanticMatchMethod.AFLOW_PROTOTYPE:
            try:
                label = get_protostructure_label(
                    structure,
                    method="spglib",
                    raise_errors=True,
                    init_symprec=AFLOW_LABEL_SYMPREC_ANGSTROM,
                    fallback_symprec=None,
                )
            except (TypeError, ValueError):
                label = None
            if label and label.partition(":")[0] == plan.prototype.aflow_tag:
                return SemanticCandidateMatch(
                    SemanticCandidateStatus.MATCH,
                    plan,
                    match_method=SemanticMatchMethod.AFLOW_PROTOSTRUCTURE_LABEL,
                )

        return SemanticCandidateMatch(SemanticCandidateStatus.NO_MATCH, plan)
