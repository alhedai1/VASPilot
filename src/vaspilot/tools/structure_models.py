"""Strict contracts for deterministic Materials Project structure resolution."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RequestOperation(str, Enum):
    SEARCH = "search"
    SELECT = "select"


class SelectionBehavior(str, Enum):
    RETURN_ALL = "return_all"
    REQUIRE_UNIQUE = "require_unique"
    MOST_STABLE = "most_stable"


class TheoreticalFilter(str, Enum):
    ANY = "any"
    ONLY_THEORETICAL = "only_theoretical"
    ONLY_EXPERIMENTAL = "only_experimental"


class CrystalSystem(str, Enum):
    TRICLINIC = "triclinic"
    MONOCLINIC = "monoclinic"
    ORTHORHOMBIC = "orthorhombic"
    TETRAGONAL = "tetragonal"
    TRIGONAL = "trigonal"
    HEXAGONAL = "hexagonal"
    CUBIC = "cubic"


class ResultOrder(str, Enum):
    ENERGY_THEN_ID = "energy_then_id"
    MATERIAL_ID = "material_id"
    FORMULA_THEN_ENERGY = "formula_then_energy"
    NSITES_THEN_ENERGY = "nsites_then_energy"


class ResolutionStatus(str, Enum):
    SEARCH_RESULTS = "search_results"
    SELECTED = "selected"
    AMBIGUOUS = "ambiguous"
    UNSUPPORTED_SEMANTIC = "unsupported_semantic"
    NO_MATCHES = "no_matches"
    CONSTRAINTS_NOT_SATISFIED = "constraints_not_satisfied"
    INSUFFICIENT_DATA = "insufficient_data"
    API_ERROR = "api_error"
    DOWNLOAD_ERROR = "download_error"


class IntRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum: Optional[int] = Field(default=None, ge=0)
    maximum: Optional[int] = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "IntRange":
        if self.minimum is None and self.maximum is None:
            raise ValueError("at least one range bound is required")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum must not exceed maximum")
        return self


class FloatRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    minimum: Optional[float] = None
    maximum: Optional[float] = None

    @model_validator(mode="after")
    def validate_bounds(self) -> "FloatRange":
        if self.minimum is None and self.maximum is None:
            raise ValueError("at least one range bound is required")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("minimum must not exceed maximum")
        return self


class StructureRequest(BaseModel):
    """An already-parsed structure query with no inferred constraints."""

    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    operation: RequestOperation
    selection: SelectionBehavior

    material_ids: tuple[str, ...] = ()
    formula: Optional[str] = None
    chemical_system: tuple[str, ...] = ()
    include_elements: tuple[str, ...] = ()
    exclude_elements: tuple[str, ...] = ()

    crystal_system: Optional[CrystalSystem] = None
    spacegroup_symbol: Optional[str] = None
    spacegroup_number: Optional[int] = Field(default=None, ge=1, le=230)
    num_sites: Optional[IntRange] = None
    energy_above_hull: Optional[FloatRange] = None

    stable_only: bool = False
    theoretical: TheoreticalFilter = TheoreticalFilter.ANY

    semantic_label: Optional[str] = None
    result_limit: int = Field(default=20, ge=1, le=1000)
    order_by: ResultOrder = ResultOrder.ENERGY_THEN_ID

    @field_validator("material_ids", "chemical_system", "include_elements", "exclude_elements")
    @classmethod
    def reject_blank_tuple_items(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        if any(not value.strip() for value in values):
            raise ValueError("constraint values must not be blank")
        return values

    @field_validator("formula", "spacegroup_symbol", "semantic_label")
    @classmethod
    def reject_blank_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is not None and not value.strip():
            raise ValueError("constraint value must not be blank")
        return value

    @model_validator(mode="after")
    def validate_operation(self) -> "StructureRequest":
        if self.operation == RequestOperation.SEARCH:
            if self.selection != SelectionBehavior.RETURN_ALL:
                raise ValueError("search operation requires return_all selection")
        elif self.selection == SelectionBehavior.RETURN_ALL:
            raise ValueError("select operation cannot use return_all selection")

        # Elements are supported as additional filters, but an element-only query
        # can require downloading a substantial fraction of Materials Project.
        if not any((self.material_ids, self.formula, self.chemical_system)):
            raise ValueError(
                "material_ids, formula, or chemical_system is required to bound the query"
            )
        return self


class SymmetrySummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    symbol: str
    number: int
    crystal_system: str
    point_group: str


class SemanticMatchSummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    requested_label: str
    normalized_label: str
    match_method: str
    matched_aflow_tag: str
    matched_name: str
    semantic_policy_version: str


class CandidateResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    material_id: str
    formula: str
    chemical_system: str
    nsites: int
    energy_above_hull: Optional[float]
    formation_energy_per_atom: Optional[float]
    is_stable: bool
    theoretical: Optional[bool]
    deprecated: bool
    symmetry: SymmetrySummary
    equivalence_group: Optional[int] = None
    semantic_match: Optional[SemanticMatchSummary] = None


class EquivalenceGroupResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    group_index: int
    representative_material_id: str
    member_material_ids: tuple[str, ...]
    energy_above_hull: Optional[float]


class StructureResolutionResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ResolutionStatus
    request: StructureRequest
    candidates: tuple[CandidateResult, ...] = ()
    selected: Optional[CandidateResult] = None
    structure_path: Optional[str] = None
    equivalence_groups: tuple[EquivalenceGroupResult, ...] = ()
    total_api_records: int = 0
    validated_records: int = 0
    resolver_policy_version: str
    requested_semantic_label: Optional[str] = None
    normalized_semantic_label: Optional[str] = None
    semantic_policy_version: Optional[str] = None
    error: Optional[str] = None
