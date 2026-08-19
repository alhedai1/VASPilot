"""Isolated natural-language parser for strict StructureRequest objects."""

from __future__ import annotations

import json
import re
from enum import Enum
from typing import Any, Callable, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator
from pymatgen.core import Composition, Element
from pymatgen.symmetry.groups import SpaceGroup

from .structure_models import (
    RequestOperation,
    ResultOrder,
    SelectionBehavior,
    StructureRequest,
    TheoreticalFilter,
)


PARSER_POLICY_VERSION = "structure-request-parser-v1"
MAX_CORRECTIVE_RETRIES = 2
_MULTIPLE_INPUT_INTENT = re.compile(
    r"\b(?:compare|comparison|several|multiple|more\s+than\s+one|all\s+(?:the\s+)?structures)\b",
    re.IGNORECASE,
)


class ParserMode(str, Enum):
    STANDARD = "standard"
    MIXED_SINGLE_INPUT = "mixed_single_input"


class ParserStatus(str, Enum):
    PARSED = "parsed"
    CLARIFICATION_REQUIRED = "clarification_required"
    PARSE_ERROR = "parse_error"


class RequestEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    field: str
    quote: str

    @field_validator("field", "quote")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        if not value:
            raise ValueError("evidence values must not be blank")
        return value


class StructureRequestParseResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ParserStatus
    request: Optional[StructureRequest] = None
    evidence: tuple[RequestEvidence, ...] = ()
    clarification: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = Field(ge=0, le=MAX_CORRECTIVE_RETRIES)
    parser_policy_version: str = PARSER_POLICY_VERSION


class _LLMEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ParserStatus
    request: Optional[dict[str, Any]] = None
    evidence: tuple[RequestEvidence, ...] = ()
    clarification: Optional[str] = None


SYSTEM_PROMPT = """You extract an already-identified crystal-structure retrieval request.
Return exactly one JSON object and no Markdown.

Allowed status values: parsed, clarification_required.
For parsed, return: {"status":"parsed","request":{...},"evidence":[...]}
For clarification, return: {"status":"clarification_required","request":null,
"evidence":[],"clarification":"a concise question"}

The request schema supports only:
operation (search|select), selection (return_all|require_unique|most_stable),
material_ids, formula, chemical_system, include_elements, exclude_elements,
crystal_system, spacegroup_symbol, spacegroup_number, num_sites,
energy_above_hull, stable_only, theoretical, semantic_label, result_limit,
order_by.

Action rules:
- explicit search/list/show candidates or plural-result intent => search + return_all
- explicit most stable/lowest energy above hull => select + most_stable
- explicit get/retrieve/download/use one structure => select + require_unique
- one explicit MP ID => select + require_unique
- conflicting or unclear action intent => clarification_required

Extract only facts explicitly stated by the user. Never infer crystallographic
properties from a phase/polytype/prototype label. In particular, 2H does not
imply a space group, crystal system, site count, or energy constraint.
Copy a named phase/polytype/prototype into semantic_label.
Use theoretical values: any, only_theoretical, only_experimental.
Use order_by values: energy_then_id, material_id, formula_then_energy,
nsites_then_energy. Omit result_limit and order_by unless explicitly requested.
Omit all other unspecified fields; application defaults will fill them.

Evidence is an array of {"field":"field.path","quote":"exact source text"}.
Provide evidence for operation, selection, and every explicitly extracted
constraint. Quotes must be exact substrings. Range endpoints use paths such as
num_sites.maximum or energy_above_hull.minimum. Do not provide evidence for
omitted/default values.
"""

MIXED_SINGLE_INPUT_PROMPT = """You extract only the Materials Project structure
constraints from a larger downstream workflow that requires exactly one initial
structure. Return exactly one JSON object and no Markdown.

For parsed output use exactly this shape:
{"status":"parsed","request":{"formula":"MoS2"},
"evidence":[{"field":"formula","quote":"MoS2"}],"clarification":null}
Every evidence item has exactly the keys "field" and "quote". The value of
"quote" must be a verbatim substring of the user request. Never use a key named
"evidence" inside an evidence item and never omit evidence for an extracted field.
For clarification use exactly:
{"status":"clarification_required","request":null,"evidence":[],
"clarification":"a concise question"}
Example: for "relax 2H-MoS2 using VASP", extract request
{"formula":"MoS2","semantic_label":"2H"} with exact evidence for both fields.

Allowed statuses: parsed, clarification_required. For parsed, request may contain
only: material_ids, formula, chemical_system, include_elements, exclude_elements,
crystal_system, spacegroup_symbol, spacegroup_number, num_sites,
energy_above_hull, stable_only, theoretical, semantic_label, selection.

The application owns operation=select and defaults selection=require_unique.
Include selection=most_stable only when the user explicitly says most stable or
lowest energy above hull. Do not include operation. Do not provide evidence for
application-owned defaults. Provide exact source evidence for every extracted
scientific constraint and for explicit most_stable selection.

Never infer crystallographic properties from phase/polytype/prototype labels.
Never translate common chemical names into formulas. Never invent space group,
crystal system, site count, energy constraints, MP IDs, or filtering policy.
If the workflow requests several structures or a comparison, return
clarification_required because this mode accepts one authoritative input only.
"""


def parser_from_config(
    config: dict[str, Any],
    llm_config_key: str = "fn_call_llm",
    max_corrective_retries: int = MAX_CORRECTIVE_RETRIES,
    mode: ParserMode = ParserMode.STANDARD,
) -> "StructureRequestParser":
    """Build a direct parser call using VASPilot's existing LLM mapping."""

    from crewai import LLM

    llm_name = config["llm_config"][llm_config_key]
    llm_params = config["llm_mapper"][llm_name]
    llm = LLM(**llm_params)
    return StructureRequestParser(
        llm_callable=llm.call,
        max_corrective_retries=max_corrective_retries,
        mode=mode,
    )


class StructureRequestParser:
    """Parse one request with strict JSON and deterministic evidence checks."""

    def __init__(
        self,
        llm_callable: Callable[[list[dict[str, str]]], str],
        max_corrective_retries: int = MAX_CORRECTIVE_RETRIES,
        mode: ParserMode = ParserMode.STANDARD,
    ) -> None:
        if not 0 <= max_corrective_retries <= MAX_CORRECTIVE_RETRIES:
            raise ValueError(f"max_corrective_retries must be 0..{MAX_CORRECTIVE_RETRIES}")
        self.llm_callable = llm_callable
        self.max_corrective_retries = max_corrective_retries
        self.mode = mode

    def parse(self, source_text: str) -> StructureRequestParseResult:
        if not source_text.strip():
            return StructureRequestParseResult(
                status=ParserStatus.PARSE_ERROR,
                error="source text must not be blank",
                retry_count=0,
            )
        if (
            self.mode == ParserMode.MIXED_SINGLE_INPUT
            and _MULTIPLE_INPUT_INTENT.search(source_text)
        ):
            return StructureRequestParseResult(
                status=ParserStatus.CLARIFICATION_REQUIRED,
                clarification=(
                    "This workflow currently accepts one authoritative Materials "
                    "Project structure. Please specify which single structure to use."
                ),
                retry_count=0,
            )

        messages = [
            {
                "role": "system",
                "content": (
                    MIXED_SINGLE_INPUT_PROMPT
                    if self.mode == ParserMode.MIXED_SINGLE_INPUT
                    else SYSTEM_PROMPT
                ),
            },
            {"role": "user", "content": source_text},
        ]
        last_error = "unknown parser error"
        for attempt in range(self.max_corrective_retries + 1):
            try:
                raw = self.llm_callable(messages)
                envelope = _LLMEnvelope.model_validate(json.loads(raw))
                if envelope.status == ParserStatus.CLARIFICATION_REQUIRED:
                    if envelope.request is not None or envelope.evidence:
                        raise ValueError("clarification must not include a request or evidence")
                    if not envelope.clarification:
                        raise ValueError("clarification text is required")
                    return StructureRequestParseResult(
                        status=ParserStatus.CLARIFICATION_REQUIRED,
                        clarification=envelope.clarification,
                        retry_count=attempt,
                    )
                if envelope.status != ParserStatus.PARSED:
                    raise ValueError("LLM cannot return parse_error")
                if envelope.request is None:
                    raise ValueError("parsed status requires a request")

                request_data = dict(envelope.request)
                workflow_defaults: set[str] = set()
                if self.mode == ParserMode.MIXED_SINGLE_INPUT:
                    if "operation" in request_data:
                        raise ValueError("mixed parser operation is application-owned")
                    request_data["operation"] = RequestOperation.SELECT
                    workflow_defaults.add("operation")
                    if "selection" not in request_data:
                        request_data["selection"] = SelectionBehavior.REQUIRE_UNIQUE
                        workflow_defaults.add("selection")
                    elif request_data["selection"] != SelectionBehavior.MOST_STABLE.value:
                        raise ValueError(
                            "mixed parser may specify only explicit most_stable selection"
                        )
                if request_data.get("formula"):
                    request_data["formula"] = Composition(
                        request_data["formula"]
                    ).reduced_formula
                if request_data.get("semantic_label"):
                    request_data["semantic_label"] = self._normalize_semantic_label(
                        request_data["semantic_label"], request_data.get("formula")
                    )
                request = StructureRequest.model_validate(request_data)
                self._validate_evidence(
                    source_text, request, envelope.evidence, workflow_defaults
                )
                return StructureRequestParseResult(
                    status=ParserStatus.PARSED,
                    request=request,
                    evidence=envelope.evidence,
                    retry_count=attempt,
                )
            except Exception as exc:
                last_error = str(exc)
                if attempt >= self.max_corrective_retries:
                    break
                messages.extend(
                    [
                        {"role": "assistant", "content": str(raw) if "raw" in locals() else ""},
                        {
                            "role": "user",
                            "content": (
                                "Correct the JSON only. Preserve the original user intent; "
                                "do not add constraints. Validation error: " + last_error
                            ),
                        },
                    ]
                )

        return StructureRequestParseResult(
            status=ParserStatus.PARSE_ERROR,
            error=last_error,
            retry_count=self.max_corrective_retries,
        )

    def _validate_evidence(
        self,
        source_text: str,
        request: StructureRequest,
        evidence: Sequence[RequestEvidence],
        workflow_defaults: Optional[set[str]] = None,
    ) -> None:
        evidence_by_field: dict[str, list[str]] = {}
        for item in evidence:
            if item.quote not in source_text:
                raise ValueError(f"evidence quote is not exact source text: {item.quote!r}")
            evidence_by_field.setdefault(item.field, []).append(item.quote)

        workflow_defaults = workflow_defaults or set()
        required = self._required_evidence_fields(request) - workflow_defaults
        supplied = set(evidence_by_field)
        missing = required - supplied
        unexpected = supplied - required
        if missing:
            raise ValueError(f"missing evidence for: {sorted(missing)}")
        if unexpected:
            raise ValueError(f"evidence supplied for default or absent fields: {sorted(unexpected)}")

        self._validate_action_evidence(request, evidence_by_field, workflow_defaults)
        self._validate_literal_evidence(source_text, request, evidence_by_field)

    @staticmethod
    def _required_evidence_fields(request: StructureRequest) -> set[str]:
        fields = {"operation", "selection"}
        for name in (
            "material_ids",
            "formula",
            "chemical_system",
            "include_elements",
            "exclude_elements",
            "crystal_system",
            "spacegroup_symbol",
            "spacegroup_number",
            "num_sites",
            "energy_above_hull",
            "semantic_label",
        ):
            value = getattr(request, name)
            if value not in (None, (), False):
                if name in {"num_sites", "energy_above_hull"}:
                    if value.minimum is not None:
                        fields.add(f"{name}.minimum")
                    if value.maximum is not None:
                        fields.add(f"{name}.maximum")
                else:
                    fields.add(name)
        if request.stable_only:
            fields.add("stable_only")
        if request.theoretical != TheoreticalFilter.ANY:
            fields.add("theoretical")
        if request.result_limit != StructureRequest.model_fields["result_limit"].default:
            fields.add("result_limit")
        if request.order_by != StructureRequest.model_fields["order_by"].default:
            fields.add("order_by")
        return fields

    @staticmethod
    def _validate_action_evidence(
        request: StructureRequest,
        evidence: dict[str, list[str]],
        workflow_defaults: Optional[set[str]] = None,
    ) -> None:
        workflow_defaults = workflow_defaults or set()
        if {"operation", "selection"}.issubset(workflow_defaults):
            return
        operation_text = " ".join(evidence.get("operation", ())).casefold()
        selection_text = " ".join(evidence.get("selection", ())).casefold()
        search_words = ("search", "find", "list", "show candidates", "structures")
        select_words = ("get", "retrieve", "download", "use", "most stable", "lowest energy")
        if "operation" not in workflow_defaults and request.operation == RequestOperation.SEARCH:
            if not any(word in operation_text for word in search_words):
                raise ValueError("search operation lacks explicit action evidence")
        elif "operation" not in workflow_defaults and not any(word in operation_text for word in select_words) and not request.material_ids:
            raise ValueError("select operation lacks explicit action evidence")

        if "selection" in workflow_defaults:
            return
        if request.selection == SelectionBehavior.RETURN_ALL:
            if not any(word in selection_text for word in search_words):
                raise ValueError("return_all lacks explicit plural/search evidence")
        elif request.selection == SelectionBehavior.MOST_STABLE:
            if not any(word in selection_text for word in ("most stable", "lowest energy above hull")):
                raise ValueError("most_stable lacks explicit evidence")
        elif not any(word in selection_text for word in select_words) and not request.material_ids:
            raise ValueError("require_unique lacks explicit single-selection evidence")

    def _validate_literal_evidence(
        self,
        source_text: str,
        request: StructureRequest,
        evidence: dict[str, list[str]],
    ) -> None:
        def combined(field: str) -> str:
            return " ".join(evidence.get(field, ()))

        for material_id in request.material_ids:
            if material_id.casefold() not in combined("material_ids").casefold():
                raise ValueError(f"material ID is not literal evidence: {material_id}")

        if request.formula and not StructureRequestParser._formula_supported(
            request.formula, combined("formula")
        ):
            raise ValueError("formula is not supported by its evidence")

        for field in ("chemical_system", "include_elements", "exclude_elements"):
            for symbol in getattr(request, field):
                quote = combined(field)
                element = Element(symbol)
                symbol_pattern = rf"(?<![A-Za-z]){re.escape(symbol)}(?![a-z])"
                name_pattern = rf"\b{re.escape(element.long_name)}\b"
                if not re.search(symbol_pattern, quote) and not re.search(
                    name_pattern, quote, flags=re.IGNORECASE
                ):
                    raise ValueError(f"element {symbol} is not supported by {field} evidence")

        if request.crystal_system:
            if request.crystal_system.value.casefold() not in combined("crystal_system").casefold():
                raise ValueError("crystal system is not literal evidence")
        if request.spacegroup_number is not None:
            if not re.search(
                rf"(?<!\d){request.spacegroup_number}(?!\d)", combined("spacegroup_number")
            ):
                raise ValueError("space-group number is not literal evidence")
        if request.spacegroup_symbol:
            quote = combined("spacegroup_symbol")
            try:
                supported = SpaceGroup(quote.strip()).symbol == SpaceGroup(
                    request.spacegroup_symbol
                ).symbol
            except ValueError:
                supported = request.spacegroup_symbol.casefold() in quote.casefold()
            if not supported:
                raise ValueError("space-group symbol is not literal evidence")

        for range_name in ("num_sites", "energy_above_hull"):
            value = getattr(request, range_name)
            if value is None:
                continue
            for bound_name in ("minimum", "maximum"):
                bound = getattr(value, bound_name)
                if bound is None:
                    continue
                quote = combined(f"{range_name}.{bound_name}")
                if not StructureRequestParser._number_is_literal(bound, quote):
                    raise ValueError(f"{range_name}.{bound_name} is not literal evidence")

        if request.result_limit != StructureRequest.model_fields["result_limit"].default:
            if not StructureRequestParser._number_is_literal(
                request.result_limit, combined("result_limit")
            ):
                raise ValueError("result limit is not literal evidence")
        if request.stable_only and not re.search(
            r"\bstable\b", combined("stable_only"), flags=re.IGNORECASE
        ):
            raise ValueError("stable-only constraint is not literal evidence")
        if request.theoretical == TheoreticalFilter.ONLY_EXPERIMENTAL:
            if not re.search(
                r"\bexperimental\b", combined("theoretical"), flags=re.IGNORECASE
            ):
                raise ValueError("experimental constraint is not literal evidence")
        elif request.theoretical == TheoreticalFilter.ONLY_THEORETICAL:
            if not re.search(
                r"\btheoretical\b", combined("theoretical"), flags=re.IGNORECASE
            ):
                raise ValueError("theoretical constraint is not literal evidence")
        if request.order_by != StructureRequest.model_fields["order_by"].default:
            order_terms = {
                ResultOrder.MATERIAL_ID: ("material id", "mp id"),
                ResultOrder.FORMULA_THEN_ENERGY: ("formula",),
                ResultOrder.NSITES_THEN_ENERGY: ("number of sites", "site count", "sites"),
            }
            quote = combined("order_by").casefold()
            if not any(term in quote for term in order_terms.get(request.order_by, ())):
                raise ValueError("result ordering is not literal evidence")
        if request.semantic_label:
            if request.semantic_label.casefold() not in combined("semantic_label").casefold():
                raise ValueError("semantic label is not copied from the source")
        elif (
            self.mode == ParserMode.MIXED_SINGLE_INPUT
            and request.formula
            and self._hyphenated_semantic_label(source_text, request.formula)
        ):
            raise ValueError("explicit hyphenated semantic label was omitted")

    @staticmethod
    def _formula_supported(normalized_formula: str, quote: str) -> bool:
        for token in re.findall(r"[A-Za-z][A-Za-z0-9.()]*", quote):
            try:
                if Composition(token).reduced_formula == normalized_formula:
                    return True
            except (TypeError, ValueError):
                continue
        return False

    @staticmethod
    def _number_is_literal(value: float | int, quote: str) -> bool:
        for token in re.findall(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?", quote):
            try:
                if math_isclose(float(token), float(value)):
                    return True
            except ValueError:
                continue
        return False

    @staticmethod
    def _normalize_semantic_label(value: str, formula: Optional[str] = None) -> str:
        """Remove generic wrappers without interpreting the label itself."""

        normalized = value.strip()
        if formula:
            normalized = re.sub(
                rf"(?:\s+phase\s+of\s+|\s+of\s+|[-\s]+){re.escape(formula)}$",
                "",
                normalized,
                flags=re.IGNORECASE,
            ).strip()
        normalized = re.sub(
            r"^(?:phase|polytype|prototype)\s+",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
        normalized = re.sub(
            r"\s+(?:phase|polytype|prototype)$",
            "",
            normalized,
            flags=re.IGNORECASE,
        )
        if not normalized:
            raise ValueError("semantic label contains only a generic wrapper")
        return normalized

    @staticmethod
    def _hyphenated_semantic_label(source_text: str, formula: str) -> Optional[str]:
        match = re.search(
            rf"(?<![A-Za-z0-9])([A-Za-z0-9][A-Za-z0-9_-]*)-{re.escape(formula)}(?![A-Za-z0-9])",
            source_text,
            flags=re.IGNORECASE,
        )
        return match.group(1) if match else None


def math_isclose(left: float, right: float) -> bool:
    return abs(left - right) <= max(1e-12, 1e-12 * max(abs(left), abs(right)))
