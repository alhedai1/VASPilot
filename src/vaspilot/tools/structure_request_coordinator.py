"""Isolated parse-resolve-render coordination for structure requests."""

from __future__ import annotations

from collections import OrderedDict
import os
from pathlib import Path
from threading import Event, Lock
from typing import Callable, Optional, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .structure_models import (
    CandidateResult,
    ResolutionStatus,
    StructureRequest,
    StructureResolutionResult,
)
from .structure_request_parser import (
    ParserStatus,
    StructureRequestParseResult,
)


COORDINATOR_POLICY_VERSION = "structure-request-coordinator-v1"
DEFAULT_INVOCATION_STORE_CAPACITY = 256


class RequestParser(Protocol):
    def parse(self, source_text: str) -> StructureRequestParseResult: ...


class RequestResolver(Protocol):
    def resolve(self, request: StructureRequest) -> StructureResolutionResult: ...


class StructureCoordinatorResult(BaseModel):
    """Completed coordinator outcome with parser/resolver contracts intact."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    invocation_key: str
    parser_result: StructureRequestParseResult
    resolution_result: Optional[StructureResolutionResult] = None
    request: Optional[StructureRequest] = None
    parser_status: ParserStatus
    resolver_status: Optional[ResolutionStatus] = None
    selected: Optional[CandidateResult] = None
    structure_path: Optional[str] = None
    candidates: tuple[CandidateResult, ...] = ()
    parser_retry_count: int = Field(ge=0)
    resolver_policy_version: Optional[str] = None
    semantic_policy_version: Optional[str] = None
    rendered_response: str
    returned_from_store: bool = False
    coordinator_policy_version: str = COORDINATOR_POLICY_VERSION

    @field_validator("invocation_key")
    @classmethod
    def reject_blank_key(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("invocation_key must not be blank")
        return value

    @model_validator(mode="after")
    def preserve_nested_result_invariants(self) -> "StructureCoordinatorResult":
        parse = self.parser_result
        resolution = self.resolution_result
        if self.parser_status != parse.status or self.parser_retry_count != parse.retry_count:
            raise ValueError("coordinator parser fields must match parser_result")
        if self.request != parse.request:
            raise ValueError("coordinator request must be the unaltered parsed request")
        if parse.status == ParserStatus.PARSED:
            if resolution is None:
                raise ValueError("parsed requests require a resolution_result")
            if resolution.request != parse.request:
                raise ValueError("resolver request must equal the parsed request")
            expected = (
                resolution.status,
                resolution.selected,
                resolution.structure_path,
                resolution.candidates,
                resolution.resolver_policy_version,
                resolution.semantic_policy_version,
            )
            actual = (
                self.resolver_status,
                self.selected,
                self.structure_path,
                self.candidates,
                self.resolver_policy_version,
                self.semantic_policy_version,
            )
            if actual != expected:
                raise ValueError("coordinator resolver fields must match resolution_result")
        elif resolution is not None or self.resolver_status is not None:
            raise ValueError("non-parsed outcomes must not contain resolver data")
        return self


T = TypeVar("T")


class InvocationStore(Protocol):
    def execute_once(
        self,
        invocation_key: str,
        operation: Callable[[], T],
        binding: object = None,
    ) -> tuple[T, bool]:
        """Return (value, was_cached), executing operation once per completed key."""


class InMemoryInvocationStore:
    """Bounded, process-local, thread-safe completed-invocation store."""

    def __init__(self, capacity: int = DEFAULT_INVOCATION_STORE_CAPACITY) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._lock = Lock()
        self._completed: OrderedDict[str, object] = OrderedDict()
        self._in_flight: dict[str, Event] = {}
        self._bindings: dict[str, object] = {}

    def execute_once(
        self,
        invocation_key: str,
        operation: Callable[[], T],
        binding: object = None,
    ) -> tuple[T, bool]:
        while True:
            with self._lock:
                if invocation_key in self._bindings and self._bindings[invocation_key] != binding:
                    raise ValueError(
                        "invocation_key is already bound to different source text "
                        "or output directory"
                    )
                if invocation_key in self._completed:
                    value = self._completed.pop(invocation_key)
                    self._completed[invocation_key] = value
                    return value, True  # type: ignore[return-value]
                event = self._in_flight.get(invocation_key)
                if event is None:
                    event = Event()
                    self._in_flight[invocation_key] = event
                    self._bindings[invocation_key] = binding
                    owner = True
                else:
                    owner = False
            if owner:
                break
            event.wait()

        try:
            value = operation()
        except BaseException:
            with self._lock:
                self._in_flight.pop(invocation_key).set()
                self._bindings.pop(invocation_key, None)
            raise

        with self._lock:
            self._completed[invocation_key] = value
            self._completed.move_to_end(invocation_key)
            while len(self._completed) > self.capacity:
                evicted_key, _ = self._completed.popitem(last=False)
                self._bindings.pop(evicted_key, None)
            self._in_flight.pop(invocation_key).set()
        return value, False


def render_structure_result(
    parser_result: StructureRequestParseResult,
    resolution_result: Optional[StructureResolutionResult],
) -> str:
    """Render a completed outcome deterministically without an LLM."""

    if parser_result.status == ParserStatus.CLARIFICATION_REQUIRED:
        return f"clarification_required: {parser_result.clarification}"
    if parser_result.status == ParserStatus.PARSE_ERROR:
        return f"parse_error: {parser_result.error}"
    if resolution_result is None:
        raise ValueError("parsed result requires resolution_result")

    status = resolution_result.status
    if status == ResolutionStatus.SEARCH_RESULTS:
        lines = [f"search_results: {len(resolution_result.candidates)} candidate(s)"]
        lines.extend(_render_candidate(candidate) for candidate in resolution_result.candidates)
        return "\n".join(lines)
    if status == ResolutionStatus.SELECTED:
        selected = resolution_result.selected
        if selected is None or resolution_result.structure_path is None:
            raise ValueError("selected result is missing candidate or structure path")
        return f"selected: {selected.material_id}\nstructure_path: {resolution_result.structure_path}"
    if status == ResolutionStatus.AMBIGUOUS:
        groups = [
            f"group {group.group_index}: {','.join(group.member_material_ids)}"
            for group in resolution_result.equivalence_groups
        ]
        return "\n".join(["ambiguous: multiple distinct structures"] + groups)
    if status == ResolutionStatus.UNSUPPORTED_SEMANTIC:
        label = resolution_result.requested_semantic_label or "<unspecified>"
        return f"unsupported_semantic: {label} is not supported by the deterministic semantic policy"
    if status == ResolutionStatus.NO_MATCHES:
        return "no_matches: no records satisfy all explicit constraints"
    if status == ResolutionStatus.CONSTRAINTS_NOT_SATISFIED:
        return "constraints_not_satisfied: the explicit material ID does not satisfy all constraints"
    if status == ResolutionStatus.INSUFFICIENT_DATA:
        return "insufficient_data: required ranking data is unavailable"
    if status in (ResolutionStatus.API_ERROR, ResolutionStatus.DOWNLOAD_ERROR):
        return f"{status.value}: {resolution_result.error}"
    raise ValueError(f"unhandled resolver status: {status}")


def _render_candidate(candidate: CandidateResult) -> str:
    energy = (
        "missing"
        if candidate.energy_above_hull is None
        else format(candidate.energy_above_hull, ".12g")
    )
    return (
        f"- {candidate.material_id} | {candidate.formula} | "
        f"{candidate.symmetry.symbol} ({candidate.symmetry.number}) | "
        f"nsites={candidate.nsites} | energy_above_hull={energy}"
    )


class StructureRequestCoordinator:
    """Enforce one parser and, when parsed, one resolver call per key."""

    def __init__(
        self,
        parser: RequestParser,
        resolver_factory: Callable[[Path], RequestResolver],
        invocation_store: Optional[InvocationStore] = None,
        renderer: Callable[[StructureRequestParseResult, Optional[StructureResolutionResult]], str] = render_structure_result,
    ) -> None:
        self.parser = parser
        self.resolver_factory = resolver_factory
        self.invocation_store = invocation_store or InMemoryInvocationStore()
        self.renderer = renderer

    def handle_structure_request(
        self,
        source_text: str,
        output_directory: str | Path,
        invocation_key: str,
    ) -> StructureCoordinatorResult:
        if not invocation_key.strip():
            raise ValueError("invocation_key must not be blank")
        output_path = Path(output_directory)
        normalized_output = os.path.normcase(str(output_path.resolve(strict=False)))

        def execute() -> StructureCoordinatorResult:
            parse = self.parser.parse(source_text)
            resolution: Optional[StructureResolutionResult] = None
            if parse.status == ParserStatus.PARSED:
                if parse.request is None:
                    raise ValueError("parsed status is missing StructureRequest")
                resolution = self.resolver_factory(output_path).resolve(parse.request)
            rendered = self.renderer(parse, resolution)
            return _coordinator_result(invocation_key, parse, resolution, rendered)

        result, was_cached = self.invocation_store.execute_once(
            invocation_key,
            execute,
            binding=(source_text, normalized_output),
        )
        if was_cached:
            return result.model_copy(update={"returned_from_store": True})
        return result


def _coordinator_result(
    invocation_key: str,
    parse: StructureRequestParseResult,
    resolution: Optional[StructureResolutionResult],
    rendered: str,
) -> StructureCoordinatorResult:
    return StructureCoordinatorResult(
        invocation_key=invocation_key,
        parser_result=parse,
        resolution_result=resolution,
        request=parse.request,
        parser_status=parse.status,
        resolver_status=resolution.status if resolution else None,
        selected=resolution.selected if resolution else None,
        structure_path=resolution.structure_path if resolution else None,
        candidates=resolution.candidates if resolution else (),
        parser_retry_count=parse.retry_count,
        resolver_policy_version=resolution.resolver_policy_version if resolution else None,
        semantic_policy_version=resolution.semantic_policy_version if resolution else None,
        rendered_response=rendered,
    )
