"""Application boundary for deterministic MP structure retrieval workflows."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Protocol

from pydantic import BaseModel, ConfigDict, field_validator, model_validator
from .structure_models import ResolutionStatus

from .structure_request_applicability import (
    ApplicabilityStatus,
    StructureApplicabilityResult,
)
from .structure_request_coordinator import (
    InMemoryInvocationStore,
    InvocationStore,
    StructureCoordinatorResult,
)


class ApplicabilityClassifier(Protocol):
    def classify(self, source_text: str) -> StructureApplicabilityResult: ...


class StructureCoordinator(Protocol):
    def handle_structure_request(
        self, source_text: str, output_directory: str | Path, invocation_key: str
    ) -> StructureCoordinatorResult: ...


class StructureApplicationBoundaryResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    applicability: StructureApplicabilityResult
    should_run_crewai: bool
    coordinator_result: Optional[StructureCoordinatorResult] = None
    resolved_structure: Optional["ResolvedStructureContext"] = None
    rendered_response: Optional[str] = None

    @model_validator(mode="after")
    def validate_boundary(self) -> "StructureApplicationBoundaryResult":
        mp_request = self.applicability.status in {
            ApplicabilityStatus.PURE_MP_STRUCTURE,
            ApplicabilityStatus.MIXED_MP_STRUCTURE,
        }
        if mp_request != (self.coordinator_result is not None):
            raise ValueError("MP requests must contain a coordinator result")
        expected_run = self.applicability.status in {
            ApplicabilityStatus.NOT_PURE,
            ApplicabilityStatus.LOCAL_OR_UNRELATED,
        } or self.resolved_structure is not None
        if self.should_run_crewai != expected_run:
            raise ValueError("CrewAI continuation is inconsistent with boundary status")
        if self.resolved_structure is not None and (
            self.applicability.status != ApplicabilityStatus.MIXED_MP_STRUCTURE
        ):
            raise ValueError("resolved context is valid only for mixed MP workflows")
        if self.should_run_crewai and self.rendered_response is not None:
            raise ValueError("pass-through results must not contain rendered output")
        if not self.should_run_crewai and not self.rendered_response:
            raise ValueError("intercepted results require deterministic output")
        return self


class ResolvedStructureContext(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    material_id: str
    structure_path: str
    formula: str
    resolver_status: ResolutionStatus
    resolver_policy_version: str
    semantic_label: Optional[str] = None
    semantic_policy_version: Optional[str] = None

    @field_validator("structure_path")
    @classmethod
    def canonical_existing_path(cls, value: str) -> str:
        path = Path(value).resolve(strict=True)
        if not path.is_file():
            raise ValueError("selected structure path must be a file")
        return str(path)

    @model_validator(mode="after")
    def require_selected(self) -> "ResolvedStructureContext":
        if self.resolver_status != ResolutionStatus.SELECTED:
            raise ValueError("resolved context requires selected resolver status")
        return self

    @classmethod
    def from_result(
        cls,
        result: StructureCoordinatorResult,
        output_directory: str | Path,
    ) -> "ResolvedStructureContext":
        resolution = result.resolution_result
        if (
            resolution is None
            or resolution.status != ResolutionStatus.SELECTED
            or resolution.selected is None
            or resolution.structure_path is None
        ):
            raise ValueError("coordinator result is not a selected structure")
        root = Path(output_directory).resolve(strict=True)
        selected_path = Path(resolution.structure_path).resolve(strict=True)
        if not selected_path.is_relative_to(root):
            raise ValueError("selected structure path is outside conversation directory")
        return cls(
            material_id=resolution.selected.material_id,
            structure_path=str(selected_path),
            formula=resolution.selected.formula,
            resolver_status=resolution.status,
            resolver_policy_version=resolution.resolver_policy_version,
            semantic_label=resolution.normalized_semantic_label,
            semantic_policy_version=resolution.semantic_policy_version,
        )


class StructureApplicationBoundary:
    def __init__(
        self,
        classifier: ApplicabilityClassifier,
        coordinator: StructureCoordinator,
        invocation_store: Optional[InvocationStore] = None,
        mixed_coordinator: Optional[StructureCoordinator] = None,
    ):
        self.classifier = classifier
        self.coordinator = coordinator
        self.mixed_coordinator = mixed_coordinator or coordinator
        self.invocation_store = invocation_store or InMemoryInvocationStore()

    def handle(
        self,
        source_text: str,
        output_directory: str | Path,
        invocation_key: str,
    ) -> StructureApplicationBoundaryResult:
        normalized_output = os.path.normcase(
            str(Path(output_directory).resolve(strict=False))
        )

        def execute() -> StructureApplicationBoundaryResult:
            applicability = self.classifier.classify(source_text)
            if applicability.status == ApplicabilityStatus.PURE_MP_STRUCTURE:
                coordinator_result = self.coordinator.handle_structure_request(
                    source_text, output_directory, invocation_key
                )
                return StructureApplicationBoundaryResult(
                    applicability=applicability,
                    should_run_crewai=False,
                    coordinator_result=coordinator_result,
                    rendered_response=coordinator_result.rendered_response,
                )
            if applicability.status == ApplicabilityStatus.MIXED_MP_STRUCTURE:
                coordinator_result = self.mixed_coordinator.handle_structure_request(
                    source_text, output_directory, invocation_key
                )
                if coordinator_result.resolver_status == ResolutionStatus.SELECTED:
                    context = ResolvedStructureContext.from_result(
                        coordinator_result, output_directory
                    )
                    return StructureApplicationBoundaryResult(
                        applicability=applicability,
                        should_run_crewai=True,
                        coordinator_result=coordinator_result,
                        resolved_structure=context,
                    )
                return StructureApplicationBoundaryResult(
                    applicability=applicability,
                    should_run_crewai=False,
                    coordinator_result=coordinator_result,
                    rendered_response=coordinator_result.rendered_response,
                )
            if applicability.status in {
                ApplicabilityStatus.NOT_PURE,
                ApplicabilityStatus.LOCAL_OR_UNRELATED,
            }:
                return StructureApplicationBoundaryResult(
                    applicability=applicability,
                    should_run_crewai=True,
                )
            if applicability.status == ApplicabilityStatus.CLARIFICATION_REQUIRED:
                rendered = f"clarification_required: {applicability.clarification}"
            else:
                rendered = f"classification_error: {applicability.error}"
            return StructureApplicationBoundaryResult(
                applicability=applicability,
                should_run_crewai=False,
                rendered_response=rendered,
            )

        result, _ = self.invocation_store.execute_once(
            f"application:{invocation_key}",
            execute,
            binding=(source_text, normalized_output),
        )
        return result
