"""Application boundary for pure deterministic MP structure retrieval."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Protocol

from pydantic import BaseModel, ConfigDict, model_validator

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
    rendered_response: Optional[str] = None

    @model_validator(mode="after")
    def validate_boundary(self) -> "StructureApplicationBoundaryResult":
        pure = self.applicability.status == ApplicabilityStatus.PURE_MP_STRUCTURE
        if pure != (self.coordinator_result is not None):
            raise ValueError("only pure MP requests may contain a coordinator result")
        if self.should_run_crewai != (
            self.applicability.status == ApplicabilityStatus.NOT_PURE
        ):
            raise ValueError("CrewAI may run only for not-pure requests")
        if self.should_run_crewai and self.rendered_response is not None:
            raise ValueError("pass-through results must not contain rendered output")
        if not self.should_run_crewai and not self.rendered_response:
            raise ValueError("intercepted results require deterministic output")
        return self


class StructureApplicationBoundary:
    def __init__(
        self,
        classifier: ApplicabilityClassifier,
        coordinator: StructureCoordinator,
        invocation_store: Optional[InvocationStore] = None,
    ):
        self.classifier = classifier
        self.coordinator = coordinator
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
            if applicability.status == ApplicabilityStatus.NOT_PURE:
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
