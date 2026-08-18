"""Strict LLM classification for pure Materials Project structure requests."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Callable, Optional

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator


APPLICABILITY_POLICY_VERSION = "structure-request-applicability-v1"
MAX_CLASSIFICATION_RETRIES = 2


class ApplicabilityStatus(str, Enum):
    PURE_MP_STRUCTURE = "pure_mp_structure"
    NOT_PURE = "not_pure"
    CLARIFICATION_REQUIRED = "clarification_required"
    CLASSIFICATION_ERROR = "classification_error"


class ApplicabilityEvidence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)

    retrieval_intent: str
    material_anchor: str


class StructureApplicabilityResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: ApplicabilityStatus
    evidence: Optional[ApplicabilityEvidence] = None
    clarification: Optional[str] = None
    error: Optional[str] = None
    retry_count: int = Field(ge=0, le=MAX_CLASSIFICATION_RETRIES)
    policy_version: str = APPLICABILITY_POLICY_VERSION

    @model_validator(mode="after")
    def validate_status_payload(self) -> "StructureApplicabilityResult":
        if self.status == ApplicabilityStatus.PURE_MP_STRUCTURE and self.evidence is None:
            raise ValueError("pure MP classification requires evidence")
        if self.status == ApplicabilityStatus.CLARIFICATION_REQUIRED and not self.clarification:
            raise ValueError("clarification text is required")
        if self.status == ApplicabilityStatus.CLASSIFICATION_ERROR and not self.error:
            raise ValueError("classification error text is required")
        if self.status != ApplicabilityStatus.PURE_MP_STRUCTURE and self.evidence is not None:
            raise ValueError("only pure MP classification may include evidence")
        if self.status != ApplicabilityStatus.CLARIFICATION_REQUIRED and self.clarification is not None:
            raise ValueError("clarification is valid only for clarification_required")
        if self.status != ApplicabilityStatus.CLASSIFICATION_ERROR and self.error is not None:
            raise ValueError("error is valid only for classification_error")
        return self


class _ClassifierEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: ApplicabilityStatus
    evidence: Optional[ApplicabilityEvidence] = None
    clarification: Optional[str] = None


SYSTEM_PROMPT = """Classify whether an arbitrary VASPilot task is ONLY a request to
search for, list, retrieve, select, show, or download crystal structure records
from Materials Project. Return exactly one JSON object and no Markdown.

Allowed statuses: pure_mp_structure, not_pure, clarification_required.
PURE means structure retrieval is the complete final action. It includes explicit
MP IDs and conventional prototype/phase searches. NOT_PURE includes calculations,
relaxation, electronic band structure, density of states, existing output parsing,
local/uploaded files, structure creation, literature search, explanation, and any
larger workflow that merely needs a structure.

Do not classify from the word "structure" alone. Do not extract or infer any
crystallographic constraint. For pure_mp_structure provide exact source substrings:
{"status":"pure_mp_structure","evidence":{"retrieval_intent":"exact quote",
"material_anchor":"exact quote"},"clarification":null}
The retrieval quote must demonstrate search/retrieve/list/select/show/find/download
intent. The anchor must identify a formula, material name, MP ID, phase, prototype,
or explicit Materials Project source. For other statuses evidence must be null.
Use clarification_required only when the final requested action is genuinely unclear.
"""


def applicability_classifier_from_config(
    config: dict[str, Any],
    llm_config_key: str = "fn_call_llm",
    max_corrective_retries: int = MAX_CLASSIFICATION_RETRIES,
) -> "StructureRequestApplicabilityClassifier":
    from crewai import LLM

    llm_name = config["llm_config"][llm_config_key]
    llm = LLM(**config["llm_mapper"][llm_name])
    return StructureRequestApplicabilityClassifier(
        llm_callable=llm.call,
        max_corrective_retries=max_corrective_retries,
    )


class StructureRequestApplicabilityClassifier:
    def __init__(
        self,
        llm_callable: Callable[[list[dict[str, str]]], str],
        max_corrective_retries: int = MAX_CLASSIFICATION_RETRIES,
    ) -> None:
        if not 0 <= max_corrective_retries <= MAX_CLASSIFICATION_RETRIES:
            raise ValueError(
                f"max_corrective_retries must be 0..{MAX_CLASSIFICATION_RETRIES}"
            )
        self.llm_callable = llm_callable
        self.max_corrective_retries = max_corrective_retries

    def classify(self, source_text: str) -> StructureApplicabilityResult:
        if not source_text.strip():
            return StructureApplicabilityResult(
                status=ApplicabilityStatus.CLASSIFICATION_ERROR,
                error="source text must not be blank",
                retry_count=0,
            )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": source_text},
        ]
        last_error = "unknown classification error"
        for attempt in range(self.max_corrective_retries + 1):
            raw = ""
            try:
                raw = self.llm_callable(messages)
                envelope = _ClassifierEnvelope.model_validate(json.loads(raw))
                if envelope.status == ApplicabilityStatus.CLASSIFICATION_ERROR:
                    raise ValueError("LLM cannot return classification_error")
                if envelope.status == ApplicabilityStatus.PURE_MP_STRUCTURE:
                    if envelope.evidence is None:
                        raise ValueError("pure MP classification requires evidence")
                    self._validate_evidence(source_text, envelope.evidence)
                elif envelope.evidence is not None:
                    raise ValueError("only pure MP classification may include evidence")
                return StructureApplicabilityResult(
                    status=envelope.status,
                    evidence=envelope.evidence,
                    clarification=envelope.clarification,
                    retry_count=attempt,
                )
            except (ValueError, TypeError, json.JSONDecodeError, ValidationError) as exc:
                last_error = str(exc)
            except Exception as exc:
                last_error = str(exc)
            if attempt < self.max_corrective_retries:
                messages.extend(
                    [
                        {"role": "assistant", "content": str(raw)},
                        {
                            "role": "user",
                            "content": "Correct the JSON only. Validation error: " + last_error,
                        },
                    ]
                )
        return StructureApplicabilityResult(
            status=ApplicabilityStatus.CLASSIFICATION_ERROR,
            error=last_error,
            retry_count=self.max_corrective_retries,
        )

    @staticmethod
    def _validate_evidence(source_text: str, evidence: ApplicabilityEvidence) -> None:
        if not evidence.retrieval_intent or evidence.retrieval_intent not in source_text:
            raise ValueError("retrieval intent evidence must be exact source text")
        if not evidence.material_anchor or evidence.material_anchor not in source_text:
            raise ValueError("material anchor evidence must be exact source text")
        if evidence.retrieval_intent == evidence.material_anchor:
            raise ValueError("retrieval intent and material anchor evidence must be distinct")
