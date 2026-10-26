"""Structured incident analysis request and response models.

IncidentAnalysis is the schema enforced by OpenAI Structured Outputs (strict mode).
All fields are required; severity is a closed enum so the model cannot invent values.
"""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


Severity = Literal["P1", "P2", "P3", "P4"]


class IncidentAnalysis(BaseModel):
    """Enterprise incident analysis returned by the model under schema enforcement."""

    incident_id: str = Field(min_length=1, max_length=120)
    severity: Severity
    summary: str = Field(min_length=1, max_length=4000)
    probable_cause: str = Field(min_length=1, max_length=4000)
    recommended_actions: list[str] = Field(min_length=1, max_length=20)
    requires_human_approval: bool


class IncidentAnalyzeRequest(BaseModel):
    """Fictional incident details supplied by the operator."""

    incident_id: str = Field(min_length=1, max_length=120)
    description: str = Field(min_length=1, max_length=20000)

    @field_validator("incident_id", "description")
    @classmethod
    def strip_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Value cannot be blank")
        return cleaned


class IncidentAnalyzeResponse(BaseModel):
    analysis: IncidentAnalysis
    request_id: str
    model: str | None
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    estimated_cost_usd: float | None = None
