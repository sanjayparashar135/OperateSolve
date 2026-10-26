"""Request and response models for the Phase 2.5B parallel LLM experiment."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator

ExecutionMode = Literal["sequential", "parallel"]
TaskStatus = Literal["complete", "error", "timeout"]


class ParallelExperimentRequest(BaseModel):
    """Fictional incident description and chosen execution mode."""

    description: str = Field(min_length=1, max_length=20000)
    mode: ExecutionMode

    @field_validator("description")
    @classmethod
    def strip_description(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Value cannot be blank")
        return cleaned


class ParallelTaskResult(BaseModel):
    task_id: str
    task_name: str
    status: TaskStatus
    result: str | None = None
    error: str | None = None
    request_id: str
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = None


class ParallelExperimentTotals(BaseModel):
    latency_ms: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None = None
    successful_tasks: int
    failed_tasks: int


class ParallelExperimentResponse(BaseModel):
    mode: ExecutionMode
    model: str
    tasks: list[ParallelTaskResult]
    totals: ParallelExperimentTotals
