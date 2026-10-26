"""Structured incident analysis orchestration (Phase 2.5A).

Calls the provider with schema-enforced Structured Outputs. Does not store
conversations; usage is recorded under purpose=incident_analysis.
"""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any

from pydantic import ValidationError

from app.config import Settings
from app.db.database import utc_now
from app.db.repository import Repository
from app.schemas.incident import IncidentAnalysis
from app.services.llm.base import (
    ChatTurn,
    LLMError,
    LLMIncomplete,
    LLMRefusal,
    LLMStructured,
    LLMUsage,
)
from app.services.llm.registry import ModelRegistry
from app.services.pricing import PricingTable, estimate_cost

logger = logging.getLogger(__name__)

INSTRUCTIONS = (
    "You are an enterprise operations incident analyst for OperateSolve. "
    "Analyze the fictional incident details and return a structured assessment. "
    "Be concise and operational. Severity must be one of P1, P2, P3, or P4 "
    "(P1 = critical customer-facing outage). Set requires_human_approval to true "
    "whenever the recommended actions include irreversible changes, customer "
    "communication, or production data mutation."
)


class IncidentAnalysisError(Exception):
    """Raised when analysis cannot produce a validated structured result."""

    def __init__(self, message: str, *, code: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class IncidentService:
    def __init__(
        self,
        repository: Repository,
        provider: Any,
        settings: Settings,
        registry: ModelRegistry,
        pricing: PricingTable | None = None,
    ) -> None:
        self.repo = repository
        self.provider = provider
        self.settings = settings
        self.registry = registry
        self.pricing = pricing

    async def analyze(self, incident_id: str, description: str) -> dict[str, Any]:
        parse = getattr(self.provider, "parse", None)
        if parse is None:
            raise IncidentAnalysisError(
                "The configured LLM provider does not support Structured Outputs.",
                code="unsupported_provider",
                status_code=409,
            )

        model_id = self.settings.openai_model
        spec = self.registry.get(model_id)
        if spec is None:
            raise IncidentAnalysisError(
                f"Configured model '{model_id}' is not in the model registry.",
                code="unknown_model",
                status_code=409,
            )
        status = self.registry.status(spec)
        if not status.available:
            raise IncidentAnalysisError(
                status.reason or f"Model '{spec.model_id}' is unavailable.",
                code="model_unavailable",
                status_code=409,
            )
        if "structured_outputs" not in spec.capabilities:
            raise IncidentAnalysisError(
                (
                    f"Model '{spec.model_id}' does not support schema-enforced Structured Outputs. "
                    "Choose a model with the structured_outputs capability; "
                    "OperateSolve will not fall back to unstructured JSON."
                ),
                code="unsupported_model",
                status_code=409,
            )

        request_id = str(uuid.uuid4())
        started_at = utc_now()
        started = time.perf_counter()
        effort = self.registry.reasoning_effort(spec, self.settings.openai_reasoning_effort)
        user_prompt = (
            f"Incident ID: {incident_id}\n\n"
            f"Incident description:\n{description}\n\n"
            "Return the structured incident analysis. Use the same incident_id."
        )

        logger.info(
            "incident_analysis_started request_id=%s model=%s incident_id=%s",
            request_id,
            spec.model_id,
            incident_id,
        )

        result = await parse(
            [ChatTurn(role="user", content=user_prompt)],
            text_format=IncidentAnalysis,
            model=spec.model_id,
            instructions=INSTRUCTIONS,
            reasoning_effort=effort,
        )

        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = getattr(result, "usage", None)
        model_name = getattr(result, "model", None) or spec.model_id
        provider_response_id = getattr(result, "provider_response_id", None)

        if isinstance(result, LLMError):
            self._record_usage(
                request_id=request_id,
                model=model_name,
                status="error",
                usage=usage,
                duration_ms=latency_ms,
                started_at=started_at,
                provider_response_id=provider_response_id,
                routing_reason=result.message,
            )
            code = "not_configured" if not result.request_sent else "provider_error"
            status_code = 503 if not result.request_sent else 502
            raise IncidentAnalysisError(result.message, code=code, status_code=status_code)

        if isinstance(result, LLMRefusal):
            self._record_usage(
                request_id=request_id,
                model=model_name,
                status="refused",
                usage=usage,
                duration_ms=latency_ms,
                started_at=started_at,
                provider_response_id=provider_response_id,
                routing_reason="model refusal",
            )
            raise IncidentAnalysisError(
                result.message or "The model refused to analyze this incident.",
                code="refusal",
                status_code=422,
            )

        if isinstance(result, LLMIncomplete):
            self._record_usage(
                request_id=request_id,
                model=model_name,
                status="incomplete",
                usage=usage,
                duration_ms=latency_ms,
                started_at=started_at,
                provider_response_id=provider_response_id,
                routing_reason=result.reason or "incomplete",
            )
            raise IncidentAnalysisError(
                result.message,
                code="incomplete",
                status_code=422,
            )

        if not isinstance(result, LLMStructured):
            self._record_usage(
                request_id=request_id,
                model=model_name,
                status="error",
                usage=usage,
                duration_ms=latency_ms,
                started_at=started_at,
                provider_response_id=provider_response_id,
                routing_reason="unexpected provider result",
            )
            raise IncidentAnalysisError(
                "Unexpected provider result for structured incident analysis.",
                code="provider_error",
                status_code=502,
            )

        try:
            analysis = (
                result.data
                if isinstance(result.data, IncidentAnalysis)
                else IncidentAnalysis.model_validate(result.data)
            )
        except ValidationError:
            self._record_usage(
                request_id=request_id,
                model=model_name,
                status="error",
                usage=usage,
                duration_ms=latency_ms,
                started_at=started_at,
                provider_response_id=provider_response_id,
                routing_reason="schema validation failed",
            )
            raise IncidentAnalysisError(
                "The structured response could not be validated against the incident schema.",
                code="schema_invalid",
                status_code=502,
            ) from None

        # Prefer the caller-supplied id so the UI stays consistent even if the model echoes differently.
        if analysis.incident_id != incident_id:
            analysis = analysis.model_copy(update={"incident_id": incident_id})

        cost = self._record_usage(
            request_id=request_id,
            model=model_name,
            status="complete",
            usage=usage,
            duration_ms=latency_ms,
            started_at=started_at,
            provider_response_id=provider_response_id,
            routing_reason="structured incident analysis",
        )

        logger.info(
            "incident_analysis_finished request_id=%s model=%s latency_ms=%s "
            "severity=%s requires_human_approval=%s",
            request_id,
            model_name,
            latency_ms,
            analysis.severity,
            analysis.requires_human_approval,
        )

        return {
            "analysis": analysis.model_dump(),
            "request_id": request_id,
            "model": model_name,
            "latency_ms": latency_ms,
            "input_tokens": usage.input_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
            "cached_input_tokens": usage.cached_input_tokens if usage else None,
            "cache_write_tokens": usage.cache_write_tokens if usage else None,
            "reasoning_tokens": usage.reasoning_tokens if usage else None,
            "estimated_cost_usd": cost,
        }

    def _record_usage(
        self,
        *,
        request_id: str,
        model: str | None,
        status: str,
        usage: LLMUsage | None,
        duration_ms: int,
        started_at: str,
        provider_response_id: str | None,
        routing_reason: str | None,
    ) -> float | None:
        try:
            self.repo.record_usage(
                request_id=request_id,
                conversation_id=None,
                message_id=None,
                provider=self.settings.llm_provider,
                requested_model=self.settings.openai_model,
                model=model,
                status=status,
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                cached_input_tokens=usage.cached_input_tokens if usage else None,
                cache_write_tokens=usage.cache_write_tokens if usage else None,
                reasoning_tokens=usage.reasoning_tokens if usage else None,
                duration_ms=duration_ms,
                started_at=started_at,
                finished_at=utc_now(),
                provider_response_id=provider_response_id,
                selection=None,
                task_mode=None,
                routing_mode=None,
                routing_reason=routing_reason,
                first_token_ms=None,
                purpose="incident_analysis",
            )
        except Exception:
            logger.exception("incident_analysis_usage_persist_failed request_id=%s", request_id)

        if self.pricing is None or usage is None or not usage.reported:
            return None
        model_pricing = self.pricing.find(model)
        if model_pricing is None:
            return None
        estimate = estimate_cost(
            model_pricing,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cached_input_tokens=usage.cached_input_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        )
        return float(estimate.total)
