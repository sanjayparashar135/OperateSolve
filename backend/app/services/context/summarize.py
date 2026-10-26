"""Incremental conversation summarization for excluded sliding-window history.

Summaries are stored per conversation with a checkpoint cursor. They are injected
as untrusted user-visible context, never as system instructions, and never shown
as assistant chat replies.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.db.database import utc_now
from app.db.repository import Repository
from app.services.context.budget import ContextBudget
from app.services.context.tokens import estimate_tokens
from app.services.context.window import ContextPlan, HistoryMessage, build_context_plan
from app.services.llm.base import ChatTurn, LLMComplete, LLMError, LLMProvider, LLMUsage
from app.services.llm.registry import ModelRegistry
from app.services.pricing import PricingTable, estimate_cost

logger = logging.getLogger(__name__)

SUMMARY_INSTRUCTIONS = (
    "You compress earlier turns of an OperateSolve operations chat into a durable "
    "continuity brief. Capture only: important facts and entities; user goals and "
    "preferences; decisions and constraints; unresolved questions and commitments; "
    "technical details needed later. Omit chatter, repetition, and anything not "
    "supported by the provided text. Do not invent telemetry, tickets, or impact. "
    "Output the sections below and nothing else.\n"
    "Facts:\n"
    "Goals/preferences:\n"
    "Decisions/constraints:\n"
    "Open questions/commitments:\n"
    "Technical details:"
)


@dataclass(frozen=True)
class SummarizationOutcome:
    plan: ContextPlan
    triggered: bool
    reused_existing: bool
    failed: bool
    failure_reason: str | None
    request_id: str | None
    model: str | None
    latency_ms: int | None
    usage: LLMUsage | None
    estimated_cost_usd: float | None
    checkpoint_position: int | None
    checkpoint_message_id: str | None
    summary_tokens: int | None


class ConversationSummarizer:
    def __init__(
        self,
        repository: Repository,
        provider: LLMProvider,
        settings: Settings,
        registry: ModelRegistry,
        pricing: PricingTable | None = None,
    ) -> None:
        self.repo = repository
        self.provider = provider
        self.settings = settings
        self.registry = registry
        self.pricing = pricing

    async def apply(
        self,
        *,
        conversation_id: str,
        instructions: str,
        messages: list[HistoryMessage],
        budget: ContextBudget,
        chat_message_id: str,
    ) -> SummarizationOutcome:
        """Build a context plan, summarizing excluded history at most once."""

        stored = self.repo.get_summary(conversation_id)
        if stored is not None and not self._summary_is_valid(stored, messages):
            logger.info(
                "summary_invalidated conversation_id=%s reason=stale_checkpoint",
                conversation_id,
            )
            self.repo.clear_summary(conversation_id)
            stored = None

        existing_text = stored["summary_text"] if stored else None
        covered_through = stored["covered_through_position"] if stored else None
        draft = build_context_plan(
            instructions=instructions,
            messages=messages,
            budget=budget,
            summary_text=existing_text,
            covered_through_position=covered_through,
        )

        if not draft.truncated or not self.settings.summarization_enabled:
            return SummarizationOutcome(
                plan=draft,
                triggered=False,
                reused_existing=bool(existing_text),
                failed=False,
                failure_reason=None,
                request_id=None,
                model=None,
                latency_ms=None,
                usage=None,
                estimated_cost_usd=None,
                checkpoint_position=stored["covered_through_position"] if stored else None,
                checkpoint_message_id=stored["covered_through_message_id"] if stored else None,
                summary_tokens=stored.get("summary_tokens") if stored else None,
            )

        newly_excluded = self._messages_to_summarize(draft.excluded, stored)
        excluded_tokens = sum(estimate_tokens(message.content) for message in newly_excluded)
        trigger = self.settings.summarization_trigger_excluded_tokens

        if not newly_excluded:
            return SummarizationOutcome(
                plan=draft,
                triggered=False,
                reused_existing=bool(existing_text),
                failed=False,
                failure_reason=None,
                request_id=None,
                model=None,
                latency_ms=None,
                usage=None,
                estimated_cost_usd=None,
                checkpoint_position=stored["covered_through_position"] if stored else None,
                checkpoint_message_id=stored["covered_through_message_id"] if stored else None,
                summary_tokens=stored.get("summary_tokens") if stored else None,
            )

        if existing_text is None and excluded_tokens < trigger:
            return SummarizationOutcome(
                plan=draft,
                triggered=False,
                reused_existing=False,
                failed=False,
                failure_reason=None,
                request_id=None,
                model=None,
                latency_ms=None,
                usage=None,
                estimated_cost_usd=None,
                checkpoint_position=None,
                checkpoint_message_id=None,
                summary_tokens=None,
            )

        result = await self._summarize_once(
            conversation_id=conversation_id,
            previous_summary=existing_text,
            excluded=newly_excluded,
            chat_message_id=chat_message_id,
        )
        if result.failed or not result.summary_text:
            bare = build_context_plan(
                instructions=instructions,
                messages=messages,
                budget=budget,
                summary_text=None,
            )
            return SummarizationOutcome(
                plan=bare,
                triggered=True,
                reused_existing=False,
                failed=True,
                failure_reason=result.failure_reason,
                request_id=result.request_id,
                model=result.model,
                latency_ms=result.latency_ms,
                usage=result.usage,
                estimated_cost_usd=result.estimated_cost_usd,
                checkpoint_position=None,
                checkpoint_message_id=None,
                summary_tokens=None,
            )

        plan = build_context_plan(
            instructions=instructions,
            messages=messages,
            budget=budget,
            summary_text=result.summary_text,
            covered_through_position=result.checkpoint_position,
        )
        return SummarizationOutcome(
            plan=plan,
            triggered=True,
            reused_existing=False,
            failed=False,
            failure_reason=None,
            request_id=result.request_id,
            model=result.model,
            latency_ms=result.latency_ms,
            usage=result.usage,
            estimated_cost_usd=result.estimated_cost_usd,
            checkpoint_position=result.checkpoint_position,
            checkpoint_message_id=result.checkpoint_message_id,
            summary_tokens=result.summary_tokens,
        )

    def _summary_is_valid(self, stored: dict[str, Any], messages: list[HistoryMessage]) -> bool:
        by_id = {message.id: message for message in messages}
        covered_id = stored["covered_through_message_id"]
        covered_pos = stored["covered_through_position"]
        message = by_id.get(covered_id)
        if message is None:
            return False
        if message.position != covered_pos:
            return False
        if messages and covered_pos >= messages[-1].position:
            return False
        return True

    def _messages_to_summarize(
        self,
        excluded: tuple[HistoryMessage, ...],
        stored: dict[str, Any] | None,
    ) -> list[HistoryMessage]:
        if not excluded:
            return []
        if stored is None:
            return list(excluded)
        return [message for message in excluded if message.position > stored["covered_through_position"]]

    async def _summarize_once(
        self,
        *,
        conversation_id: str,
        previous_summary: str | None,
        excluded: list[HistoryMessage],
        chat_message_id: str,
    ) -> _SummaryAttempt:
        model_id = self.settings.summarization_model.strip() or self.settings.openai_model
        spec = self.registry.get(model_id)
        if spec is None:
            return _SummaryAttempt(
                failed=True,
                failure_reason=f"Summarization model '{model_id}' is not registered.",
            )
        status = self.registry.status(spec)
        if not status.available:
            return _SummaryAttempt(
                failed=True,
                failure_reason=status.reason or f"Summarization model '{spec.model_id}' is unavailable.",
            )

        complete = getattr(self.provider, "complete", None)
        if complete is None:
            return _SummaryAttempt(
                failed=True,
                failure_reason="Provider does not support non-streaming completion for summarization.",
            )

        request_id = str(uuid.uuid4())
        prompt = _summary_user_prompt(previous_summary, excluded)
        effort = self.registry.reasoning_effort(spec, self.settings.summarization_reasoning_effort)
        max_out = min(
            self.settings.summarization_max_output_tokens,
            spec.max_output_tokens,
        )
        started_at = utc_now()
        started = time.perf_counter()
        logger.info(
            "summarization_started request_id=%s conversation_id=%s model=%s excluded_messages=%s",
            request_id,
            conversation_id,
            spec.model_id,
            len(excluded),
        )

        try:
            event = await complete(
                [ChatTurn(role="user", content=prompt)],
                model=spec.model_id,
                instructions=SUMMARY_INSTRUCTIONS,
                reasoning_effort=effort,
                max_output_tokens=max_out,
            )
        except Exception:
            logger.exception(
                "summarization_failed request_id=%s conversation_id=%s",
                request_id,
                conversation_id,
            )
            return _SummaryAttempt(
                failed=True,
                failure_reason="Summarization request failed.",
                request_id=request_id,
                model=spec.model_id,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        if isinstance(event, LLMError):
            if event.request_sent:
                self._record_usage(
                    request_id=request_id,
                    conversation_id=conversation_id,
                    message_id=chat_message_id,
                    model=spec.model_id,
                    status="error",
                    usage=event.usage,
                    duration_ms=latency_ms,
                    started_at=started_at,
                    provider_response_id=event.provider_response_id,
                )
            logger.warning(
                "summarization_fallback request_id=%s conversation_id=%s reason=%r",
                request_id,
                conversation_id,
                event.message,
            )
            return _SummaryAttempt(
                failed=True,
                failure_reason=event.message or "Summarization failed.",
                request_id=request_id,
                model=spec.model_id,
                latency_ms=latency_ms,
                usage=event.usage,
            )

        assert isinstance(event, LLMComplete)
        text = (event.text or "").strip()
        if not text:
            self._record_usage(
                request_id=request_id,
                conversation_id=conversation_id,
                message_id=chat_message_id,
                model=event.model or spec.model_id,
                status="error",
                usage=event.usage,
                duration_ms=latency_ms,
                started_at=started_at,
                provider_response_id=event.provider_response_id,
            )
            return _SummaryAttempt(
                failed=True,
                failure_reason="Summarization returned an empty response.",
                request_id=request_id,
                model=event.model or spec.model_id,
                latency_ms=latency_ms,
                usage=event.usage,
            )

        checkpoint = excluded[-1]
        summary_tokens = estimate_tokens(text)
        self.repo.upsert_summary(
            conversation_id,
            summary_text=text,
            covered_through_position=checkpoint.position,
            covered_through_message_id=checkpoint.id,
            summary_tokens=summary_tokens,
            model=event.model or spec.model_id,
        )
        cost = self._record_usage(
            request_id=request_id,
            conversation_id=conversation_id,
            message_id=chat_message_id,
            model=event.model or spec.model_id,
            status="complete",
            usage=event.usage,
            duration_ms=latency_ms,
            started_at=started_at,
            provider_response_id=event.provider_response_id,
        )
        logger.info(
            "summarization_finished request_id=%s conversation_id=%s model=%s latency_ms=%s "
            "input_tokens=%s output_tokens=%s summary_tokens=%s checkpoint_position=%s "
            "checkpoint_message_id=%s estimated_cost_usd=%s",
            request_id,
            conversation_id,
            event.model or spec.model_id,
            latency_ms,
            "unavailable" if event.usage is None or event.usage.input_tokens is None else event.usage.input_tokens,
            "unavailable" if event.usage is None or event.usage.output_tokens is None else event.usage.output_tokens,
            summary_tokens,
            checkpoint.position,
            checkpoint.id,
            "unavailable" if cost is None else f"{cost:.8f}",
        )
        return _SummaryAttempt(
            failed=False,
            summary_text=text,
            request_id=request_id,
            model=event.model or spec.model_id,
            latency_ms=latency_ms,
            usage=event.usage,
            estimated_cost_usd=cost,
            checkpoint_position=checkpoint.position,
            checkpoint_message_id=checkpoint.id,
            summary_tokens=summary_tokens,
        )

    def _record_usage(
        self,
        *,
        request_id: str,
        conversation_id: str,
        message_id: str,
        model: str | None,
        status: str,
        usage: LLMUsage | None,
        duration_ms: int,
        started_at: str,
        provider_response_id: str | None,
    ) -> float | None:
        input_tokens = usage.input_tokens if usage else None
        output_tokens = usage.output_tokens if usage else None
        try:
            self.repo.record_usage(
                request_id=request_id,
                conversation_id=conversation_id,
                message_id=message_id,
                provider=self.settings.llm_provider,
                requested_model=self.settings.summarization_model or model,
                model=model,
                status=status,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
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
                routing_reason="conversation summarization",
                first_token_ms=None,
                purpose="summarization",
            )
        except Exception:
            logger.exception("summarization_usage_persist_failed request_id=%s", request_id)
        if self.pricing is None or usage is None or not usage.reported:
            return None
        model_pricing = self.pricing.find(model)
        if model_pricing is None:
            return None
        cost = estimate_cost(
            model_pricing,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cached_input_tokens=usage.cached_input_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        )
        return float(cost.total)


@dataclass
class _SummaryAttempt:
    failed: bool
    failure_reason: str | None = None
    summary_text: str | None = None
    request_id: str | None = None
    model: str | None = None
    latency_ms: int | None = None
    usage: LLMUsage | None = None
    estimated_cost_usd: float | None = None
    checkpoint_position: int | None = None
    checkpoint_message_id: str | None = None
    summary_tokens: int | None = None


def _summary_user_prompt(previous_summary: str | None, excluded: list[HistoryMessage]) -> str:
    parts: list[str] = []
    if previous_summary:
        parts.append("Previous summary:\n" + previous_summary.strip())
        parts.append("Newly excluded turns to merge in (do not repeat the previous summary verbatim):")
    else:
        parts.append("Turns to summarize:")
    for message in excluded:
        label = "User" if message.role == "user" else "Assistant"
        parts.append(f"{label}: {message.content}")
    return "\n\n".join(parts)
