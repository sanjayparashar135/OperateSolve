"""In-memory model provider for API tests."""

import asyncio
import time
from typing import Any

from app.services.llm.base import (
    ChatTurn,
    LLMComplete,
    LLMDelta,
    LLMDone,
    LLMError,
    LLMIncomplete,
    LLMRefusal,
    LLMStructured,
    LLMUsage,
)


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.complete_calls: list[dict] = []
        self.parse_calls: list[dict] = []
        self.chunks = ["Hello ", "operator"]
        self.fail = False
        self.request_sent = True
        self.input_tokens: int | None = 12
        self.output_tokens: int | None = 5
        self.cached_input_tokens: int | None = None
        self.cache_write_tokens: int | None = None
        self.reasoning_tokens: int | None = None
        # None echoes the requested model, as the real API does.
        self.model: str | None = None
        self.fail_models: set[str] = set()
        self.summary_text = (
            "Facts: prior context\n"
            "Goals/preferences: none\n"
            "Decisions/constraints: none\n"
            "Open questions/commitments: none\n"
            "Technical details: none"
        )
        self.summary_fail = False
        self.summary_input_tokens: int | None = 40
        self.summary_output_tokens: int | None = 20
        self.structured_result: Any | None = None
        self.parse_mode: str = "ok"  # ok | refuse | incomplete | error | invalid
        self.parse_input_tokens: int | None = 80
        self.parse_output_tokens: int | None = 40
        self.parse_cached_input_tokens: int | None = 10
        self.parse_error_message = "The model request failed. You can retry."
        self.parse_refusal_message = "I'm sorry, I cannot assist with that request."
        self.parse_incomplete_reason = "max_output_tokens"
        # Phase 2.5B experiment controls for complete().
        self.complete_delay_seconds: float = 0.0
        self.complete_fail = False
        self.complete_fail_substrings: list[str] = []
        self.complete_hang_substrings: list[str] = []
        self.complete_input_tokens: int | None = 25
        self.complete_output_tokens: int | None = 15
        self._complete_active = 0
        self.max_complete_concurrency = 0
        self.complete_started_at: list[float] = []
        self.complete_finished_at: list[float] = []

    def usage(self) -> LLMUsage | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return LLMUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_input_tokens=self.cached_input_tokens,
            cache_write_tokens=self.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens,
        )

    async def stream(
        self,
        turns: list[ChatTurn],
        *,
        model: str | None = None,
        instructions: str | None = None,
        reasoning_effort: str | None = None,
    ):
        self.calls.append(
            {
                "turns": list(turns),
                "model": model,
                "instructions": instructions,
                "reasoning_effort": reasoning_effort,
            }
        )
        if self.fail or model in self.fail_models:
            yield LLMError(message="The model request failed. You can retry.", request_sent=self.request_sent)
            return
        for chunk in self.chunks:
            yield LLMDelta(text=chunk)
        yield LLMDone(
            model=self.model or model,
            usage=self.usage(),
            provider_response_id="resp_test",
        )

    async def complete(
        self,
        turns: list[ChatTurn],
        *,
        model: str | None = None,
        instructions: str | None = None,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMComplete | LLMError:
        content = turns[0].content if turns else ""
        self.complete_calls.append(
            {
                "turns": list(turns),
                "model": model,
                "instructions": instructions,
                "reasoning_effort": reasoning_effort,
                "max_output_tokens": max_output_tokens,
            }
        )
        self.complete_started_at.append(time.perf_counter())
        self._complete_active += 1
        self.max_complete_concurrency = max(self.max_complete_concurrency, self._complete_active)
        try:
            if any(needle in content for needle in self.complete_hang_substrings):
                await asyncio.sleep(3600)
            if self.complete_delay_seconds > 0:
                await asyncio.sleep(self.complete_delay_seconds)
            if self.summary_fail or self.complete_fail or model in self.fail_models:
                return LLMError(message="Summarization failed.", request_sent=True)
            if any(needle in content for needle in self.complete_fail_substrings):
                return LLMError(message="Experiment task failed.", request_sent=True)

            # Experiment prompts get task-specific text; summarization keeps the checkpoint format.
            if content.startswith("Task:"):
                text = self._experiment_text(content)
                input_tokens = self.complete_input_tokens
                output_tokens = self.complete_output_tokens
            else:
                text = self.summary_text
                input_tokens = self.summary_input_tokens
                output_tokens = self.summary_output_tokens

            return LLMComplete(
                text=text,
                model=self.model or model,
                usage=LLMUsage(
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ),
                provider_response_id="resp_complete",
            )
        finally:
            self._complete_active -= 1
            self.complete_finished_at.append(time.perf_counter())

    def _experiment_text(self, content: str) -> str:
        if "probable root causes" in content:
            return "1. Saturated payment DB pool.\n2. Deploy rolled a bad connection limit.\n3. Retry storm amplified load."
        if "business impact" in content:
            return "Checkout conversion is degraded. Revenue risk is high in us-east-1. Urgent triage is warranted."
        if "remediation steps" in content:
            return "1. Roll back the 14:00 deploy.\n2. Raise pool limits temporarily.\n3. Confirm payment provider health (needs human approval)."
        return "Experiment task result."

    async def parse(
        self,
        turns: list[ChatTurn],
        *,
        text_format: type,
        model: str | None = None,
        instructions: str | None = None,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMStructured | LLMRefusal | LLMIncomplete | LLMError:
        self.parse_calls.append(
            {
                "turns": list(turns),
                "text_format": text_format,
                "model": model,
                "instructions": instructions,
                "reasoning_effort": reasoning_effort,
                "max_output_tokens": max_output_tokens,
            }
        )
        usage = LLMUsage(
            input_tokens=self.parse_input_tokens,
            output_tokens=self.parse_output_tokens,
            cached_input_tokens=self.parse_cached_input_tokens,
        )
        reported_model = self.model or model
        if self.parse_mode == "error" or model in self.fail_models:
            return LLMError(message=self.parse_error_message, request_sent=self.request_sent, usage=usage)
        if self.parse_mode == "refuse":
            return LLMRefusal(
                message=self.parse_refusal_message,
                model=reported_model,
                usage=usage,
                provider_response_id="resp_refuse",
            )
        if self.parse_mode == "incomplete":
            return LLMIncomplete(
                message="The structured response was cut off before it was complete.",
                reason=self.parse_incomplete_reason,
                model=reported_model,
                usage=usage,
                provider_response_id="resp_incomplete",
            )
        if self.parse_mode == "invalid":
            return LLMStructured(
                data={"not": "a valid incident"},
                model=reported_model,
                usage=usage,
                provider_response_id="resp_invalid",
            )

        payload = self.structured_result
        if payload is None:
            payload = {
                "incident_id": "INC-1001",
                "severity": "P2",
                "summary": "Checkout API latency spike affecting payments.",
                "probable_cause": "Saturated connection pool on the payments service.",
                "recommended_actions": [
                    "Scale payments workers",
                    "Drain stuck connections",
                    "Verify payment provider health",
                ],
                "requires_human_approval": True,
            }
        data = payload if isinstance(payload, text_format) else text_format.model_validate(payload)
        return LLMStructured(
            data=data,
            model=reported_model,
            usage=usage,
            provider_response_id="resp_structured",
        )
