"""OpenAI Responses API streaming provider."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import openai
from openai import AsyncOpenAI

from app.config import Settings
from app.services.llm.base import ChatTurn, LLMDelta, LLMDone, LLMError, LLMEvent

logger = logging.getLogger(__name__)

REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh", "max"}


class OpenAIResponsesProvider:
    """Streams text from the OpenAI Responses API.

    The client is created lazily so tests and health checks do not need a key.
    """

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: AsyncOpenAI | None = None

    def _get_client(self) -> AsyncOpenAI:
        if self._client is None:
            self._client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        return self._client

    async def stream(
        self,
        turns: list[ChatTurn],
        *,
        model: str | None = None,
        instructions: str | None = None,
    ) -> AsyncIterator[LLMEvent]:
        if not self.settings.llm_configured:
            yield LLMError(
                message=(
                    "OpenAI API key is not configured. "
                    "Add OPENAI_API_KEY to backend/.env and restart the API."
                )
            )
            return

        selected_model = model or self.settings.openai_model
        kwargs: dict[str, Any] = {
            "model": selected_model,
            "instructions": instructions or "",
            "input": [{"role": turn.role, "content": turn.content} for turn in turns],
            "stream": True,
            "max_output_tokens": self.settings.openai_max_output_tokens,
        }
        effort = self.settings.openai_reasoning_effort.strip().lower()
        if effort:
            if effort not in REASONING_EFFORTS:
                logger.warning("ignoring unsupported reasoning effort=%s", effort)
            else:
                kwargs["reasoning"] = {"effort": effort}

        try:
            response_stream = await self._get_client().responses.create(**kwargs)
        except Exception as exc:
            logger.warning(
                "openai_request_failed model=%s status=%s detail=%s",
                selected_model,
                getattr(exc, "status_code", None),
                _clip(self._redact(str(exc))),
            )
            yield LLMError(message=self._public_error(exc, selected_model))
            return

        model_name = selected_model
        response_id: str | None = None
        input_tokens: int | None = None
        output_tokens: int | None = None
        try:
            async for event in response_stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.created":
                    created = getattr(event, "response", None)
                    response_id = _text(getattr(created, "id", None)) or response_id
                    model_name = _text(getattr(created, "model", None)) or model_name
                elif event_type == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    if delta:
                        yield LLMDelta(text=delta)
                elif event_type == "response.completed":
                    completed = getattr(event, "response", None)
                    model_name = _text(getattr(completed, "model", None)) or model_name
                    response_id = _text(getattr(completed, "id", None)) or response_id
                    input_tokens, output_tokens = extract_usage(getattr(completed, "usage", None))
                elif event_type == "response.incomplete":
                    incomplete = getattr(event, "response", None)
                    model_name = _text(getattr(incomplete, "model", None)) or model_name
                    response_id = _text(getattr(incomplete, "id", None)) or response_id
                    input_tokens, output_tokens = extract_usage(getattr(incomplete, "usage", None))
                elif event_type in {"response.failed", "error"}:
                    yield LLMError(
                        message=_event_error(event),
                        provider_response_id=response_id,
                    )
                    return
            yield LLMDone(
                model=model_name,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                provider_response_id=response_id,
            )
        except Exception as exc:
            logger.warning(
                "openai_stream_failed model=%s status=%s detail=%s",
                selected_model,
                getattr(exc, "status_code", None),
                _clip(self._redact(str(exc))),
            )
            yield LLMError(
                message=self._public_error(exc, selected_model),
                provider_response_id=response_id,
            )
        finally:
            await _close_stream(response_stream)

    def _public_error(self, exc: Exception, model: str) -> str:
        status = getattr(exc, "status_code", None)
        if status == 401 or isinstance(exc, openai.AuthenticationError):
            return "OpenAI rejected the API key. Update OPENAI_API_KEY in backend/.env and restart the API."
        if status == 403:
            return "This OpenAI project cannot use the configured model."
        if status == 404:
            return (
                f"The model '{model}' was not found. "
                "Set OPENAI_MODEL in backend/.env to a model this API key can access."
            )
        if status == 429 or isinstance(exc, openai.RateLimitError):
            return "OpenAI rate limit reached. Wait a moment and retry."
        if isinstance(exc, openai.APIConnectionError):
            return "Could not reach the OpenAI API. Check the network connection and retry."
        return "The model request failed. You can retry."

    def _redact(self, text: str) -> str:
        secret = self.settings.openai_api_key.strip()
        if len(secret) >= 12:
            return text.replace(secret, "[redacted]")
        return text


def extract_usage(usage: Any) -> tuple[int | None, int | None]:
    """Read token counts from an SDK object or dict. Missing values stay null."""

    if usage is None:
        return None, None
    if isinstance(usage, dict):
        raw_input = usage.get("input_tokens", usage.get("prompt_tokens"))
        raw_output = usage.get("output_tokens", usage.get("completion_tokens"))
    else:
        raw_input = getattr(usage, "input_tokens", None)
        if raw_input is None:
            raw_input = getattr(usage, "prompt_tokens", None)
        raw_output = getattr(usage, "output_tokens", None)
        if raw_output is None:
            raw_output = getattr(usage, "completion_tokens", None)
    return _optional_int(raw_input), _optional_int(raw_output)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str | None:
    if isinstance(value, str) and value:
        return value
    return None


def _event_error(event: Any) -> str:
    response = getattr(event, "response", None)
    error = getattr(event, "error", None) or getattr(response, "error", None)
    message = getattr(event, "message", None) or getattr(error, "message", None)
    if isinstance(message, str) and message.strip():
        return _clip(message.strip())
    return "The model failed to complete this response. You can retry."


def _clip(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


async def _close_stream(stream: Any) -> None:
    for name in ("close", "aclose"):
        closer = getattr(stream, name, None)
        if closer is None:
            continue
        try:
            result = closer()
            if hasattr(result, "__await__"):
                await result
        except Exception:
            logger.warning("openai_stream_close_failed")
        return
