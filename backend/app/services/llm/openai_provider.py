"""OpenAI Responses API streaming provider."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from typing import Any

import openai
from openai import AsyncOpenAI

from app.config import Settings
from app.services.llm import prompt_cache
from app.services.llm.base import (
    ChatTurn,
    LLMComplete,
    LLMDelta,
    LLMDone,
    LLMError,
    LLMEvent,
    LLMIncomplete,
    LLMRefusal,
    LLMStructured,
    LLMUsage,
)

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
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[LLMEvent]:
        if not self.settings.llm_configured:
            yield LLMError(
                message=(
                    "OpenAI API key is not configured. "
                    "Add OPENAI_API_KEY to backend/.env and restart the API."
                ),
                request_sent=False,
            )
            return

        selected_model = model or self.settings.openai_model
        kwargs: dict[str, Any] = {
            "model": selected_model,
            "instructions": instructions or "",
            "input": [{"role": turn.role, "content": turn.content} for turn in turns],
            "stream": True,
            "max_output_tokens": self.settings.openai_max_output_tokens,
            **prompt_cache.request_params(selected_model),
        }
        effort = (reasoning_effort or self.settings.openai_reasoning_effort).strip().lower()
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
        usage: LLMUsage | None = None
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
                elif event_type in {"response.completed", "response.incomplete"}:
                    # Streaming usage arrives only on the terminal response object.
                    final = getattr(event, "response", None)
                    model_name = _text(getattr(final, "model", None)) or model_name
                    response_id = _text(getattr(final, "id", None)) or response_id
                    usage = extract_usage(getattr(final, "usage", None))
                elif event_type in {"response.failed", "error"}:
                    failed = getattr(event, "response", None)
                    yield LLMError(
                        message=_event_error(event),
                        provider_response_id=_text(getattr(failed, "id", None)) or response_id,
                        usage=extract_usage(getattr(failed, "usage", None)),
                    )
                    return
            yield LLMDone(
                model=model_name,
                usage=usage,
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

    async def complete(
        self,
        turns: list[ChatTurn],
        *,
        model: str | None = None,
        instructions: str | None = None,
        reasoning_effort: str | None = None,
        max_output_tokens: int | None = None,
    ) -> LLMComplete | LLMError:
        """Non-streaming Responses call for summarization and other side tasks."""

        if not self.settings.llm_configured:
            return LLMError(
                message=(
                    "OpenAI API key is not configured. "
                    "Add OPENAI_API_KEY to backend/.env and restart the API."
                ),
                request_sent=False,
            )

        selected_model = model or self.settings.openai_model
        kwargs: dict[str, Any] = {
            "model": selected_model,
            "instructions": instructions or "",
            "input": [{"role": turn.role, "content": turn.content} for turn in turns],
            "stream": False,
            "store": False,
            "max_output_tokens": max_output_tokens or self.settings.openai_max_output_tokens,
            **prompt_cache.request_params(selected_model),
        }
        effort = (reasoning_effort or self.settings.openai_reasoning_effort).strip().lower()
        if effort:
            if effort not in REASONING_EFFORTS:
                logger.warning("ignoring unsupported reasoning effort=%s", effort)
            else:
                kwargs["reasoning"] = {"effort": effort}

        try:
            response = await self._get_client().responses.create(**kwargs)
        except Exception as exc:
            logger.warning(
                "openai_complete_failed model=%s status=%s detail=%s",
                selected_model,
                getattr(exc, "status_code", None),
                _clip(self._redact(str(exc))),
            )
            return LLMError(message=self._public_error(exc, selected_model))

        text = _response_text(response)
        return LLMComplete(
            text=text,
            model=_text(getattr(response, "model", None)) or selected_model,
            usage=extract_usage(getattr(response, "usage", None)),
            provider_response_id=_text(getattr(response, "id", None)),
        )

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
        """Schema-enforced Structured Outputs via Responses API parse (strict json_schema).

        Does not fall back to unstructured JSON mode when the model or call fails.
        """

        if not self.settings.llm_configured:
            return LLMError(
                message=(
                    "OpenAI API key is not configured. "
                    "Add OPENAI_API_KEY to backend/.env and restart the API."
                ),
                request_sent=False,
            )

        selected_model = model or self.settings.openai_model
        kwargs: dict[str, Any] = {
            "model": selected_model,
            "instructions": instructions or "",
            "input": [{"role": turn.role, "content": turn.content} for turn in turns],
            "text_format": text_format,
            "store": False,
            "max_output_tokens": max_output_tokens or self.settings.openai_max_output_tokens,
            **prompt_cache.request_params(selected_model),
        }
        effort = (reasoning_effort or self.settings.openai_reasoning_effort).strip().lower()
        if effort:
            if effort not in REASONING_EFFORTS:
                logger.warning("ignoring unsupported reasoning effort=%s", effort)
            else:
                kwargs["reasoning"] = {"effort": effort}

        try:
            response = await self._get_client().responses.parse(**kwargs)
        except openai.LengthFinishReasonError as exc:
            usage = extract_usage(getattr(exc, "usage", None))
            return LLMIncomplete(
                message=(
                    "The structured response was cut off before it was complete "
                    "(max output tokens). Raise OPENAI_MAX_OUTPUT_TOKENS and retry."
                ),
                reason="max_output_tokens",
                model=selected_model,
                usage=usage,
            )
        except Exception as exc:
            logger.warning(
                "openai_parse_failed model=%s status=%s detail=%s",
                selected_model,
                getattr(exc, "status_code", None),
                _clip(self._redact(str(exc))),
            )
            return LLMError(message=self._public_error(exc, selected_model))

        model_name = _text(getattr(response, "model", None)) or selected_model
        response_id = _text(getattr(response, "id", None))
        usage = extract_usage(getattr(response, "usage", None))
        status = _text(getattr(response, "status", None))

        if status == "incomplete":
            details = getattr(response, "incomplete_details", None)
            reason = _text(getattr(details, "reason", None)) if details is not None else None
            return LLMIncomplete(
                message=_incomplete_message(reason),
                reason=reason,
                model=model_name,
                usage=usage,
                provider_response_id=response_id,
            )

        refusal = _response_refusal(response)
        if refusal is not None:
            return LLMRefusal(
                message=refusal,
                model=model_name,
                usage=usage,
                provider_response_id=response_id,
            )

        parsed = getattr(response, "output_parsed", None)
        if parsed is None:
            return LLMIncomplete(
                message="The model returned no schema-valid structured output.",
                reason="missing_output",
                model=model_name,
                usage=usage,
                provider_response_id=response_id,
            )

        if not isinstance(parsed, text_format):
            try:
                parsed = text_format.model_validate(parsed)
            except Exception:
                return LLMError(
                    message="The structured response could not be validated against the schema.",
                    provider_response_id=response_id,
                    usage=usage,
                )

        return LLMStructured(
            data=parsed,
            model=model_name,
            usage=usage,
            provider_response_id=response_id,
        )

    async def check_model_access(self, model_id: str) -> bool | None:
        """GET /v1/models/{id}: free, and confirms this key can see the model."""

        if not self.settings.llm_configured:
            return None
        try:
            await self._get_client().models.retrieve(model_id)
        except Exception as exc:
            status = getattr(exc, "status_code", None)
            if status in {403, 404}:
                return False
            logger.warning(
                "openai_model_check_failed model=%s status=%s detail=%s",
                model_id,
                status,
                _clip(self._redact(str(exc))),
            )
            return None
        return True

    def _public_error(self, exc: Exception, model: str) -> str:
        status = getattr(exc, "status_code", None)
        if status == 401 or isinstance(exc, openai.AuthenticationError):
            return "OpenAI rejected the API key. Update OPENAI_API_KEY in backend/.env and restart the API."
        if status == 403:
            return f"This OpenAI project cannot use the model '{model}'. Choose another model."
        if status == 404:
            return (
                f"The model '{model}' was not found for this API key. "
                "Choose another model, or turn it off with DISABLED_MODELS in backend/.env."
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


def extract_usage(usage: Any) -> LLMUsage | None:
    """Read token counts from an SDK object or dict. Missing values stay null.

    Responses API shape: input_tokens, output_tokens,
    input_tokens_details.{cached_tokens, cache_write_tokens},
    output_tokens_details.reasoning_tokens. Chat Completions names are accepted too.
    """

    if usage is None:
        return None
    input_details = _first(usage, "input_tokens_details", "prompt_tokens_details")
    output_details = _first(usage, "output_tokens_details", "completion_tokens_details")
    return LLMUsage(
        input_tokens=_optional_int(_first(usage, "input_tokens", "prompt_tokens")),
        output_tokens=_optional_int(_first(usage, "output_tokens", "completion_tokens")),
        cached_input_tokens=_optional_int(_first(input_details, "cached_tokens")),
        cache_write_tokens=_optional_int(_first(input_details, "cache_write_tokens")),
        reasoning_tokens=_optional_int(_first(output_details, "reasoning_tokens")),
    )


def _first(source: Any, *names: str) -> Any:
    if source is None:
        return None
    for name in names:
        value = source.get(name) if isinstance(source, dict) else getattr(source, name, None)
        if value is not None:
            return value
    return None


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


def _response_text(response: Any) -> str:
    direct = getattr(response, "output_text", None)
    if isinstance(direct, str) and direct:
        return direct
    parts: list[str] = []
    for item in getattr(response, "output", None) or []:
        for block in getattr(item, "content", None) or []:
            text = getattr(block, "text", None)
            if isinstance(text, str) and text:
                parts.append(text)
    return "".join(parts)


def _response_refusal(response: Any) -> str | None:
    for item in getattr(response, "output", None) or []:
        for block in getattr(item, "content", None) or []:
            if getattr(block, "type", None) == "refusal":
                text = getattr(block, "refusal", None)
                if isinstance(text, str) and text.strip():
                    return text.strip()
    return None


def _incomplete_message(reason: str | None) -> str:
    if reason == "max_output_tokens":
        return (
            "The structured response was cut off before it was complete "
            "(max output tokens). Raise OPENAI_MAX_OUTPUT_TOKENS and retry."
        )
    if reason == "content_filter":
        return "The structured response was incomplete because of a content filter."
    return "The structured response was incomplete. You can retry."


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
