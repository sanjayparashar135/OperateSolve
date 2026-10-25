"""OpenAI Responses streaming adapter."""

import logging
from types import SimpleNamespace

import pytest
from openai.types.responses import ResponseUsage

from app.logging_config import configure_logging
from app.services.llm.base import ChatTurn, LLMDelta, LLMDone, LLMError, LLMUsage
from app.services.llm.openai_provider import OpenAIResponsesProvider, extract_usage
from tests.conftest import make_settings


class AsyncEvents:
    def __init__(self, items):
        self._items = list(items)
        self.closed = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)

    async def close(self):
        self.closed = True


def event(**kwargs):
    return SimpleNamespace(**kwargs)


@pytest.mark.asyncio
async def test_maps_deltas_and_missing_usage(tmp_path):
    settings = make_settings(tmp_path, openai_api_key="sk-test-secret-value")
    provider = OpenAIResponsesProvider(settings)
    stream = AsyncEvents(
        [
            event(type="response.created", response=SimpleNamespace(id="resp_1", model="gpt-6-luna")),
            event(type="response.output_text.delta", delta="Hi"),
            event(type="response.completed", response=SimpleNamespace(id="resp_1", model="gpt-6-luna", usage=None)),
        ]
    )

    async def create(**kwargs):
        create.kwargs = kwargs
        return stream

    provider._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    events = [item async for item in provider.stream([ChatTurn("user", "Hello")], instructions="Be brief")]

    assert isinstance(events[0], LLMDelta)
    assert events[0].text == "Hi"
    assert isinstance(events[1], LLMDone)
    assert events[1].usage is None
    assert events[1].provider_response_id == "resp_1"
    assert create.kwargs["stream"] is True
    assert create.kwargs["model"] == "gpt-6-luna"
    assert create.kwargs["instructions"] == "Be brief"
    assert create.kwargs["input"] == [{"role": "user", "content": "Hello"}]
    assert create.kwargs["reasoning"] == {"effort": "low"}
    assert stream.closed is True


@pytest.mark.asyncio
async def test_streamed_usage_includes_cached_write_and_reasoning_tokens(tmp_path):
    provider = OpenAIResponsesProvider(make_settings(tmp_path, openai_api_key="sk-test-secret-value"))
    usage = ResponseUsage(
        input_tokens=12000,
        input_tokens_details={"cached_tokens": 4000, "cache_write_tokens": 7000},
        output_tokens=900,
        output_tokens_details={"reasoning_tokens": 300},
        total_tokens=12900,
    )
    stream = AsyncEvents(
        [
            event(type="response.created", response=SimpleNamespace(id="resp_2", model="gpt-6-luna")),
            event(type="response.output_text.delta", delta="Hi"),
            event(type="response.completed", response=SimpleNamespace(id="resp_2", model="gpt-6-luna-2026-05-18", usage=usage)),
        ]
    )

    async def create(**kwargs):
        return stream

    provider._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    events = [item async for item in provider.stream([ChatTurn("user", "Hello")])]

    done = events[-1]
    assert isinstance(done, LLMDone)
    assert done.model == "gpt-6-luna-2026-05-18"
    assert done.usage == LLMUsage(
        input_tokens=12000,
        output_tokens=900,
        cached_input_tokens=4000,
        cache_write_tokens=7000,
        reasoning_tokens=300,
    )
    assert done.usage.reported is True


def test_extract_usage_accepts_dicts_and_partial_payloads():
    assert extract_usage(None) is None
    chat_style = extract_usage(
        {"prompt_tokens": 50, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 32}}
    )
    assert chat_style == LLMUsage(input_tokens=50, output_tokens=7, cached_input_tokens=32)
    partial = extract_usage({"input_tokens": 10})
    assert partial is not None
    assert partial.output_tokens is None
    assert partial.cached_input_tokens is None
    assert partial.reported is False


@pytest.mark.asyncio
async def test_failed_response_carries_reported_usage(tmp_path):
    provider = OpenAIResponsesProvider(make_settings(tmp_path, openai_api_key="sk-test-secret-value"))
    failed = SimpleNamespace(
        id="resp_3",
        error=SimpleNamespace(message="server overloaded"),
        usage={"input_tokens": 40, "output_tokens": 0, "input_tokens_details": {"cached_tokens": 0}},
    )
    stream = AsyncEvents([event(type="response.failed", response=failed)])

    async def create(**kwargs):
        return stream

    provider._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    events = [item async for item in provider.stream([ChatTurn("user", "Hello")])]

    assert isinstance(events[0], LLMError)
    assert events[0].request_sent is True
    assert events[0].provider_response_id == "resp_3"
    assert events[0].usage == LLMUsage(input_tokens=40, output_tokens=0, cached_input_tokens=0)


@pytest.mark.asyncio
async def test_missing_key_does_not_call_openai(tmp_path):
    provider = OpenAIResponsesProvider(make_settings(tmp_path, openai_api_key=""))
    events = [item async for item in provider.stream([ChatTurn("user", "Hello")])]
    assert isinstance(events[0], LLMError)
    assert "OPENAI_API_KEY" in events[0].message
    assert events[0].request_sent is False


@pytest.mark.asyncio
async def test_routed_model_and_reasoning_effort_are_sent(tmp_path):
    provider = OpenAIResponsesProvider(make_settings(tmp_path, openai_api_key="sk-test-secret-value"))
    stream = AsyncEvents([event(type="response.output_text.delta", delta="Hi")])

    async def create(**kwargs):
        create.kwargs = kwargs
        return stream

    provider._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    events = [
        item
        async for item in provider.stream([ChatTurn("user", "Hello")], model="gpt-5-nano", reasoning_effort="minimal")
    ]
    assert create.kwargs["model"] == "gpt-5-nano"
    assert create.kwargs["reasoning"] == {"effort": "minimal"}
    assert create.kwargs["max_output_tokens"] == 256
    assert isinstance(events[-1], LLMDone)
    assert events[-1].model == "gpt-5-nano"


@pytest.mark.asyncio
async def test_check_model_access(tmp_path):
    provider = OpenAIResponsesProvider(make_settings(tmp_path, openai_api_key="sk-test-secret-value"))

    class NotFound(Exception):
        status_code = 404

    class Overloaded(Exception):
        status_code = 503

    async def retrieve(model_id):
        if model_id == "missing":
            raise NotFound()
        if model_id == "flaky":
            raise Overloaded()
        return SimpleNamespace(id=model_id)

    provider._client = SimpleNamespace(models=SimpleNamespace(retrieve=retrieve))
    assert await provider.check_model_access("gpt-6-luna") is True
    assert await provider.check_model_access("missing") is False
    assert await provider.check_model_access("flaky") is None
    assert await OpenAIResponsesProvider(make_settings(tmp_path)).check_model_access("gpt-6-luna") is None


@pytest.mark.asyncio
async def test_auth_errors_are_redacted(tmp_path, caplog):
    secret = "sk-test-secret-value"
    settings = make_settings(tmp_path, openai_api_key=secret)
    configure_logging("INFO", [secret])
    caplog.set_level(logging.WARNING)
    provider = OpenAIResponsesProvider(settings)

    class Unauthorized(Exception):
        status_code = 401

        def __str__(self) -> str:
            return f"invalid api key {secret}"

    async def create(**kwargs):
        raise Unauthorized()

    provider._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    events = [item async for item in provider.stream([ChatTurn("user", "Hello")])]
    assert isinstance(events[0], LLMError)
    assert secret not in events[0].message
    assert "API key" in events[0].message
    assert secret not in caplog.text
    assert "[redacted]" in caplog.text
