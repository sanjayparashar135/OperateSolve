"""OpenAI Responses streaming adapter."""

import logging
from types import SimpleNamespace

import pytest

from app.logging_config import configure_logging
from app.services.llm.base import ChatTurn, LLMDelta, LLMDone, LLMError
from app.services.llm.openai_provider import OpenAIResponsesProvider
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
    assert events[1].input_tokens is None
    assert events[1].output_tokens is None
    assert events[1].provider_response_id == "resp_1"
    assert create.kwargs["stream"] is True
    assert create.kwargs["model"] == "gpt-6-luna"
    assert create.kwargs["instructions"] == "Be brief"
    assert create.kwargs["input"] == [{"role": "user", "content": "Hello"}]
    assert create.kwargs["reasoning"] == {"effort": "low"}
    assert stream.closed is True


@pytest.mark.asyncio
async def test_missing_key_does_not_call_openai(tmp_path):
    provider = OpenAIResponsesProvider(make_settings(tmp_path, openai_api_key=""))
    events = [item async for item in provider.stream([ChatTurn("user", "Hello")])]
    assert isinstance(events[0], LLMError)
    assert "OPENAI_API_KEY" in events[0].message


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
