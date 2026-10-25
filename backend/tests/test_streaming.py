"""Streaming, persistence, retries, and usage metrics."""

import json

import pytest

from app.db.database import init_db
from app.db.repository import Repository
from app.main import create_app
from app.services.chat_service import ChatService
from tests.conftest import make_settings
from tests.fakes import FakeProvider


def parse_sse(body: str) -> list[dict]:
    events = []
    for block in body.split("\n\n"):
        data = "".join(line[6:] for line in block.split("\n") if line.startswith("data: "))
        if data:
            events.append(json.loads(data))
    return events


def test_stream_persists_context_metrics_and_title(client, provider: FakeProvider):
    chat_id = client.post("/api/chats", json={}).json()["id"]
    first = client.post(
        f"/api/chats/{chat_id}/messages",
        json={"content": "Payment latency is climbing after the deploy", "client_request_id": "client-req-1"},
    )
    assert first.status_code == 200
    events = parse_sse(first.text)
    assert [event["type"] for event in events] == ["meta", "delta", "delta", "done"]
    assert "".join(event["text"] for event in events if event["type"] == "delta") == "Hello operator"
    done = events[-1]["message"]
    assert done["status"] == "complete"
    assert done["model"] == "gpt-6-luna"
    assert done["input_tokens"] == 12
    assert done["output_tokens"] == 5
    assert done["latency_ms"] is not None
    assert events[0]["title"].startswith("Payment latency")

    detail = client.get(f"/api/chats/{chat_id}").json()
    assert detail["conversation"]["title"].startswith("Payment latency")
    assert detail["messages"][1]["input_tokens"] == 12

    second = client.post(
        f"/api/chats/{chat_id}/messages",
        json={"content": "What should we check next?", "client_request_id": "client-req-2"},
    )
    assert second.status_code == 200
    turns = provider.calls[1]["turns"]
    assert [turn.role for turn in turns] == ["user", "assistant", "user"]
    assert turns[0].content.startswith("Payment latency")
    assert turns[1].content == "Hello operator"
    assert turns[2].content == "What should we check next?"
    assert "OperateSolve" in provider.calls[0]["instructions"]


def test_duplicate_request_does_not_call_the_model_again(client, provider: FakeProvider):
    chat_id = client.post("/api/chats", json={}).json()["id"]
    payload = {"content": "Draft the status update", "client_request_id": "client-req-dup"}
    first = client.post(f"/api/chats/{chat_id}/messages", json=payload)
    second = client.post(f"/api/chats/{chat_id}/messages", json=payload)
    assert first.status_code == 200
    assert second.status_code == 200
    assert len(provider.calls) == 1
    replay = parse_sse(second.text)
    assert replay[0]["replay"] is True
    assert replay[-1]["type"] == "done"
    assert replay[-1]["message"]["content"] == "Hello operator"
    assert len(client.get(f"/api/chats/{chat_id}").json()["messages"]) == 2


def test_missing_usage_is_stored_as_null(client, provider: FakeProvider):
    provider.input_tokens = None
    provider.output_tokens = None
    chat_id = client.post("/api/chats", json={}).json()["id"]
    response = client.post(
        f"/api/chats/{chat_id}/messages",
        json={"content": "Summarize the runbook gap", "client_request_id": "client-req-usage"},
    )
    message = parse_sse(response.text)[-1]["message"]
    assert message["input_tokens"] is None
    assert message["output_tokens"] is None
    assert message["model"] == "gpt-6-luna"
    stored = client.get(f"/api/chats/{chat_id}").json()["messages"][1]
    assert stored["input_tokens"] is None
    assert stored["output_tokens"] is None


def test_retry_after_failure_and_keeps_custom_title(client, provider: FakeProvider):
    chat_id = client.post("/api/chats", json={"title": "Keep me"}).json()["id"]
    provider.fail = True
    failed = client.post(
        f"/api/chats/{chat_id}/messages",
        json={"content": "super-secret-incident-note-8841", "client_request_id": "client-req-fail"},
    )
    error_event = parse_sse(failed.text)[-1]
    assert error_event["type"] == "error"
    assistant_id = error_event["message"]["id"]
    assert error_event["message"]["status"] == "error"

    provider.fail = False
    provider.chunks = ["Recovered"]
    retried = client.post(f"/api/chats/{chat_id}/messages/{assistant_id}/retry")
    assert retried.status_code == 200
    done = parse_sse(retried.text)[-1]["message"]
    assert done["status"] == "complete"
    assert done["content"] == "Recovered"
    assert len(provider.calls) == 2
    assert [turn.role for turn in provider.calls[1]["turns"]] == ["user"]
    assert client.get(f"/api/chats/{chat_id}").json()["conversation"]["title"] == "Keep me"


def test_streaming_conflict_and_validation(client, app, provider: FakeProvider):
    repo = Repository(app.state.settings.database_path)
    chat = repo.create_conversation(None)
    repo.insert_exchange(chat["id"], "already running", "client-req-running", None)
    conflict = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"content": "Another question", "client_request_id": "client-req-other"},
    )
    assert conflict.status_code == 409
    assert provider.calls == []

    invalid = client.post(
        f"/api/chats/{chat['id']}/messages",
        json={"content": "   ", "client_request_id": "short"},
    )
    assert invalid.status_code == 422


def test_logs_omit_message_content(client, caplog):
    caplog.set_level("INFO")
    chat_id = client.post("/api/chats", json={}).json()["id"]
    secret = "super-secret-incident-note-8841"
    response = client.post(
        f"/api/chats/{chat_id}/messages",
        json={"content": secret, "client_request_id": "client-req-log"},
    )
    assert response.status_code == 200
    assert secret not in caplog.text
    assert "generation_finished" in caplog.text
    assert "input_tokens=12" in caplog.text


def test_startup_marks_orphaned_stream_as_retryable(tmp_path, provider: FakeProvider):
    settings = make_settings(tmp_path)
    create_app(settings, provider=provider)
    repo = Repository(settings.database_path)
    chat = repo.create_conversation(None)
    _, assistant = repo.insert_exchange(chat["id"], "hello there", "client-req-orphan", None)
    assert assistant["status"] == "streaming"
    create_app(settings, provider=provider)
    restored = repo.get_message(assistant["id"])
    assert restored is not None
    assert restored["status"] == "error"
    assert "interrupted" in (restored["error"] or "").lower()


@pytest.mark.asyncio
async def test_client_disconnect_persists_partial_response(tmp_path):
    provider = FakeProvider()
    provider.chunks = ["Hello ", "operator"]
    settings = make_settings(tmp_path)
    init_db(settings.database_path)
    repo = Repository(settings.database_path)
    service = ChatService(repo, provider, settings)
    chat = repo.create_conversation(None)
    state = {"checks": 0}

    async def is_disconnected() -> bool:
        state["checks"] += 1
        return state["checks"] > 1

    prepared = service.prepare_new_message(chat["id"], "Question for the copilot", "client-req-stop")
    chunks: list[str] = []
    async for chunk in service.generate(prepared, is_disconnected, "local-operator"):
        chunks.append(chunk)

    messages = repo.list_messages(chat["id"])
    assert messages[-1]["status"] == "stopped"
    assert messages[-1]["content"] == "Hello "
    assert not any('"type": "done"' in chunk for chunk in chunks)
