"""Per-request usage recording, dashboard aggregation, and the schema migration."""

import sqlite3
from datetime import datetime, timezone

import pytest

from app.db.database import SCHEMA, init_db
from app.db.repository import Repository
from app.main import create_app
from app.services.pricing import PricingTable, parse_pricing
from app.services.usage_service import aggregate_usage
from tests.conftest import make_service, make_settings
from tests.fakes import FakeProvider
from tests.test_streaming import parse_sse

PRICING = parse_pricing(
    {"models": {"gpt-6-luna": {"input": 0.10, "cached_input": 0.01, "cache_write": 0.125, "output": 0.50}}}
)


def send(client, chat_id: str, request_id: str, content: str = "Check the payment queue"):
    return client.post(f"/api/chats/{chat_id}/messages", json={"content": content, "client_request_id": request_id})


def usage_rows(app) -> list[dict]:
    return Repository(app.state.settings.database_path).list_usage()


def test_streamed_chat_records_usage_once_and_dashboard_reflects_it(client, app, provider: FakeProvider):
    provider.input_tokens = 2_000
    provider.cached_input_tokens = 1_024
    provider.cache_write_tokens = 512
    provider.output_tokens = 300
    provider.reasoning_tokens = 120
    chat_id = client.post("/api/chats", json={"title": "Queue backlog"}).json()["id"]

    response = send(client, chat_id, "client-req-usage-1")
    done = parse_sse(response.text)[-1]["message"]
    assert done["cached_input_tokens"] == 1_024

    # A replay of the same client request must not add a second usage row.
    send(client, chat_id, "client-req-usage-1")
    rows = usage_rows(app)
    assert len(rows) == 1
    row = rows[0]
    assert row["conversation_id"] == chat_id
    assert row["message_id"] == done["id"]
    assert row["request_id"] == done["request_id"]
    assert row["model"] == "gpt-6-luna"
    assert row["status"] == "complete"
    assert row["usage_reported"] is True
    assert (row["input_tokens"], row["cached_input_tokens"], row["cache_write_tokens"]) == (2_000, 1_024, 512)
    assert (row["output_tokens"], row["reasoning_tokens"]) == (300, 120)
    assert row["duration_ms"] is not None
    assert row["started_at"] <= row["finished_at"]
    assert row["source"] == "live"

    report = client.get("/api/usage").json()
    totals = report["totals"]
    assert totals["requests"] == 1
    assert totals["input_tokens"] == 2_000
    assert totals["cached_input_tokens"] == 1_024
    assert totals["cache_write_tokens"] == 512
    assert totals["output_tokens"] == 300
    # (464 * 0.10 + 1024 * 0.01 + 512 * 0.125 + 300 * 0.50) / 1M
    assert totals["estimated_cost_usd"] == pytest.approx(0.00027064)
    assert report["by_model"][0]["model"] == "gpt-6-luna"
    assert report["by_model"][0]["rates"]["input"] == pytest.approx(0.10)
    assert report["by_conversation"][0]["title"] == "Queue backlog"
    assert report["recent"][0]["estimated_cost_usd"] == pytest.approx(0.00027064)
    assert report["pricing"]["verified_at"]
    assert report["pricing"]["error"] is None


def test_each_retry_attempt_is_recorded(client, app, provider: FakeProvider):
    chat_id = client.post("/api/chats", json={}).json()["id"]
    provider.fail = True
    failed = parse_sse(send(client, chat_id, "client-req-retry-usage").text)[-1]
    provider.fail = False
    client.post(f"/api/chats/{chat_id}/messages/{failed['message']['id']}/retry")

    rows = usage_rows(app)
    assert sorted(row["status"] for row in rows) == ["complete", "error"]
    failed_row = next(row for row in rows if row["status"] == "error")
    assert failed_row["usage_reported"] is False
    totals = client.get("/api/usage").json()["totals"]
    assert totals["requests"] == 2
    assert totals["requests_missing_usage"] == 1
    assert totals["input_tokens"] == 12


def test_locally_refused_request_is_not_recorded(client, app, provider: FakeProvider):
    provider.fail = True
    provider.request_sent = False
    chat_id = client.post("/api/chats", json={}).json()["id"]
    send(client, chat_id, "client-req-no-key")
    assert usage_rows(app) == []


def test_missing_usage_is_recorded_but_not_costed(client, app, provider: FakeProvider):
    provider.input_tokens = None
    provider.output_tokens = None
    chat_id = client.post("/api/chats", json={}).json()["id"]
    send(client, chat_id, "client-req-missing")
    totals = client.get("/api/usage").json()["totals"]
    assert totals["requests"] == 1
    assert totals["requests_missing_usage"] == 1
    assert totals["input_tokens"] == 0
    assert totals["estimated_cost_usd"] is None


def test_usage_survives_conversation_delete(client, provider: FakeProvider):
    chat_id = client.post("/api/chats", json={"title": "Gone soon"}).json()["id"]
    send(client, chat_id, "client-req-delete")
    client.delete(f"/api/chats/{chat_id}")
    report = client.get("/api/usage").json()
    assert report["totals"]["requests"] == 1
    assert report["by_conversation"][0]["deleted"] is True
    assert report["by_conversation"][0]["title"] is None


def test_usage_days_parameter_is_validated(client):
    assert client.get("/api/usage?days=0").status_code == 422
    assert client.get("/api/usage?days=7").json()["days"] == 7


def test_broken_pricing_file_keeps_chat_working(tmp_path, provider: FakeProvider):
    bad = tmp_path / "pricing.json"
    bad.write_text("{not json", encoding="utf-8")
    from fastapi.testclient import TestClient

    app = create_app(make_settings(tmp_path, pricing_config_path=str(bad)), provider=provider)
    with TestClient(app) as client:
        chat_id = client.post("/api/chats", json={}).json()["id"]
        assert parse_sse(send(client, chat_id, "client-req-bad-pricing").text)[-1]["type"] == "done"
        report = client.get("/api/usage").json()
    assert report["pricing"]["error"]
    assert report["totals"]["requests"] == 1
    assert report["totals"]["requests_unpriced"] == 1
    assert report["totals"]["estimated_cost_usd"] is None


@pytest.mark.asyncio
async def test_stopped_stream_records_request_without_usage(tmp_path):
    provider = FakeProvider()
    settings = make_settings(tmp_path)
    init_db(settings.database_path)
    repo = Repository(settings.database_path)
    service = make_service(settings, provider, repo)
    chat = repo.create_conversation(None)
    state = {"checks": 0}

    async def is_disconnected() -> bool:
        state["checks"] += 1
        return state["checks"] > 1

    prepared = service.prepare_new_message(chat["id"], "Question", "client-req-stop-usage")
    async for _ in service.generate(prepared, is_disconnected, "local-operator"):
        pass

    rows = repo.list_usage()
    assert len(rows) == 1
    assert rows[0]["status"] == "stopped"
    assert rows[0]["usage_reported"] is False


def record(**overrides) -> dict:
    base = {
        "request_id": "r1",
        "conversation_id": "c1",
        "conversation_title": "Incident",
        "message_id": "m1",
        "provider": "openai",
        "requested_model": "gpt-6-luna",
        "model": "gpt-6-luna",
        "status": "complete",
        "input_tokens": 1_000_000,
        "output_tokens": 100_000,
        "cached_input_tokens": 0,
        "cache_write_tokens": 0,
        "reasoning_tokens": 0,
        "usage_reported": True,
        "duration_ms": 1000,
        "started_at": "2026-10-24T10:00:00+00:00",
        "finished_at": "2026-10-24T10:00:01+00:00",
        "source": "live",
        "purpose": "chat",
    }
    base.update(overrides)
    return base


def test_aggregation_by_model_conversation_and_day():
    now = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)
    records = [
        record(request_id="r4", model="mystery-model", conversation_id="c2", conversation_title=None, started_at="2026-10-25T09:00:00+00:00"),
        record(request_id="r3", usage_reported=False, input_tokens=None, output_tokens=None, cached_input_tokens=None, status="stopped", duration_ms=50, started_at="2026-10-25T08:00:00+00:00"),
        record(request_id="r2", cached_input_tokens=None, duration_ms=3000, source="backfill", started_at="2026-10-24T11:00:00+00:00"),
        record(request_id="r1", cached_input_tokens=500_000),
    ]
    report = aggregate_usage(records, PRICING, days=7, now=now)

    totals = report["totals"]
    assert totals["requests"] == 4
    assert totals["requests_with_usage"] == 3
    assert totals["requests_missing_usage"] == 1
    assert totals["requests_unpriced"] == 1
    assert totals["requests_cached_unknown"] == 1
    assert totals["requests_backfilled"] == 1
    assert totals["input_tokens"] == 3_000_000
    assert totals["cached_input_tokens"] == 500_000
    # r1: 0.5M*0.10 + 0.5M*0.01 + 0.1M*0.50 = 0.105 ; r2: 1M*0.10 + 0.1M*0.50 = 0.15
    assert totals["estimated_cost_usd"] == pytest.approx(0.255)
    # Stopped requests are excluded from latency; completed: 1000, 3000, 1000.
    assert totals["average_latency_ms"] == round(5000 / 3)
    assert totals["first_request_at"] == "2026-10-24T10:00:00+00:00"
    assert totals["last_request_at"] == "2026-10-25T09:00:00+00:00"

    models = {item["model"]: item for item in report["by_model"]}
    assert models["gpt-6-luna"]["requests"] == 3
    assert models["mystery-model"]["estimated_cost_usd"] is None
    assert models["mystery-model"]["rates"] is None

    conversations = {item["conversation_id"]: item for item in report["by_conversation"]}
    assert conversations["c1"]["requests"] == 3
    assert conversations["c1"]["estimated_cost_usd"] == pytest.approx(0.255)
    assert conversations["c2"]["deleted"] is True

    daily = {item["date"]: item for item in report["daily"]}
    assert len(report["daily"]) == 7
    assert daily["2026-10-24"]["requests"] == 2
    assert daily["2026-10-24"]["estimated_cost_usd"] == pytest.approx(0.255)
    assert daily["2026-10-25"]["requests"] == 2
    assert daily["2026-10-20"]["requests"] == 0
    assert [item["request_id"] for item in report["recent"]] == ["r4", "r3", "r2", "r1"]
    assert report["recent"][1]["estimated_cost_usd"] is None


def test_empty_aggregation_has_no_cost():
    report = aggregate_usage([], PricingTable(), days=None, now=datetime(2026, 10, 25, tzinfo=timezone.utc))
    assert report["totals"]["requests"] == 0
    assert report["totals"]["estimated_cost_usd"] is None
    assert report["totals"]["average_latency_ms"] is None
    assert len(report["daily"]) == 1


def test_phase1_database_is_migrated_and_backfilled_once(tmp_path):
    path = str(tmp_path / "phase1.db")
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    connection.executescript(
        """
        INSERT INTO conversations VALUES ('c1', 'Old chat', 0, '2026-10-01T00:00:00+00:00', '2026-10-01T00:00:00+00:00');
        INSERT INTO messages (id, conversation_id, position, role, content, status, client_request_id, created_at, updated_at)
            VALUES ('u1', 'c1', 1, 'user', 'hi', 'complete', 'client-1', '2026-10-01T00:00:00+00:00', '2026-10-01T00:00:00+00:00');
        INSERT INTO messages (id, conversation_id, position, role, content, status, model, latency_ms, input_tokens, output_tokens, request_id, provider_response_id, created_at, updated_at)
            VALUES ('a1', 'c1', 2, 'assistant', 'hello', 'complete', 'gpt-6-luna', 900, 40, 10, 'req-1', 'resp_1', '2026-10-01T00:00:00+00:00', '2026-10-01T00:00:02+00:00');
        INSERT INTO messages (id, conversation_id, position, role, content, status, model, latency_ms, request_id, error, created_at, updated_at)
            VALUES ('a2', 'c1', 4, 'assistant', '', 'error', 'gpt-6-luna', 0, 'req-2', 'OpenAI API key is not configured.', '2026-10-01T00:01:00+00:00', '2026-10-01T00:01:00+00:00');
        """
    )
    connection.commit()
    connection.close()

    init_db(path)
    init_db(path)

    repo = Repository(path)
    rows = repo.list_usage()
    assert len(rows) == 1
    assert rows[0]["request_id"] == "req-1"
    assert rows[0]["source"] == "backfill"
    assert rows[0]["input_tokens"] == 40
    assert rows[0]["cached_input_tokens"] is None
    assert rows[0]["conversation_title"] == "Old chat"
    assert repo.get_message("a1")["cached_input_tokens"] is None
    with sqlite3.connect(path) as check:
        assert check.execute("PRAGMA user_version").fetchone()[0] == 3
    conversation = repo.get_conversation("c1")
    assert (conversation["model_selection"], conversation["task_mode"]) == ("auto", "general")
    assert rows[0]["routing_mode"] is None
