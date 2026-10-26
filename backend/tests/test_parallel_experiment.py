"""Phase 2.5B — sequential vs parallel LLM experiment."""

from pathlib import Path

import pytest

from app.db.repository import Repository
from app.main import create_app
from app.services.pricing import load_pricing
from tests.conftest import make_settings
from tests.fakes import FakeProvider

INCIDENT = (
    "Payments p95 latency rose from 120ms to 4.5s after the 14:00 deploy. "
    "Checkout errors are elevated in us-east-1; auth and catalog look healthy."
)


def _post(client, mode: str, description: str = INCIDENT):
    return client.post(
        "/api/experiments/parallel",
        json={"description": description, "mode": mode},
    )


def test_sequential_runs_tasks_in_order(client, provider):
    provider.complete_delay_seconds = 0.05
    response = _post(client, "sequential")
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "sequential"
    assert body["model"] == "gpt-6-luna"
    assert [task["task_id"] for task in body["tasks"]] == [
        "root_causes",
        "business_impact",
        "remediation",
    ]
    assert all(task["status"] == "complete" for task in body["tasks"])
    assert all(task["result"] for task in body["tasks"])
    assert body["totals"]["successful_tasks"] == 3
    assert body["totals"]["failed_tasks"] == 0

    # Sequential: each call starts after the previous finished.
    assert len(provider.complete_started_at) == 3
    assert len(provider.complete_finished_at) == 3
    assert provider.max_complete_concurrency == 1
    for index in range(1, 3):
        assert provider.complete_started_at[index] >= provider.complete_finished_at[index - 1]

    # Wall clock should be roughly the sum of per-task latencies.
    sum_task_latency = sum(task["latency_ms"] for task in body["tasks"])
    assert body["totals"]["latency_ms"] >= sum_task_latency * 0.7


def test_parallel_runs_tasks_concurrently(client, provider):
    provider.complete_delay_seconds = 0.12
    response = _post(client, "parallel")
    assert response.status_code == 200
    body = response.json()
    assert body["mode"] == "parallel"
    assert [task["task_id"] for task in body["tasks"]] == [
        "root_causes",
        "business_impact",
        "remediation",
    ]
    assert all(task["status"] == "complete" for task in body["tasks"])
    assert provider.max_complete_concurrency == 3

    # Wall clock should be closer to one task than to the sum of three.
    max_task_latency = max(task["latency_ms"] for task in body["tasks"])
    sum_task_latency = sum(task["latency_ms"] for task in body["tasks"])
    assert body["totals"]["latency_ms"] < sum_task_latency * 0.8
    assert body["totals"]["latency_ms"] >= max_task_latency * 0.7


def test_aggregates_tokens_and_cost(client, provider, settings):
    provider.complete_input_tokens = 30
    provider.complete_output_tokens = 20
    response = _post(client, "sequential")
    assert response.status_code == 200
    body = response.json()
    assert body["totals"]["input_tokens"] == 90
    assert body["totals"]["output_tokens"] == 60

    pricing = load_pricing(settings.pricing_file)
    model_pricing = pricing.find("gpt-6-luna")
    assert model_pricing is not None
    expected = (
        (90 * float(model_pricing.standard.input) + 60 * float(model_pricing.standard.output))
        / 1_000_000
    )
    assert body["totals"]["estimated_cost_usd"] == pytest.approx(expected, rel=1e-6)
    for task in body["tasks"]:
        assert task["estimated_cost_usd"] is not None
        assert task["input_tokens"] == 30
        assert task["output_tokens"] == 20


def test_partial_failure_still_returns_other_tasks(client, provider):
    provider.complete_input_tokens = 30
    provider.complete_output_tokens = 20
    provider.complete_fail_substrings = ["business impact"]
    response = _post(client, "parallel")
    assert response.status_code == 200
    body = response.json()
    by_id = {task["task_id"]: task for task in body["tasks"]}
    assert by_id["root_causes"]["status"] == "complete"
    assert by_id["remediation"]["status"] == "complete"
    assert by_id["business_impact"]["status"] == "error"
    assert by_id["business_impact"]["error"]
    assert by_id["business_impact"]["result"] is None
    assert body["totals"]["successful_tasks"] == 2
    assert body["totals"]["failed_tasks"] == 1
    # Failed calls without reported usage contribute 0; successful ones still aggregate.
    assert body["totals"]["input_tokens"] == 60
    assert body["totals"]["output_tokens"] == 40


def test_timeout_marks_task_without_aborting_others(tmp_path: Path, provider: FakeProvider):
    settings = make_settings(
        tmp_path,
        parallel_experiment_timeout_seconds=0.05,
        parallel_experiment_max_concurrency=3,
    )
    provider.complete_hang_substrings = ["remediation steps"]
    provider.complete_delay_seconds = 0.02
    app = create_app(settings, provider=provider)
    with __import__("fastapi.testclient", fromlist=["TestClient"]).TestClient(app) as client:
        response = _post(client, "parallel")
    assert response.status_code == 200
    body = response.json()
    by_id = {task["task_id"]: task for task in body["tasks"]}
    assert by_id["root_causes"]["status"] == "complete"
    assert by_id["business_impact"]["status"] == "complete"
    assert by_id["remediation"]["status"] == "timeout"
    assert "timed out" in (by_id["remediation"]["error"] or "").lower()
    assert body["totals"]["successful_tasks"] == 2
    assert body["totals"]["failed_tasks"] == 1


def test_usage_recorded_once_per_task_no_duplication(client, provider, settings):
    response = _post(client, "sequential")
    assert response.status_code == 200
    body = response.json()
    request_ids = [task["request_id"] for task in body["tasks"]]
    assert len(request_ids) == 3
    assert len(set(request_ids)) == 3

    repo = Repository(settings.database_path)
    rows = [row for row in repo.list_usage() if row["purpose"] == "parallel_experiment"]
    assert len(rows) == 3
    assert {row["request_id"] for row in rows} == set(request_ids)
    assert all(row["status"] == "complete" for row in rows)

    # Repeating record_usage with the same request_id must not create a second row.
    first = rows[0]
    inserted = repo.record_usage(
        request_id=first["request_id"],
        conversation_id=None,
        message_id=None,
        provider="openai",
        requested_model="gpt-6-luna",
        model="gpt-6-luna",
        status="complete",
        input_tokens=1,
        output_tokens=1,
        cached_input_tokens=None,
        cache_write_tokens=None,
        reasoning_tokens=None,
        duration_ms=1,
        started_at=first["started_at"],
        finished_at=first["finished_at"],
        provider_response_id=None,
        purpose="parallel_experiment",
    )
    assert inserted is False
    rows_after = [row for row in repo.list_usage() if row["purpose"] == "parallel_experiment"]
    assert len(rows_after) == 3


def test_invalid_mode_rejected(client):
    response = client.post(
        "/api/experiments/parallel",
        json={"description": INCIDENT, "mode": "batch"},
    )
    assert response.status_code == 422


def test_blank_description_rejected(client):
    response = client.post(
        "/api/experiments/parallel",
        json={"description": "   ", "mode": "sequential"},
    )
    assert response.status_code == 422


def test_existing_chat_still_works(client, provider):
    created = client.post("/api/chats", json={}).json()
    stream = client.post(
        f"/api/chats/{created['id']}/messages",
        json={"content": "Reply with OK", "client_request_id": "exp-reg-1"},
    )
    assert stream.status_code == 200
    assert len(provider.calls) == 1


def test_incident_analysis_still_works(client, provider):
    response = client.post(
        "/api/incidents/analyze",
        json={"incident_id": "INC-55", "description": INCIDENT},
    )
    assert response.status_code == 200
    assert response.json()["analysis"]["incident_id"] == "INC-55"
    assert len(provider.parse_calls) == 1
