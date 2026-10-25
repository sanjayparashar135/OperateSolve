"""Model registry, deterministic routing, per-conversation selection, and per-attempt billing."""

from datetime import date

import pytest
from fastapi.testclient import TestClient

from app.db.repository import Repository
from app.main import build_registry, create_app
from app.services.llm.registry import MODEL_SPECS, ModelRegistry
from app.services.model_router import InvalidModelSelection, ModelRouter, ModelUnavailable
from tests.conftest import make_settings
from tests.fakes import FakeProvider
from tests.test_streaming import parse_sse


def router_for(disabled=(), today=date(2026, 10, 25), effort="low", provider="openai") -> ModelRouter:
    registry = ModelRegistry(
        MODEL_SPECS,
        provider=provider,
        baseline_model="gpt-6-luna",
        disabled=disabled,
        today=lambda: today,
    )
    return ModelRouter(registry, reasoning_effort=effort)


def send(client, chat_id: str, request_id: str, content: str = "Summarize the incident", **preference):
    return client.post(
        f"/api/chats/{chat_id}/messages",
        json={"content": content, "client_request_id": request_id, **preference},
    )


def usage_rows(app) -> list[dict]:
    return Repository(app.state.settings.database_path).list_usage()


@pytest.fixture
def make_client(tmp_path):
    clients = []

    def build(provider: FakeProvider | None = None, **overrides):
        provider = provider or FakeProvider()
        app = create_app(make_settings(tmp_path, **overrides), provider=provider)
        client = TestClient(app)
        client.__enter__()
        clients.append(client)
        return client, app, provider

    yield build
    for client in clients:
        client.__exit__(None, None, None)


# Router rules


@pytest.mark.parametrize(
    ("mode", "model"),
    [("simple", "gpt-5-nano"), ("general", "gpt-6-luna"), ("detailed", "gpt-5-mini")],
)
def test_auto_routes_by_task_mode(mode, model):
    decision = router_for().route("auto", mode)
    assert decision.routing_mode == "auto"
    assert decision.model == model
    assert decision.requested_model == model
    assert decision.task_mode == mode
    assert mode.capitalize() in decision.reason


def test_auto_defaults_to_general():
    decision = router_for().route(None, None)
    assert (decision.selection, decision.task_mode, decision.model) == ("auto", "general", "gpt-6-luna")


@pytest.mark.parametrize("mode", ["simple", "general", "detailed"])
def test_manual_selection_overrides_task_mode(mode):
    decision = router_for().route("mini", mode)
    assert decision.routing_mode == "manual"
    assert decision.model == "gpt-5-mini"
    assert decision.task_mode is None
    assert decision.reason == "Manual selection: Mini (gpt-5-mini)."


def test_unavailable_auto_target_falls_back_to_luna_with_reason():
    decision = router_for(disabled=["nano"]).route("auto", "simple")
    assert decision.routing_mode == "fallback"
    assert decision.requested_model == "gpt-5-nano"
    assert decision.model == "gpt-6-luna"
    assert "Nano (gpt-5-nano) is unavailable" in decision.reason
    assert "DISABLED_MODELS" in decision.reason
    assert "fell back to Luna (gpt-6-luna)" in decision.reason


def test_unavailable_manual_model_is_rejected_not_substituted():
    with pytest.raises(ModelUnavailable, match="Mini \\(gpt-5-mini\\) is unavailable"):
        router_for(disabled=["gpt-5-mini"]).route("mini", "general")


def test_no_fallback_when_baseline_is_also_unavailable():
    with pytest.raises(ModelUnavailable, match="fallback"):
        router_for(disabled=["nano", "luna"]).route("auto", "simple")


def test_deprecated_models_retire_on_shutdown_date():
    router = router_for(today=date(2026, 12, 11))
    with pytest.raises(ModelUnavailable, match="Retired by the provider on 2026-12-11"):
        router.route("nano", None)
    assert router.route("auto", "detailed").routing_mode == "fallback"
    assert router_for(today=date(2026, 12, 10)).route("nano", None).model == "gpt-5-nano"


def test_models_for_another_provider_are_unavailable():
    with pytest.raises(ModelUnavailable, match="openai provider"):
        router_for(provider="bedrock").route("luna", None)


def test_failed_access_check_marks_model_unavailable():
    router = router_for()
    router.registry.record_access("mini", False)
    with pytest.raises(ModelUnavailable, match="cannot access"):
        router.route("mini", None)
    router.registry.record_access("mini", None)
    assert router.route("mini", None).model == "gpt-5-mini"


@pytest.mark.parametrize(
    ("selection", "mode"),
    [("gpt-6-astra", None), ("gpt-5-mini", None), ("auto", "urgent"), ("", "general")],
)
def test_invalid_selection_is_rejected(selection, mode):
    with pytest.raises(InvalidModelSelection):
        router_for().route(selection or "unknown", mode)


def test_reasoning_effort_is_adjusted_per_model():
    router = router_for(effort="none")
    assert router.route("luna", None).reasoning_effort == "none"
    assert router.route("nano", None).reasoning_effort == "minimal"
    assert router_for(effort="low").route("mini", None).reasoning_effort == "low"
    assert router_for(effort="").route("luna", None).reasoning_effort is None


def test_baseline_must_be_registered(tmp_path):
    with pytest.raises(ValueError, match="not a registered model"):
        build_registry(make_settings(tmp_path, openai_model="gpt-6-astra"))


# API: catalog, selection, persistence


def test_models_catalog_reports_availability_pricing_and_rules(make_client):
    client, _, _ = make_client(disabled_models="mini")
    catalog = client.get("/api/models").json()
    models = {item["key"]: item for item in catalog["models"]}
    assert list(models) == ["luna", "mini", "nano"]
    assert models["luna"]["baseline"] is True
    assert models["luna"]["available"] is True
    assert models["luna"]["rates"]["input"] == pytest.approx(0.10)
    assert models["mini"]["available"] is False
    assert "DISABLED_MODELS" in models["mini"]["unavailable_reason"]
    assert models["nano"]["deprecated"] is True
    assert models["nano"]["shutdown_date"] == "2026-12-11"
    assert models["nano"]["rates"] == {"input": 0.05, "cached_input": 0.005, "cache_write": None, "output": 0.40}
    assert all(item["priced"] for item in models.values())
    rules = {rule["task_mode"]: rule for rule in catalog["auto_rules"]}
    assert rules["simple"]["resolved_model"] == "gpt-5-nano"
    assert rules["detailed"]["target"] == "gpt-5-mini"
    assert rules["detailed"]["resolved_model"] == "gpt-6-luna"
    assert rules["detailed"]["routing_mode"] == "fallback"


def test_startup_access_check_disables_inaccessible_models(make_client):
    provider = FakeProvider()

    async def check_model_access(model_id: str):
        return {"gpt-5-mini": False, "gpt-5-nano": None}.get(model_id, True)

    provider.check_model_access = check_model_access
    client, _, _ = make_client(provider, openai_api_key="sk-test-not-real")
    catalog = client.get("/api/models").json()
    models = {item["key"]: item for item in catalog["models"]}
    assert models["mini"]["available"] is False
    assert "cannot access" in models["mini"]["unavailable_reason"]
    assert models["nano"]["available"] is True
    assert catalog["access_checked_at"]


def test_new_conversation_defaults_to_auto_general(client, app, provider: FakeProvider):
    chat = client.post("/api/chats", json={}).json()
    assert (chat["model_selection"], chat["task_mode"]) == ("auto", "general")
    events = parse_sse(send(client, chat["id"], "client-req-default").text)
    assert events[0]["routing"]["model"] == "gpt-6-luna"
    assert events[0]["routing"]["routing_mode"] == "auto"
    assert provider.calls[0]["model"] == "gpt-6-luna"
    assert provider.calls[0]["reasoning_effort"] == "low"


def test_manual_selection_is_used_persisted_and_logged(client, app, provider: FakeProvider, caplog):
    caplog.set_level("INFO")
    chat = client.post("/api/chats", json={"model_selection": "mini", "task_mode": "simple"}).json()
    assert chat["model_selection"] == "mini"
    done = parse_sse(send(client, chat["id"], "client-req-manual").text)[-1]["message"]

    assert provider.calls[0]["model"] == "gpt-5-mini"
    assert done["model"] == "gpt-5-mini"
    assert done["requested_model"] == "gpt-5-mini"
    assert done["routing_mode"] == "manual"
    assert done["routing_reason"] == "Manual selection: Mini (gpt-5-mini)."

    detail = client.get(f"/api/chats/{chat['id']}").json()
    assert detail["conversation"]["model_selection"] == "mini"
    assert detail["messages"][1]["routing_mode"] == "manual"

    [row] = usage_rows(app)
    assert (row["selection"], row["routing_mode"], row["requested_model"], row["model"]) == (
        "mini",
        "manual",
        "gpt-5-mini",
        "gpt-5-mini",
    )
    assert row["first_token_ms"] is not None
    assert "routing_mode=manual requested_model=gpt-5-mini model=gpt-5-mini" in caplog.text


def test_message_preference_is_saved_on_the_conversation(client, provider: FakeProvider):
    chat_id = client.post("/api/chats", json={}).json()["id"]
    send(client, chat_id, "client-req-pref", model_selection="auto", task_mode="simple")
    assert provider.calls[-1]["model"] == "gpt-5-nano"
    conversation = client.get(f"/api/chats/{chat_id}").json()["conversation"]
    assert (conversation["model_selection"], conversation["task_mode"]) == ("auto", "simple")
    # The next turn reuses the saved preference without sending it again.
    send(client, chat_id, "client-req-pref-2")
    assert provider.calls[-1]["model"] == "gpt-5-nano"


def test_switching_models_in_an_existing_conversation(client, app, provider: FakeProvider):
    chat_id = client.post("/api/chats", json={}).json()["id"]
    send(client, chat_id, "client-req-switch-1", content="First question")

    patched = client.patch(f"/api/chats/{chat_id}/model", json={"model_selection": "nano"})
    assert patched.status_code == 200
    assert patched.json()["model_selection"] == "nano"
    assert patched.json()["task_mode"] == "general"
    send(client, chat_id, "client-req-switch-2", content="Second question")

    client.patch(f"/api/chats/{chat_id}/model", json={"model_selection": "auto", "task_mode": "detailed"})
    send(client, chat_id, "client-req-switch-3", content="Third question")

    assert [call["model"] for call in provider.calls] == ["gpt-6-luna", "gpt-5-nano", "gpt-5-mini"]
    # Full history goes to whichever model answers next.
    assert [turn.content for turn in provider.calls[2]["turns"] if turn.role == "user"] == [
        "First question",
        "Second question",
        "Third question",
    ]
    messages = client.get(f"/api/chats/{chat_id}").json()["messages"]
    assert [m["model"] for m in messages if m["role"] == "assistant"] == ["gpt-6-luna", "gpt-5-nano", "gpt-5-mini"]

    report = client.get("/api/usage").json()
    assert {row["model"]: row["requests"] for row in report["by_model"]} == {
        "gpt-6-luna": 1,
        "gpt-5-nano": 1,
        "gpt-5-mini": 1,
    }
    names = {row["model"]: row["display_name"] for row in report["by_model"]}
    assert names == {"gpt-6-luna": "Luna", "gpt-5-nano": "Nano", "gpt-5-mini": "Mini"}
    routes = {(row["routing_mode"], row["task_mode"], row["model"]) for row in report["by_route"]}
    assert routes == {
        ("auto", "general", "gpt-6-luna"),
        ("manual", None, "gpt-5-nano"),
        ("auto", "detailed", "gpt-5-mini"),
    }


def test_disabled_model_cannot_be_selected(make_client):
    client, app, provider = make_client(disabled_models="nano")
    chat_id = client.post("/api/chats", json={}).json()["id"]

    assert client.post("/api/chats", json={"model_selection": "nano"}).status_code == 409
    patched = client.patch(f"/api/chats/{chat_id}/model", json={"model_selection": "nano"})
    assert patched.status_code == 409
    assert "Disabled by DISABLED_MODELS" in patched.json()["detail"]

    rejected = send(client, chat_id, "client-req-disabled", model_selection="nano")
    assert rejected.status_code == 409
    assert provider.calls == []
    assert client.get(f"/api/chats/{chat_id}").json()["messages"] == []
    assert usage_rows(app) == []


def test_auto_fallback_is_explicit_and_recorded(make_client):
    client, app, provider = make_client(disabled_models="nano")
    chat_id = client.post("/api/chats", json={"task_mode": "simple"}).json()["id"]
    events = parse_sse(send(client, chat_id, "client-req-fallback").text)

    routing = events[0]["routing"]
    assert routing["routing_mode"] == "fallback"
    assert (routing["requested_model"], routing["model"]) == ("gpt-5-nano", "gpt-6-luna")
    assert provider.calls[0]["model"] == "gpt-6-luna"
    done = events[-1]["message"]
    assert done["routing_mode"] == "fallback"
    assert "fell back to Luna" in done["routing_reason"]

    [row] = usage_rows(app)
    assert (row["requested_model"], row["model"], row["routing_mode"]) == ("gpt-5-nano", "gpt-6-luna", "fallback")
    assert "unavailable" in row["routing_reason"]


def test_previously_saved_model_that_becomes_unavailable_fails_clearly(tmp_path):
    settings = make_settings(tmp_path)
    provider = FakeProvider()
    with TestClient(create_app(settings, provider=provider)) as client:
        chat_id = client.post("/api/chats", json={"model_selection": "mini"}).json()["id"]

    with TestClient(create_app(make_settings(tmp_path, disabled_models="mini"), provider=provider)) as client:
        response = send(client, chat_id, "client-req-stale")
        assert response.status_code == 409
        assert "Mini (gpt-5-mini) is unavailable" in response.json()["detail"]
        assert provider.calls == []


def test_invalid_selection_values_return_422(client):
    chat_id = client.post("/api/chats", json={}).json()["id"]
    assert client.post("/api/chats", json={"model_selection": "gpt-6-astra"}).status_code == 422
    assert client.patch(f"/api/chats/{chat_id}/model", json={"model_selection": "turbo"}).status_code == 422
    assert client.patch(f"/api/chats/{chat_id}/model", json={"model_selection": "auto", "task_mode": "x"}).status_code == 422
    assert send(client, chat_id, "client-req-invalid", model_selection="sol").status_code == 422
    assert client.patch(f"/api/chats/{chat_id}/model", json={"model_selection": "a b"}).status_code == 422


def test_replay_ignores_a_new_selection_and_does_not_bill_again(client, app, provider: FakeProvider):
    chat_id = client.post("/api/chats", json={}).json()["id"]
    send(client, chat_id, "client-req-replay-model")
    replay = parse_sse(send(client, chat_id, "client-req-replay-model", model_selection="mini").text)
    assert replay[0]["replay"] is True
    assert replay[-1]["message"]["model"] == "gpt-6-luna"
    assert len(provider.calls) == 1
    assert len(usage_rows(app)) == 1
    assert client.get(f"/api/chats/{chat_id}").json()["conversation"]["model_selection"] == "auto"


# Billing and exactly-once usage per provider attempt


def test_cost_uses_the_actual_models_rates(client, app, provider: FakeProvider):
    provider.input_tokens = 10_000
    provider.cached_input_tokens = 4_000
    provider.output_tokens = 1_000
    chat_id = client.post("/api/chats", json={"model_selection": "nano"}).json()["id"]
    send(client, chat_id, "client-req-nano-cost")
    client.patch(f"/api/chats/{chat_id}/model", json={"model_selection": "mini"})
    send(client, chat_id, "client-req-mini-cost")
    client.patch(f"/api/chats/{chat_id}/model", json={"model_selection": "luna"})
    send(client, chat_id, "client-req-luna-cost")

    report = client.get("/api/usage").json()
    costs = {row["model"]: row["estimated_cost_usd"] for row in report["by_model"]}
    # (6000 ordinary × input + 4000 cached × cached + 1000 × output) / 1M
    assert costs["gpt-5-nano"] == pytest.approx((6_000 * 0.05 + 4_000 * 0.005 + 1_000 * 0.40) / 1e6)
    assert costs["gpt-5-mini"] == pytest.approx((6_000 * 0.25 + 4_000 * 0.025 + 1_000 * 2.00) / 1e6)
    assert costs["gpt-6-luna"] == pytest.approx((6_000 * 0.10 + 4_000 * 0.01 + 1_000 * 0.50) / 1e6)
    assert report["totals"]["estimated_cost_usd"] == pytest.approx(sum(costs.values()))
    for row in report["by_model"]:
        assert row["average_cost_usd"] == pytest.approx(row["estimated_cost_usd"])
        assert row["average_latency_ms"] is not None
        assert row["average_first_token_ms"] is not None
    assert report["recent"][0]["routing_mode"] == "manual"


def test_api_error_is_not_retried_on_another_model(client, app, provider: FakeProvider):
    provider.fail_models = {"gpt-5-nano"}
    chat_id = client.post("/api/chats", json={"task_mode": "simple"}).json()["id"]
    events = parse_sse(send(client, chat_id, "client-req-nano-fails").text)

    assert events[-1]["type"] == "error"
    assert events[-1]["message"]["model"] == "gpt-5-nano"
    assert [call["model"] for call in provider.calls] == ["gpt-5-nano"]
    [row] = usage_rows(app)
    assert (row["model"], row["status"], row["routing_mode"]) == ("gpt-5-nano", "error", "auto")


def test_retry_after_switching_records_each_attempt_once(client, app, provider: FakeProvider):
    provider.fail_models = {"gpt-5-mini"}
    chat_id = client.post("/api/chats", json={"model_selection": "mini"}).json()["id"]
    failed = parse_sse(send(client, chat_id, "client-req-retry-switch").text)[-1]["message"]

    client.patch(f"/api/chats/{chat_id}/model", json={"model_selection": "luna"})
    retried = parse_sse(client.post(f"/api/chats/{chat_id}/messages/{failed['id']}/retry").text)
    assert retried[0]["routing"]["model"] == "gpt-6-luna"
    done = retried[-1]["message"]
    assert done["id"] == failed["id"]
    assert done["model"] == "gpt-6-luna"
    assert done["routing_mode"] == "manual"

    rows = sorted(usage_rows(app), key=lambda row: row["started_at"])
    assert [(row["model"], row["status"]) for row in rows] == [("gpt-5-mini", "error"), ("gpt-6-luna", "complete")]
    assert len({row["request_id"] for row in rows}) == 2
    assert all(row["message_id"] == failed["id"] for row in rows)
    assert [call["model"] for call in provider.calls] == ["gpt-5-mini", "gpt-6-luna"]


def test_retry_with_unavailable_selection_is_rejected_before_calling(make_client, tmp_path):
    client, app, provider = make_client()
    provider.fail = True
    chat_id = client.post("/api/chats", json={"model_selection": "nano"}).json()["id"]
    failed = parse_sse(send(client, chat_id, "client-req-retry-disabled").text)[-1]["message"]

    client2, _, provider2 = make_client(disabled_models="nano")
    response = client2.post(f"/api/chats/{chat_id}/messages/{failed['id']}/retry")
    assert response.status_code == 409
    assert provider2.calls == []
    message = Repository(app.state.settings.database_path).get_message(failed["id"])
    assert message["status"] == "error"
    assert len(usage_rows(app)) == 1


def test_unpriced_model_is_counted_but_not_costed(client, provider: FakeProvider):
    provider.model = "gpt-6-luna-experimental"
    chat_id = client.post("/api/chats", json={}).json()["id"]
    send(client, chat_id, "client-req-unpriced")
    report = client.get("/api/usage").json()
    [row] = report["by_model"]
    assert row["model"] == "gpt-6-luna-experimental"
    assert row["display_name"] is None
    assert row["estimated_cost_usd"] is None
    assert row["rates"] is None
    assert report["totals"]["requests_unpriced"] == 1
