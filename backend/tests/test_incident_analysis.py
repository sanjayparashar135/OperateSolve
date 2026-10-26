"""Phase 2.5A — schema-enforced structured incident analysis."""

from datetime import date

import pytest
from pydantic import ValidationError

from app.main import create_app
from app.schemas.incident import IncidentAnalysis
from app.services.llm.registry import MODEL_SPECS, ModelRegistry, ModelSpec
from tests.conftest import make_settings
from tests.fakes import FakeProvider


def _valid_payload(**overrides):
    data = {
        "incident_id": "INC-1001",
        "severity": "P2",
        "summary": "Checkout API latency spike affecting payments.",
        "probable_cause": "Saturated connection pool on the payments service.",
        "recommended_actions": ["Scale payments workers", "Drain stuck connections"],
        "requires_human_approval": True,
    }
    data.update(overrides)
    return data


def test_incident_analysis_schema_accepts_valid_payload():
    analysis = IncidentAnalysis.model_validate(_valid_payload())
    assert analysis.severity == "P2"
    assert analysis.requires_human_approval is True
    assert len(analysis.recommended_actions) == 2


@pytest.mark.parametrize(
    "overrides",
    [
        {"severity": "critical"},
        {"recommended_actions": []},
        {"summary": ""},
        {"requires_human_approval": None},
    ],
)
def test_incident_analysis_schema_rejects_invalid_payload(overrides):
    with pytest.raises(ValidationError):
        IncidentAnalysis.model_validate(_valid_payload(**overrides))


def test_analyze_incident_success(client, provider):
    provider.structured_result = _valid_payload(incident_id="INC-42")
    response = client.post(
        "/api/incidents/analyze",
        json={
            "incident_id": "INC-42",
            "description": "Payments p95 latency rose from 120ms to 4.5s after the 14:00 deploy.",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert body["analysis"]["incident_id"] == "INC-42"
    assert body["analysis"]["severity"] == "P2"
    assert body["analysis"]["recommended_actions"]
    assert body["model"] == "gpt-6-luna"
    assert body["input_tokens"] == 80
    assert body["cached_input_tokens"] == 10
    assert body["latency_ms"] >= 0
    assert len(provider.parse_calls) == 1
    assert provider.parse_calls[0]["text_format"] is IncidentAnalysis

    usage = client.get("/api/usage?days=7").json()
    recent = usage["recent"]
    assert any(row["purpose"] == "incident_analysis" and row["status"] == "complete" for row in recent)
    incident_rows = [row for row in recent if row["purpose"] == "incident_analysis"]
    assert incident_rows[0]["usage_reported"] is True
    assert incident_rows[0]["cached_input_tokens"] == 10


def test_analyze_incident_refusal(client, provider):
    provider.parse_mode = "refuse"
    response = client.post(
        "/api/incidents/analyze",
        json={"incident_id": "INC-9", "description": "Fictional sabotage request."},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "refusal"
    assert "cannot assist" in detail["message"].lower()

    usage = client.get("/api/usage").json()
    assert any(row["purpose"] == "incident_analysis" and row["status"] == "refused" for row in usage["recent"])


def test_analyze_incident_incomplete(client, provider):
    provider.parse_mode = "incomplete"
    response = client.post(
        "/api/incidents/analyze",
        json={"incident_id": "INC-8", "description": "Very long fictional incident narrative."},
    )
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "incomplete"
    assert "incomplete" in detail["message"].lower() or "cut off" in detail["message"].lower()

    usage = client.get("/api/usage").json()
    assert any(
        row["purpose"] == "incident_analysis" and row["status"] == "incomplete" for row in usage["recent"]
    )


def test_analyze_incident_api_failure(client, provider):
    provider.parse_mode = "error"
    response = client.post(
        "/api/incidents/analyze",
        json={"incident_id": "INC-7", "description": "Database primary unreachable."},
    )
    assert response.status_code == 502
    detail = response.json()["detail"]
    assert detail["code"] == "provider_error"

    usage = client.get("/api/usage").json()
    assert any(row["purpose"] == "incident_analysis" and row["status"] == "error" for row in usage["recent"])


def test_analyze_incident_unsupported_model(tmp_path, provider):
    settings = make_settings(tmp_path, openai_model="gpt-6-luna")
    specs = []
    for spec in MODEL_SPECS:
        if spec.key == "luna":
            specs.append(
                ModelSpec(
                    key=spec.key,
                    model_id=spec.model_id,
                    provider=spec.provider,
                    display_name=spec.display_name,
                    description=spec.description,
                    reasoning_efforts=spec.reasoning_efforts,
                    fallback_reasoning_effort=spec.fallback_reasoning_effort,
                    context_window=spec.context_window,
                    max_input_tokens=spec.max_input_tokens,
                    max_output_tokens=spec.max_output_tokens,
                    capabilities=("text", "responses_api", "streaming", "reasoning"),
                    deprecated=spec.deprecated,
                    shutdown_date=spec.shutdown_date,
                    replacement=spec.replacement,
                    docs_url=spec.docs_url,
                )
            )
        else:
            specs.append(spec)
    registry = ModelRegistry(specs, provider="openai", baseline_model="gpt-6-luna")
    app = create_app(settings, provider=provider, registry=registry)
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/api/incidents/analyze",
            json={"incident_id": "INC-6", "description": "Queue backlog growing."},
        )
    assert response.status_code == 409
    detail = response.json()["detail"]
    assert detail["code"] == "unsupported_model"
    assert "structured" in detail["message"].lower()
    assert provider.parse_calls == []


def test_analyze_incident_unavailable_model(tmp_path, provider):
    settings = make_settings(tmp_path, openai_model="gpt-6-luna", disabled_models="luna")
    registry = ModelRegistry(
        MODEL_SPECS,
        provider="openai",
        baseline_model="gpt-6-luna",
        disabled=["luna"],
        today=lambda: date(2026, 10, 26),
    )
    app = create_app(settings, provider=provider, registry=registry)
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/api/incidents/analyze",
            json={"incident_id": "INC-5", "description": "Cache cluster flap."},
        )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "model_unavailable"
    assert provider.parse_calls == []


def test_analyze_incident_provider_without_parse(tmp_path):
    class StreamOnlyProvider:
        async def stream(self, turns, **kwargs):
            if False:
                yield None

    settings = make_settings(tmp_path)
    app = create_app(settings, provider=StreamOnlyProvider())
    from fastapi.testclient import TestClient

    with TestClient(app) as client:
        response = client.post(
            "/api/incidents/analyze",
            json={"incident_id": "INC-4", "description": "Auth service 5xx burst."},
        )
    assert response.status_code == 409
    assert response.json()["detail"]["code"] == "unsupported_provider"


def test_analyze_incident_invalid_structured_payload(client, provider):
    provider.parse_mode = "invalid"
    response = client.post(
        "/api/incidents/analyze",
        json={"incident_id": "INC-3", "description": "Synthetic invalid model output."},
    )
    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "schema_invalid"


def test_analyze_incident_request_validation(client):
    response = client.post("/api/incidents/analyze", json={"incident_id": "  ", "description": "x"})
    assert response.status_code == 422
