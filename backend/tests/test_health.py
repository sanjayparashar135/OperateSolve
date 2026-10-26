"""Health endpoint."""

from fastapi.testclient import TestClient

from app.main import create_app
from tests.conftest import make_settings
from tests.fakes import FakeProvider


def test_health_reports_model_without_requiring_a_key(client):
    response = client.get("/api/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["database"] == "ok"
    assert body["llm_configured"] is False
    assert body["model"] == "gpt-6-luna"
    assert body["provider"] == "openai"
    assert "api_key" not in response.text.lower() or "llm_configured" in response.text


def test_root_points_at_health(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["health"] == "/api/health"


def test_api_docs_can_be_disabled(tmp_path):
    settings = make_settings(tmp_path, expose_api_docs=False)
    app = create_app(settings, provider=FakeProvider())
    with TestClient(app) as client:
        assert client.get("/docs").status_code == 404
        assert client.get("/redoc").status_code == 404
        assert client.get("/openapi.json").status_code == 404
        assert client.get("/api/health").status_code == 200
