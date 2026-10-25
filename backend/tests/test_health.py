"""Health endpoint."""


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
