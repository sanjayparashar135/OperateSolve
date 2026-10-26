"""Shared fixtures."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.db.repository import Repository
from app.main import build_registry, create_app
from app.services.chat_service import ChatService
from app.services.model_router import ModelRouter
from tests.fakes import FakeProvider


def make_settings(tmp_path: Path, **overrides) -> Settings:
    values = {
        "openai_api_key": "",
        "openai_model": "gpt-6-luna",
        "openai_reasoning_effort": "low",
        "openai_max_output_tokens": 256,
        "database_path": str(tmp_path / "test.db"),
        "cors_origins": "http://127.0.0.1:3000",
        "log_level": "INFO",
        "llm_provider": "openai",
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)


def make_service(settings: Settings, provider, repo: Repository) -> ChatService:
    registry = build_registry(settings)
    router = ModelRouter(registry, reasoning_effort=settings.openai_reasoning_effort)
    return ChatService(repo, provider, settings, router, registry=registry)


@pytest.fixture
def provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture
def app(settings: Settings, provider: FakeProvider):
    return create_app(settings, provider=provider)


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client
