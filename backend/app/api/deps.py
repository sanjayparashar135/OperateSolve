"""Shared API dependencies."""

from fastapi import Request

from app.db.repository import Repository
from app.extensions import resolve_operator
from app.services.chat_service import ChatService


def get_service(request: Request) -> ChatService:
    settings = request.app.state.settings
    return ChatService(
        repository=Repository(settings.database_path),
        provider=request.app.state.provider,
        settings=settings,
    )


def get_operator(request: Request) -> str:
    return resolve_operator(request.headers.get("authorization"))
