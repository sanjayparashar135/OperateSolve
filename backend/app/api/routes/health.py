"""Health endpoint."""

import logging

from fastapi import APIRouter, Request

from app.db.repository import Repository
from app.schemas.chat import HealthOut

router = APIRouter(prefix="/api", tags=["health"])
logger = logging.getLogger(__name__)


@router.get("/health", response_model=HealthOut)
def health(request: Request) -> HealthOut:
    settings = request.app.state.settings
    database = "ok"
    try:
        Repository(settings.database_path).ping()
    except Exception:
        database = "error"
        logger.exception("health_database_failed")
    return HealthOut(
        status="ok" if database == "ok" else "degraded",
        database=database,
        llm_configured=settings.llm_configured,
        model=settings.openai_model,
        provider=settings.llm_provider,
    )
