"""LLM token and estimated cost reporting."""

from datetime import datetime, timezone

from fastapi import APIRouter, Query, Request

from app.db.repository import Repository
from app.schemas.usage import UsageReport
from app.services.usage_service import aggregate_usage, window_start

router = APIRouter(prefix="/api", tags=["usage"])


@router.get("/usage", response_model=UsageReport)
def usage_report(
    request: Request,
    days: int | None = Query(default=None, ge=1, le=3650, description="Look-back window. Omit for all time."),
) -> dict:
    settings = request.app.state.settings
    now = datetime.now(timezone.utc)
    records = Repository(settings.database_path).list_usage(since=window_start(days, now))
    return aggregate_usage(
        records,
        request.app.state.pricing,
        days=days,
        now=now,
        model_label=request.app.state.registry.display_name,
    )
