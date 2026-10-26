"""Structured incident analysis routes (Phase 2.5A)."""

from fastapi import APIRouter, Depends, HTTPException, Request

from app.db.repository import Repository
from app.schemas.incident import IncidentAnalyzeRequest, IncidentAnalyzeResponse
from app.services.incident_service import IncidentAnalysisError, IncidentService

router = APIRouter(prefix="/api", tags=["incidents"])


def get_incident_service(request: Request) -> IncidentService:
    settings = request.app.state.settings
    return IncidentService(
        repository=Repository(settings.database_path),
        provider=request.app.state.provider,
        settings=settings,
        registry=request.app.state.registry,
        pricing=request.app.state.pricing,
    )


@router.post("/incidents/analyze", response_model=IncidentAnalyzeResponse)
async def analyze_incident(
    payload: IncidentAnalyzeRequest,
    service: IncidentService = Depends(get_incident_service),
) -> dict:
    try:
        return await service.analyze(payload.incident_id, payload.description)
    except IncidentAnalysisError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"message": exc.message, "code": exc.code},
        ) from exc
