"""LLM experiment routes (Phase 2.5B parallel execution)."""

from fastapi import APIRouter, Depends, HTTPException, Request

from app.db.repository import Repository
from app.schemas.parallel_experiment import ParallelExperimentRequest, ParallelExperimentResponse
from app.services.parallel_experiment_service import ParallelExperimentError, ParallelExperimentService

router = APIRouter(prefix="/api", tags=["experiments"])


def get_parallel_experiment_service(request: Request) -> ParallelExperimentService:
    settings = request.app.state.settings
    return ParallelExperimentService(
        repository=Repository(settings.database_path),
        provider=request.app.state.provider,
        settings=settings,
        registry=request.app.state.registry,
        pricing=request.app.state.pricing,
    )


@router.post("/experiments/parallel", response_model=ParallelExperimentResponse)
async def run_parallel_experiment(
    payload: ParallelExperimentRequest,
    service: ParallelExperimentService = Depends(get_parallel_experiment_service),
) -> dict:
    try:
        return await service.run(payload.description, payload.mode)
    except ParallelExperimentError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"message": exc.message, "code": exc.code},
        ) from exc
