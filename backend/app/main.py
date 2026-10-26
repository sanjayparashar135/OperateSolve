"""OperateSolve FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import chats, experiments, health, incidents, knowledge, models, usage
from app.config import Settings, get_settings
from app.db.database import init_db
from app.logging_config import configure_logging
from app.services.llm.base import LLMProvider
from app.services.llm.factory import build_provider
from app.services.llm.registry import MODEL_SPECS, ModelRegistry, check_access
from app.services.model_router import ModelRouter
from app.services.pricing import PricingConfigError, PricingTable, load_pricing
from app.services.rag.embeddings import EmbeddingProvider

logger = logging.getLogger(__name__)


def build_pricing(settings: Settings) -> PricingTable:
    """Chat keeps working with a broken pricing file; the dashboard shows the error instead of costs."""

    try:
        return load_pricing(settings.pricing_file)
    except PricingConfigError as exc:
        logger.error("pricing_config_invalid detail=%s", exc)
        return PricingTable(error=str(exc))


def build_registry(settings: Settings) -> ModelRegistry:
    return ModelRegistry(
        MODEL_SPECS,
        provider=settings.llm_provider,
        baseline_model=settings.openai_model,
        disabled=settings.disabled_model_list,
    )


def create_app(
    settings: Settings | None = None,
    provider: LLMProvider | None = None,
    registry: ModelRegistry | None = None,
    embedder: EmbeddingProvider | None = None,
    vector_store=None,
    bm25_index=None,
    rag_reranker=None,
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, [settings.openai_api_key.strip()])
    init_db(settings.database_path)
    pricing = build_pricing(settings)
    registry = registry or build_registry(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state_settings: Settings = app.state.settings
        if state_settings.model_access_check and state_settings.llm_configured:
            await check_access(app.state.registry, app.state.provider)
        logger.info(
            "startup baseline_model=%s provider=%s database=%s llm_configured=%s pricing_models=%s "
            "rag_retrieval_mode=%s rag_reranker_enabled=%s",
            state_settings.openai_model,
            state_settings.llm_provider,
            state_settings.database_path,
            state_settings.llm_configured,
            len(app.state.pricing.models),
            state_settings.rag_retrieval_mode,
            state_settings.rag_reranker_enabled,
        )
        for status in app.state.registry.statuses():
            logger.info(
                "model_registered key=%s model=%s available=%s priced=%s deprecated=%s reason=%r",
                status.spec.key,
                status.spec.model_id,
                status.available,
                app.state.pricing.find(status.spec.model_id) is not None,
                status.spec.deprecated,
                status.reason or "",
            )
        yield

    docs = "/docs" if settings.expose_api_docs else None
    app = FastAPI(
        title="OperateSolve API",
        version="1.0.0",
        lifespan=lifespan,
        docs_url=docs,
        redoc_url="/redoc" if settings.expose_api_docs else None,
        openapi_url="/openapi.json" if settings.expose_api_docs else None,
    )
    app.state.settings = settings
    app.state.pricing = pricing
    app.state.provider = provider or build_provider(settings)
    app.state.registry = registry
    app.state.router = ModelRouter(registry, reasoning_effort=settings.openai_reasoning_effort)
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.bm25_index = bm25_index
    app.state.rag_reranker = rag_reranker
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(chats.router)
    app.include_router(models.router)
    app.include_router(usage.router)
    app.include_router(incidents.router)
    app.include_router(experiments.router)
    app.include_router(knowledge.router)

    @app.get("/")
    def root() -> dict[str, str]:
        return {"name": "OperateSolve", "health": "/api/health"}

    @app.exception_handler(Exception)
    async def unhandled(request: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        logger.exception("unhandled path=%s", request.url.path)
        return JSONResponse(status_code=500, content={"detail": "Internal server error"})

    return app


app = create_app()
