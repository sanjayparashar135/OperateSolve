"""OperateSolve FastAPI application."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import chats, health
from app.config import Settings, get_settings
from app.db.database import init_db
from app.logging_config import configure_logging
from app.services.llm.base import LLMProvider
from app.services.llm.factory import build_provider

logger = logging.getLogger(__name__)


def create_app(settings: Settings | None = None, provider: LLMProvider | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, [settings.openai_api_key.strip()])
    init_db(settings.database_path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        logger.info(
            "startup model=%s provider=%s database=%s llm_configured=%s",
            app.state.settings.openai_model,
            app.state.settings.llm_provider,
            app.state.settings.database_path,
            app.state.settings.llm_configured,
        )
        yield

    app = FastAPI(title="OperateSolve API", version="1.0.0", lifespan=lifespan)
    app.state.settings = settings
    app.state.provider = provider or build_provider(settings)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origin_list,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(health.router)
    app.include_router(chats.router)

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
