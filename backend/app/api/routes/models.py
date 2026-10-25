"""Model catalog: what the selector may offer, and where Auto would route right now."""

from fastapi import APIRouter, Request

from app.schemas.chat import ModelCatalog
from app.services.llm.registry import ModelRegistry
from app.services.model_router import (
    AUTO,
    AUTO_RULES,
    DEFAULT_TASK_MODE,
    ModelRouter,
    ModelUnavailable,
)
from app.services.pricing import PricingTable

router = APIRouter(prefix="/api", tags=["models"])

TASK_LABELS = {"simple": "Simple", "general": "General", "detailed": "Detailed"}


@router.get("/models", response_model=ModelCatalog)
def list_models(request: Request) -> dict:
    registry: ModelRegistry = request.app.state.registry
    model_router: ModelRouter = request.app.state.router
    pricing: PricingTable = request.app.state.pricing

    models = []
    for status in registry.statuses():
        spec = status.spec
        rates = pricing.find(spec.model_id)
        models.append(
            {
                "key": spec.key,
                "model_id": spec.model_id,
                "provider": spec.provider,
                "display_name": spec.display_name,
                "description": spec.description,
                "available": status.available,
                "unavailable_reason": status.reason,
                "baseline": spec.key == registry.baseline.key,
                "deprecated": spec.deprecated,
                "shutdown_date": spec.shutdown_date.isoformat() if spec.shutdown_date else None,
                "replacement": spec.replacement,
                "capabilities": list(spec.capabilities),
                "reasoning_efforts": list(spec.reasoning_efforts),
                "reasoning_effort": registry.reasoning_effort(spec, model_router.reasoning_effort),
                "context_window": spec.context_window,
                "max_output_tokens": spec.max_output_tokens,
                "priced": rates is not None,
                "rates": None
                if rates is None
                else {
                    "input": float(rates.standard.input),
                    "cached_input": _float(rates.standard.cached_input),
                    "cache_write": _float(rates.standard.cache_write),
                    "output": float(rates.standard.output),
                },
                "docs_url": spec.docs_url,
            }
        )

    rules = []
    for mode, target_key in AUTO_RULES.items():
        target = registry.get(target_key)
        try:
            decision = model_router.route(AUTO, mode)
            resolved, routing_mode, reason = decision.model, decision.routing_mode, decision.reason
        except ModelUnavailable as exc:
            resolved, routing_mode, reason = None, None, str(exc)
        rules.append(
            {
                "task_mode": mode,
                "label": TASK_LABELS[mode],
                "target": target.model_id if target else target_key,
                "resolved_model": resolved,
                "routing_mode": routing_mode,
                "reason": reason,
            }
        )

    return {
        "default_selection": AUTO,
        "default_task_mode": DEFAULT_TASK_MODE,
        "baseline_model": registry.baseline.model_id,
        "access_checked_at": registry.access_checked_at,
        "models": models,
        "auto_rules": rules,
    }


def _float(value) -> float | None:
    return float(value) if value is not None else None
