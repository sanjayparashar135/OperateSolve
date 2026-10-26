"""Aggregate recorded LLM usage into dashboard figures.

Costs are recomputed from stored token counts with the current pricing
config on every read, so a pricing fix applies to past requests too.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from typing import Any

from app.services.pricing import CostEstimate, ModelPricing, PricingTable, estimate_cost

MAX_DAILY_POINTS = 90
RECENT_LIMIT = 25
_COST_STEP = Decimal("0.00000001")


def window_start(days: int | None, now: datetime) -> str | None:
    if days is None:
        return None
    return (now - timedelta(days=days)).isoformat()


def price_record(record: dict[str, Any], pricing: PricingTable) -> CostEstimate | None:
    """Estimated USD for one request, or None when usage is missing or the model is unpriced."""

    if not record["usage_reported"]:
        return None
    model_pricing = pricing.find(record["model"])
    if model_pricing is None:
        return None
    return estimate_cost(
        model_pricing,
        input_tokens=record["input_tokens"],
        output_tokens=record["output_tokens"],
        cached_input_tokens=record["cached_input_tokens"],
        cache_write_tokens=record["cache_write_tokens"],
    )


def aggregate_usage(
    records: list[dict[str, Any]],
    pricing: PricingTable,
    *,
    days: int | None,
    now: datetime | None = None,
    model_label: Callable[[str | None], str | None] | None = None,
) -> dict[str, Any]:
    """Build the dashboard payload. records must be newest first.

    model_label maps a model id to a display name such as "Luna"; unknown ids get None.
    """

    now = now or datetime.now(timezone.utc)
    label = model_label or (lambda _model: None)
    totals = _Bucket()
    by_model: dict[str, _Bucket] = {}
    by_route: dict[tuple[str, str, str, str], _Bucket] = {}
    by_conversation: dict[str, _Bucket] = {}
    conversation_titles: dict[str, str | None] = {}
    daily: dict[str, _Bucket] = {}
    recent: list[dict[str, Any]] = []

    for record in records:
        cost = price_record(record, pricing)
        priced = cost is not None
        model_key = record["model"] or "unknown"
        totals.add(record, cost, priced)
        by_model.setdefault(model_key, _Bucket()).add(record, cost, priced)
        route_key = (
            record.get("routing_mode") or "legacy",
            record.get("task_mode") or "",
            record.get("requested_model") or model_key,
            model_key,
        )
        by_route.setdefault(route_key, _Bucket()).add(record, cost, priced)
        daily.setdefault(record["started_at"][:10], _Bucket()).add(record, cost, priced)
        conversation_id = record["conversation_id"] or ""
        by_conversation.setdefault(conversation_id, _Bucket()).add(record, cost, priced)
        conversation_titles.setdefault(conversation_id, record["conversation_title"])
        if len(recent) < RECENT_LIMIT:
            recent.append(_recent(record, cost, priced))

    return {
        "generated_at": now.isoformat(),
        "days": days,
        "window_start": window_start(days, now),
        "pricing": _pricing_summary(pricing),
        "totals": totals.summary(),
        "by_model": sorted(
            (
                {
                    "model": model,
                    "display_name": label(model),
                    **bucket.summary(),
                    "rates": _rates(pricing.find(model)),
                }
                for model, bucket in by_model.items()
            ),
            key=lambda item: (-(item["estimated_cost_usd"] or 0), -item["requests"]),
        ),
        "by_route": sorted(
            (
                {
                    "routing_mode": routing_mode,
                    "task_mode": task_mode or None,
                    "requested_model": requested,
                    "model": model,
                    "display_name": label(model),
                    **bucket.summary(),
                }
                for (routing_mode, task_mode, requested, model), bucket in by_route.items()
            ),
            key=lambda item: (-item["requests"], item["routing_mode"], item["task_mode"] or ""),
        ),
        "by_conversation": sorted(
            (
                {
                    "conversation_id": conversation_id or None,
                    "title": conversation_titles.get(conversation_id),
                    "deleted": conversation_titles.get(conversation_id) is None,
                    **bucket.summary(),
                }
                for conversation_id, bucket in by_conversation.items()
            ),
            key=lambda item: (-(item["estimated_cost_usd"] or 0), -item["requests"]),
        ),
        "daily": _daily_series(daily, days, now),
        "recent": recent,
    }


class _Bucket:
    def __init__(self) -> None:
        self.requests = 0
        self.with_usage = 0
        self.missing_usage = 0
        self.unpriced = 0
        self.cached_unknown = 0
        self.backfilled = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.cached_input_tokens = 0
        self.cache_write_tokens = 0
        self.reasoning_tokens = 0
        # Input tokens of requests that reported a cached count: the hit-rate denominator.
        self.cache_reported_input_tokens = 0
        self.cost = Decimal(0)
        self.input_cost = Decimal(0)
        self.uncached_baseline_input = Decimal(0)
        self.cache_read_savings = Decimal(0)
        self.cache_write_overhead = Decimal(0)
        self.durations: list[int] = []
        self.first_tokens: list[int] = []
        self.first: str | None = None
        self.last: str | None = None

    def add(self, record: dict[str, Any], cost: CostEstimate | None, priced: bool) -> None:
        self.requests += 1
        started = record["started_at"]
        self.first = started if self.first is None or started < self.first else self.first
        self.last = started if self.last is None or started > self.last else self.last
        if record["source"] == "backfill":
            self.backfilled += 1
        if record["status"] == "complete" and record["duration_ms"] is not None:
            self.durations.append(record["duration_ms"])
        if record["status"] == "complete" and record.get("first_token_ms") is not None:
            self.first_tokens.append(record["first_token_ms"])
        if not record["usage_reported"]:
            self.missing_usage += 1
            return
        self.with_usage += 1
        self.input_tokens += record["input_tokens"]
        self.output_tokens += record["output_tokens"]
        if record["cached_input_tokens"] is None:
            self.cached_unknown += 1
        else:
            self.cached_input_tokens += record["cached_input_tokens"]
            self.cache_reported_input_tokens += record["input_tokens"]
        self.cache_write_tokens += record["cache_write_tokens"] or 0
        self.reasoning_tokens += record["reasoning_tokens"] or 0
        if priced and cost is not None:
            self.cost += cost.total
            self.input_cost += cost.input
            self.uncached_baseline_input += cost.uncached_baseline_input
            self.cache_read_savings += cost.cache_read_savings
            self.cache_write_overhead += cost.cache_write_overhead
        else:
            self.unpriced += 1

    def summary(self) -> dict[str, Any]:
        priced_requests = self.with_usage - self.unpriced

        def usd(value: Decimal) -> float | None:
            return _usd(value) if priced_requests > 0 else None

        return {
            "requests": self.requests,
            "requests_with_usage": self.with_usage,
            "requests_missing_usage": self.missing_usage,
            "requests_unpriced": self.unpriced,
            "requests_cached_unknown": self.cached_unknown,
            "requests_backfilled": self.backfilled,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "cache_write_tokens": self.cache_write_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "cache_hit_rate": (
                round(self.cached_input_tokens / self.cache_reported_input_tokens, 4)
                if self.cache_reported_input_tokens > 0
                else None
            ),
            "estimated_cost_usd": usd(self.cost),
            "average_cost_usd": _usd(self.cost / priced_requests) if priced_requests > 0 else None,
            "estimated_input_cost_usd": usd(self.input_cost),
            "estimated_uncached_input_cost_usd": usd(self.uncached_baseline_input),
            "estimated_cache_read_savings_usd": usd(self.cache_read_savings),
            "estimated_cache_write_overhead_usd": usd(self.cache_write_overhead),
            "estimated_cache_savings_usd": usd(self.cache_read_savings - self.cache_write_overhead),
            "average_latency_ms": _mean(self.durations),
            "average_first_token_ms": _mean(self.first_tokens),
            "first_request_at": self.first,
            "last_request_at": self.last,
        }


def _recent(record: dict[str, Any], cost: CostEstimate | None, priced: bool) -> dict[str, Any]:
    return {
        "request_id": record["request_id"],
        "started_at": record["started_at"],
        "conversation_id": record["conversation_id"],
        "conversation_title": record["conversation_title"],
        "model": record["model"],
        "requested_model": record.get("requested_model"),
        "selection": record.get("selection"),
        "task_mode": record.get("task_mode"),
        "routing_mode": record.get("routing_mode"),
        "routing_reason": record.get("routing_reason"),
        "first_token_ms": record.get("first_token_ms"),
        "purpose": record.get("purpose") or "chat",
        "status": record["status"],
        "source": record["source"],
        "usage_reported": record["usage_reported"],
        "priced": bool(priced),
        "input_tokens": record["input_tokens"],
        "cached_input_tokens": record["cached_input_tokens"],
        "cache_write_tokens": record["cache_write_tokens"],
        "output_tokens": record["output_tokens"],
        "reasoning_tokens": record["reasoning_tokens"],
        "duration_ms": record["duration_ms"],
        "estimated_cost_usd": _usd(cost.total) if cost is not None else None,
        "estimated_cache_savings_usd": _usd(cost.cache_savings) if cost is not None else None,
    }


def _daily_series(daily: dict[str, _Bucket], days: int | None, now: datetime) -> list[dict[str, Any]]:
    end = now.date()
    if days is not None:
        start = end - timedelta(days=max(days, 1) - 1)
    elif daily:
        start = date.fromisoformat(min(daily))
    else:
        start = end
    start = max(start, end - timedelta(days=MAX_DAILY_POINTS - 1))
    series = []
    current = start
    while current <= end:
        bucket = daily.get(current.isoformat())
        summary = bucket.summary() if bucket else _Bucket().summary()
        series.append(
            {
                "date": current.isoformat(),
                "requests": summary["requests"],
                "input_tokens": summary["input_tokens"],
                "output_tokens": summary["output_tokens"],
                "cached_input_tokens": summary["cached_input_tokens"],
                "estimated_cost_usd": summary["estimated_cost_usd"],
            }
        )
        current += timedelta(days=1)
    return series


def _pricing_summary(pricing: PricingTable) -> dict[str, Any]:
    return {
        "currency": pricing.currency,
        "tier": pricing.tier,
        "source_url": pricing.source_url,
        "verified_at": pricing.verified_at,
        "notes": pricing.notes,
        "error": pricing.error,
        "models": sorted(pricing.models),
    }


def _rates(model_pricing: ModelPricing | None) -> dict[str, float | None] | None:
    if model_pricing is None:
        return None
    card = model_pricing.standard
    return {
        "input": float(card.input),
        "cached_input": float(card.cached_input) if card.cached_input is not None else None,
        "cache_write": float(card.cache_write) if card.cache_write is not None else None,
        "output": float(card.output),
        "long_context_above_input_tokens": model_pricing.long_context_above,
    }


def _mean(values: list[int]) -> int | None:
    return round(sum(values) / len(values)) if values else None


def _usd(value: Decimal) -> float:
    return float(value.quantize(_COST_STEP))
