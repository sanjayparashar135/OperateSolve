"""Token pricing from backend configuration, and cost estimation.

Rates are USD per 1M tokens, read from pricing.json. Everything computed here
is an estimate: the OpenAI invoice is the source of truth.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

PER_MILLION = Decimal(1_000_000)
_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


class PricingConfigError(ValueError):
    pass


@dataclass(frozen=True)
class RateCard:
    input: Decimal
    output: Decimal
    # None means the provider does not discount reads or charge extra for writes,
    # so those tokens are billed at the ordinary input rate.
    cached_input: Decimal | None = None
    cache_write: Decimal | None = None


@dataclass(frozen=True)
class ModelPricing:
    model: str
    standard: RateCard
    long_context: RateCard | None = None
    long_context_above: int | None = None

    def rates_for(self, input_tokens: int) -> tuple[RateCard, bool]:
        if self.long_context is not None and self.long_context_above is not None and input_tokens > self.long_context_above:
            return self.long_context, True
        return self.standard, False


@dataclass(frozen=True)
class PricingTable:
    currency: str = "USD"
    tier: str = "standard"
    source_url: str | None = None
    verified_at: str | None = None
    notes: str | None = None
    models: dict[str, ModelPricing] = field(default_factory=dict)
    error: str | None = None

    def find(self, model: str | None) -> ModelPricing | None:
        """Exact model id first, then the base id of a dated snapshot (gpt-x-2026-01-01 -> gpt-x)."""

        if not model:
            return None
        key = model.strip().lower()
        return self.models.get(key) or self.models.get(_SNAPSHOT_SUFFIX.sub("", key))


@dataclass(frozen=True)
class CostEstimate:
    total: Decimal
    uncached_input: Decimal
    cached_input: Decimal
    cache_write: Decimal
    output: Decimal
    long_context: bool


def estimate_cost(
    pricing: ModelPricing,
    *,
    input_tokens: int,
    output_tokens: int,
    cached_input_tokens: int | None = None,
    cache_write_tokens: int | None = None,
) -> CostEstimate:
    """Price one request.

    Follows OpenAI's documented formula: cached reads and cache writes are
    subsets of input_tokens, so ordinary input = input - cached - writes.
    Missing cached/write counts are treated as zero (priced as ordinary input).
    """

    total_input = max(input_tokens, 0)
    cached = min(max(cached_input_tokens or 0, 0), total_input)
    writes = min(max(cache_write_tokens or 0, 0), total_input - cached)
    uncached = total_input - cached - writes
    output = max(output_tokens, 0)

    rates, long_context = pricing.rates_for(total_input)
    cached_rate = rates.input if rates.cached_input is None else rates.cached_input
    write_rate = rates.input if rates.cache_write is None else rates.cache_write

    uncached_cost = Decimal(uncached) * rates.input / PER_MILLION
    cached_cost = Decimal(cached) * cached_rate / PER_MILLION
    write_cost = Decimal(writes) * write_rate / PER_MILLION
    output_cost = Decimal(output) * rates.output / PER_MILLION
    return CostEstimate(
        total=uncached_cost + cached_cost + write_cost + output_cost,
        uncached_input=uncached_cost,
        cached_input=cached_cost,
        cache_write=write_cost,
        output=output_cost,
        long_context=long_context,
    )


def load_pricing(path: str | Path) -> PricingTable:
    file = Path(path)
    try:
        raw = json.loads(file.read_text(encoding="utf-8"), parse_float=Decimal)
    except FileNotFoundError as exc:
        raise PricingConfigError(f"Pricing file not found: {file}") from exc
    except (OSError, json.JSONDecodeError) as exc:
        raise PricingConfigError(f"Pricing file is not valid JSON: {file}") from exc
    return parse_pricing(raw)


def parse_pricing(raw: Any) -> PricingTable:
    if not isinstance(raw, dict) or not isinstance(raw.get("models"), dict):
        raise PricingConfigError("Pricing config needs a 'models' object.")
    currency = str(raw.get("currency") or "USD")
    if currency != "USD":
        raise PricingConfigError("Only USD pricing is supported.")
    unit = raw.get("unit", "per_1m_tokens")
    if unit != "per_1m_tokens":
        raise PricingConfigError("Pricing 'unit' must be 'per_1m_tokens'.")

    models: dict[str, ModelPricing] = {}
    for name, entry in raw["models"].items():
        if not isinstance(entry, dict):
            raise PricingConfigError(f"Pricing for '{name}' must be an object.")
        key = str(name).strip().lower()
        long_entry = entry.get("long_context")
        long_card: RateCard | None = None
        long_above: int | None = None
        if long_entry is not None:
            if not isinstance(long_entry, dict):
                raise PricingConfigError(f"long_context for '{name}' must be an object.")
            long_card = _rate_card(name, long_entry)
            long_above = _threshold(name, long_entry.get("above_input_tokens"))
        models[key] = ModelPricing(
            model=key,
            standard=_rate_card(name, entry),
            long_context=long_card,
            long_context_above=long_above,
        )
    return PricingTable(
        currency=currency,
        tier=str(raw.get("tier") or "standard"),
        source_url=_optional_str(raw.get("source_url")),
        verified_at=_optional_str(raw.get("verified_at")),
        notes=_optional_str(raw.get("notes")),
        models=models,
    )


def _rate_card(name: str, entry: dict[str, Any]) -> RateCard:
    return RateCard(
        input=_required_rate(name, "input", entry.get("input")),
        output=_required_rate(name, "output", entry.get("output")),
        cached_input=_rate(name, "cached_input", entry.get("cached_input")),
        cache_write=_rate(name, "cache_write", entry.get("cache_write")),
    )


def _required_rate(name: str, field_name: str, value: Any) -> Decimal:
    rate = _rate(name, field_name, value)
    if rate is None:
        raise PricingConfigError(f"Pricing for '{name}' is missing '{field_name}'.")
    return rate


def _rate(name: str, field_name: str, value: Any) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise PricingConfigError(f"Pricing '{field_name}' for '{name}' must be a number.")
    try:
        rate = Decimal(str(value))
    except InvalidOperation as exc:
        raise PricingConfigError(f"Pricing '{field_name}' for '{name}' must be a number.") from exc
    if not rate.is_finite() or rate < 0:
        raise PricingConfigError(f"Pricing '{field_name}' for '{name}' must be zero or more.")
    return rate


def _threshold(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, Decimal)) or int(value) <= 0:
        raise PricingConfigError(f"long_context for '{name}' needs a positive 'above_input_tokens'.")
    return int(value)


def _optional_str(value: Any) -> str | None:
    return str(value) if isinstance(value, str) and value.strip() else None
