"""Pricing config parsing and cost estimation."""

import json
from decimal import Decimal

import pytest

from app.config import Settings
from app.services.pricing import PricingConfigError, estimate_cost, load_pricing, parse_pricing

LUNA = {
    "input": 0.10,
    "cached_input": 0.01,
    "cache_write": 0.125,
    "output": 0.50,
    "long_context": {"above_input_tokens": 272000, "input": 0.20, "cached_input": 0.02, "cache_write": 0.25, "output": 0.75},
}


def table(**models):
    return parse_pricing(json.loads(json.dumps({"models": models}), parse_float=Decimal))


def test_plain_input_and_output():
    pricing = table(**{"gpt-6-luna": LUNA}).find("gpt-6-luna")
    cost = estimate_cost(pricing, input_tokens=200_000, output_tokens=100_000)
    assert cost.total == Decimal("0.07")
    assert cost.long_context is False


def test_cached_reads_and_cache_writes_follow_openai_formula():
    # OpenAI's documented example: 12,000 input = 1,000 ordinary + 4,000 cached + 7,000 written.
    pricing = table(**{"gpt-6-luna": LUNA}).find("gpt-6-luna")
    cost = estimate_cost(
        pricing,
        input_tokens=12_000,
        output_tokens=2_000,
        cached_input_tokens=4_000,
        cache_write_tokens=7_000,
    )
    assert cost.uncached_input == Decimal("1000") * Decimal("0.10") / 1_000_000
    assert cost.cached_input == Decimal("4000") * Decimal("0.01") / 1_000_000
    assert cost.cache_write == Decimal("7000") * Decimal("0.125") / 1_000_000
    assert cost.output == Decimal("2000") * Decimal("0.50") / 1_000_000
    assert cost.total == Decimal("0.002015")


def test_unknown_cached_counts_are_priced_as_ordinary_input():
    pricing = table(**{"gpt-6-luna": LUNA}).find("gpt-6-luna")
    unknown = estimate_cost(pricing, input_tokens=1000, output_tokens=0, cached_input_tokens=None)
    zero = estimate_cost(pricing, input_tokens=1000, output_tokens=0, cached_input_tokens=0)
    assert unknown.total == zero.total == Decimal("0.0001")


def test_inconsistent_cached_counts_are_clamped_to_input():
    pricing = table(**{"gpt-6-luna": LUNA}).find("gpt-6-luna")
    cost = estimate_cost(pricing, input_tokens=100, output_tokens=0, cached_input_tokens=500, cache_write_tokens=50)
    assert cost.uncached_input == 0
    assert cost.cache_write == 0
    assert cost.cached_input == Decimal("100") * Decimal("0.01") / 1_000_000


def test_missing_cache_rates_fall_back_to_input_rate():
    pricing = table(legacy={"input": 2.0, "output": 8.0}).find("legacy")
    cost = estimate_cost(pricing, input_tokens=1000, output_tokens=0, cached_input_tokens=400, cache_write_tokens=100)
    assert cost.total == Decimal("1000") * Decimal("2.0") / 1_000_000


def test_long_context_rates_apply_to_the_whole_request():
    pricing = table(**{"gpt-6-luna": LUNA}).find("gpt-6-luna")
    at_threshold = estimate_cost(pricing, input_tokens=272_000, output_tokens=1000)
    above = estimate_cost(pricing, input_tokens=272_001, output_tokens=1000)
    assert at_threshold.long_context is False
    assert above.long_context is True
    assert above.output == Decimal("1000") * Decimal("0.75") / 1_000_000
    assert above.uncached_input == Decimal("272001") * Decimal("0.20") / 1_000_000


def test_snapshot_ids_resolve_to_the_base_model_only():
    pricing = table(**{"gpt-6-luna": LUNA})
    assert pricing.find("GPT-6-Luna-2026-05-18") is not None
    assert pricing.find("gpt-6-luna-pro") is None
    assert pricing.find(None) is None


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"models": {"m": {"output": 1}}},
        {"models": {"m": {"input": -1, "output": 1}}},
        {"models": {"m": {"input": "abc", "output": 1}}},
        {"models": {"m": {"input": 1, "output": 1, "long_context": {"input": 2, "output": 2}}}},
        {"currency": "EUR", "models": {}},
        {"unit": "per_1k_tokens", "models": {}},
    ],
)
def test_invalid_config_is_rejected(raw):
    with pytest.raises(PricingConfigError):
        parse_pricing(raw)


def test_missing_file_is_a_config_error(tmp_path):
    with pytest.raises(PricingConfigError):
        load_pricing(tmp_path / "missing.json")


def test_bundled_pricing_covers_the_default_model():
    settings = Settings(_env_file=None)
    pricing = load_pricing(settings.pricing_file)
    luna = pricing.find(settings.openai_model)
    assert luna is not None
    assert pricing.verified_at
    assert pricing.source_url and pricing.source_url.startswith("https://developers.openai.com/")
    assert luna.standard.input == Decimal("0.10")
    assert luna.standard.cached_input == Decimal("0.01")
    assert luna.standard.cache_write == Decimal("0.125")
    assert luna.standard.output == Decimal("0.50")


def test_bundled_pricing_covers_every_registered_model():
    from app.services.llm.registry import MODEL_SPECS

    pricing = load_pricing(Settings(_env_file=None).pricing_file)
    for spec in MODEL_SPECS:
        assert pricing.find(spec.model_id) is not None, spec.model_id
    mini = pricing.find("gpt-5-mini-2025-08-07")
    nano = pricing.find("gpt-5-nano")
    assert (mini.standard.input, mini.standard.cached_input, mini.standard.output) == (
        Decimal("0.25"),
        Decimal("0.025"),
        Decimal("2.00"),
    )
    assert (nano.standard.input, nano.standard.cached_input, nano.standard.output) == (
        Decimal("0.05"),
        Decimal("0.005"),
        Decimal("0.40"),
    )
    assert mini.standard.cache_write is None and mini.long_context is None
