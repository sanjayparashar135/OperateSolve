"""Prompt caching: model-aware request settings, prefix stability, and cache accounting."""

from datetime import datetime, timezone
from decimal import Decimal
from types import SimpleNamespace

import pytest

from app.services.chat_service import INSTRUCTIONS
from app.services.llm import prompt_cache
from app.services.llm.base import ChatTurn, LLMUsage
from app.services.llm.openai_provider import OpenAIResponsesProvider, extract_usage
from app.services.llm.registry import MODEL_SPECS
from app.services.pricing import estimate_cost, parse_pricing
from app.services.usage_service import aggregate_usage
from tests.conftest import make_settings
from tests.fakes import FakeProvider
from tests.test_openai_provider import AsyncEvents, event
from tests.test_streaming import parse_sse
from tests.test_usage import record

PRICING = parse_pricing(
    {
        "models": {
            "gpt-6-luna": {
                "input": 0.10,
                "cached_input": 0.01,
                "cache_write": 0.125,
                "output": 0.50,
                "long_context": {"above_input_tokens": 272000, "input": 0.20, "cached_input": 0.02, "cache_write": 0.25, "output": 0.75},
            },
            "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
        }
    }
)
M = Decimal(1_000_000)


# --- Model-aware request settings -------------------------------------------------


def test_luna_pins_implicit_30_minute_caching_without_a_cache_key():
    params = prompt_cache.request_params("gpt-6-luna")
    assert params == {"prompt_cache_options": {"mode": "implicit", "ttl": "30m"}}
    assert prompt_cache.request_params("gpt-6-luna-2026-05-18") == params
    assert prompt_cache.policy_for("gpt-6-luna").min_prefix_tokens == 1024


def test_nano_gets_only_a_stable_cache_key():
    # Earlier models reject prompt_cache_options, and gpt-5-nano does not support 24h retention.
    assert prompt_cache.request_params("gpt-5-nano") == {"prompt_cache_key": prompt_cache.PROMPT_CACHE_KEY}
    assert prompt_cache.request_params("gpt-5-nano-2025-08-07") == prompt_cache.request_params("gpt-5-nano")


def test_mini_and_unknown_models_get_no_cache_parameters():
    assert prompt_cache.request_params("gpt-5-mini") == {}
    assert prompt_cache.request_params("gpt-5-mini-2025-08-07") == {}
    assert prompt_cache.request_params("some-future-model") == {}
    assert prompt_cache.request_params(None) == {}


def test_every_registered_model_has_an_explicit_policy():
    for spec in MODEL_SPECS:
        assert spec.model_id in prompt_cache.POLICIES, spec.model_id


def test_no_model_is_sent_retention_or_12_hour_settings():
    for model in prompt_cache.POLICIES:
        params = prompt_cache.request_params(model)
        assert "prompt_cache_retention" not in params
        assert params.get("prompt_cache_options", {}).get("ttl") in {None, "30m"}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model", "expected"),
    [
        ("gpt-6-luna", {"prompt_cache_options": {"mode": "implicit", "ttl": "30m"}}),
        ("gpt-5-nano", {"prompt_cache_key": "operatesolve-chat-v1"}),
        ("gpt-5-mini", {}),
    ],
)
async def test_provider_sends_each_models_cache_parameters(tmp_path, model, expected):
    provider = OpenAIResponsesProvider(make_settings(tmp_path, openai_api_key="sk-test-secret-value"))

    async def create(**kwargs):
        create.kwargs = kwargs
        return AsyncEvents([event(type="response.output_text.delta", delta="Hi")])

    provider._client = SimpleNamespace(responses=SimpleNamespace(create=create))
    [item async for item in provider.stream([ChatTurn("user", "Hello")], model=model)]

    sent = {key: value for key, value in create.kwargs.items() if key.startswith("prompt_cache")}
    assert sent == expected


# --- Prefix stability -------------------------------------------------------------


def send(client, chat_id: str, request_id: str, content: str):
    return client.post(f"/api/chats/{chat_id}/messages", json={"content": content, "client_request_id": request_id})


def test_each_turn_extends_the_previous_prompt_exactly(client, provider: FakeProvider):
    chat_id = client.post("/api/chats", json={}).json()["id"]
    for index, content in enumerate(["First question", "Second question", "Third question"]):
        send(client, chat_id, f"client-req-prefix-{index}", content)

    first, second, third = provider.calls
    assert first["instructions"] == second["instructions"] == third["instructions"] == INSTRUCTIONS
    assert first["model"] == second["model"] == third["model"]
    assert first["reasoning_effort"] == second["reasoning_effort"] == third["reasoning_effort"]
    # Earlier turns are resent unchanged and new turns are appended, so the cached prefix stays valid.
    assert second["turns"][: len(first["turns"])] == first["turns"]
    assert third["turns"][: len(second["turns"])] == second["turns"]
    assert [turn.role for turn in third["turns"]] == ["user", "assistant", "user", "assistant", "user"]
    assert third["turns"][1] == ChatTurn("assistant", "Hello operator")


def test_instructions_are_identical_across_conversations(client, provider: FakeProvider):
    for index in range(2):
        chat_id = client.post("/api/chats", json={}).json()["id"]
        send(client, chat_id, f"client-req-shared-{index}", "Same opening")
    assert provider.calls[0]["instructions"] == provider.calls[1]["instructions"]
    assert provider.calls[0]["turns"] == provider.calls[1]["turns"]


def test_retry_resends_the_same_prompt_as_the_failed_attempt(client, provider: FakeProvider):
    chat_id = client.post("/api/chats", json={}).json()["id"]
    send(client, chat_id, "client-req-retry-a", "First question")
    provider.fail = True
    failed = parse_sse(send(client, chat_id, "client-req-retry-b", "Second question").text)[-1]
    provider.fail = False
    client.post(f"/api/chats/{chat_id}/messages/{failed['message']['id']}/retry")

    attempt, retry = provider.calls[1], provider.calls[2]
    assert retry["turns"] == attempt["turns"]
    assert retry["instructions"] == attempt["instructions"]


# --- Cache accounting -------------------------------------------------------------


def test_savings_split_into_read_discount_and_write_premium():
    # OpenAI's example: 12,000 input = 1,000 ordinary + 4,000 cached reads + 7,000 30-minute writes.
    cost = estimate_cost(
        PRICING.find("gpt-6-luna"),
        input_tokens=12_000,
        output_tokens=0,
        cached_input_tokens=4_000,
        cache_write_tokens=7_000,
    )
    assert cost.input == (1000 * Decimal("0.10") + 4000 * Decimal("0.01") + 7000 * Decimal("0.125")) / M
    assert cost.uncached_baseline_input == 12_000 * Decimal("0.10") / M
    assert cost.cache_read_savings == 4000 * (Decimal("0.10") - Decimal("0.01")) / M
    assert cost.cache_write_overhead == 7000 * (Decimal("0.125") - Decimal("0.10")) / M
    assert cost.cache_savings == cost.uncached_baseline_input - cost.input
    assert cost.total == cost.input + cost.output


def test_first_call_that_only_writes_costs_more_than_uncached():
    cost = estimate_cost(PRICING.find("gpt-6-luna"), input_tokens=1_500, output_tokens=0, cache_write_tokens=1_400)
    assert cost.cache_read_savings == 0
    assert cost.cache_savings < 0
    assert cost.input == (100 * Decimal("0.10") + 1400 * Decimal("0.125")) / M


def test_model_without_write_rate_has_no_write_overhead():
    cost = estimate_cost(
        PRICING.find("gpt-5-mini-2025-08-07"),
        input_tokens=2_000,
        output_tokens=0,
        cached_input_tokens=1_024,
        cache_write_tokens=500,
    )
    assert cost.cache_write_overhead == 0
    assert cost.cache_read_savings == 1024 * (Decimal("0.25") - Decimal("0.025")) / M
    assert cost.uncached_baseline_input == 2000 * Decimal("0.25") / M


def test_long_context_baseline_uses_long_context_rates():
    cost = estimate_cost(PRICING.find("gpt-6-luna"), input_tokens=300_000, output_tokens=0, cached_input_tokens=200_000)
    assert cost.long_context is True
    assert cost.uncached_baseline_input == 300_000 * Decimal("0.20") / M
    assert cost.cache_read_savings == 200_000 * (Decimal("0.20") - Decimal("0.02")) / M


def test_missing_cache_fields_mean_no_savings_and_no_overhead():
    cost = estimate_cost(PRICING.find("gpt-6-luna"), input_tokens=2_000, output_tokens=10)
    assert cost.cache_savings == 0
    assert cost.input == cost.uncached_baseline_input


def test_extract_usage_tolerates_missing_or_extra_cache_details():
    no_details = extract_usage({"input_tokens": 1500, "output_tokens": 20})
    assert no_details == LLMUsage(input_tokens=1500, output_tokens=20)
    detailed = extract_usage(
        {
            "input_tokens": 1500,
            "output_tokens": 20,
            "input_tokens_details": {
                "cached_tokens": 1024,
                "cache_write_tokens": 300,
                "cache_write_tokens_details": {"ttl_30m_tokens": 300, "ttl_12h_tokens": 0},
            },
        }
    )
    assert (detailed.cached_input_tokens, detailed.cache_write_tokens) == (1024, 300)
    assert extract_usage({"input_tokens": 1500, "output_tokens": 20, "input_tokens_details": None}).cached_input_tokens is None


def test_dashboard_hit_rate_and_savings():
    now = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)
    records = [
        # Second turn: most of the prompt was read from cache.
        record(request_id="r2", input_tokens=2_000, output_tokens=0, cached_input_tokens=1_500, cache_write_tokens=400),
        # First turn: writes only.
        record(request_id="r1", input_tokens=1_500, output_tokens=0, cached_input_tokens=0, cache_write_tokens=1_500),
        # Imported row without cached detail: excluded from the hit-rate denominator.
        record(request_id="r0", input_tokens=10_000, output_tokens=0, cached_input_tokens=None, cache_write_tokens=None, source="backfill"),
        # Stopped without usage: ignored entirely.
        record(request_id="rx", usage_reported=False, input_tokens=None, output_tokens=None, cached_input_tokens=None, cache_write_tokens=None, status="stopped"),
    ]
    totals = aggregate_usage(records, PRICING, days=None, now=now)["totals"]

    assert totals["cache_hit_rate"] == pytest.approx(1_500 / 3_500, abs=1e-4)
    read = 1_500 * 0.09 / 1e6
    overhead = (400 + 1_500) * 0.025 / 1e6
    assert totals["estimated_cache_read_savings_usd"] == pytest.approx(read)
    assert totals["estimated_cache_write_overhead_usd"] == pytest.approx(overhead)
    assert totals["estimated_cache_savings_usd"] == pytest.approx(read - overhead)
    assert totals["estimated_uncached_input_cost_usd"] == pytest.approx(13_500 * 0.10 / 1e6)
    assert totals["estimated_input_cost_usd"] == pytest.approx(
        totals["estimated_uncached_input_cost_usd"] - totals["estimated_cache_savings_usd"]
    )


def test_unpriced_requests_count_toward_hit_rate_but_not_savings():
    now = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)
    report = aggregate_usage(
        [record(model="mystery-model", input_tokens=2_000, cached_input_tokens=1_000)], PRICING, days=None, now=now
    )
    totals = report["totals"]
    assert totals["cache_hit_rate"] == pytest.approx(0.5)
    assert totals["estimated_cache_savings_usd"] is None
    assert totals["estimated_uncached_input_cost_usd"] is None


def test_no_cached_counts_means_unknown_hit_rate():
    now = datetime(2026, 10, 25, 12, tzinfo=timezone.utc)
    totals = aggregate_usage([record(cached_input_tokens=None)], PRICING, days=None, now=now)["totals"]
    assert totals["cache_hit_rate"] is None
    assert aggregate_usage([], PRICING, days=None, now=now)["totals"]["cache_hit_rate"] is None


def test_usage_api_exposes_cache_metrics(client, provider: FakeProvider):
    provider.input_tokens = 2_000
    provider.cached_input_tokens = 1_024
    provider.cache_write_tokens = 512
    provider.output_tokens = 10
    chat_id = client.post("/api/chats", json={}).json()["id"]
    send(client, chat_id, "client-req-cache-api", "Question")

    report = client.get("/api/usage").json()
    totals = report["totals"]
    assert totals["cache_hit_rate"] == pytest.approx(0.512)
    assert totals["estimated_cache_read_savings_usd"] == pytest.approx(1_024 * 0.09 / 1e6)
    assert totals["estimated_cache_write_overhead_usd"] == pytest.approx(512 * 0.025 / 1e6)
    assert report["by_model"][0]["cache_hit_rate"] == pytest.approx(0.512)
    assert report["recent"][0]["estimated_cache_savings_usd"] == pytest.approx((1_024 * 0.09 - 512 * 0.025) / 1e6)
