"""Usage dashboard response models. Costs are estimates in USD."""

from pydantic import BaseModel


class UsageSummary(BaseModel):
    requests: int
    requests_with_usage: int
    requests_missing_usage: int
    requests_unpriced: int
    requests_cached_unknown: int
    requests_backfilled: int
    input_tokens: int
    cached_input_tokens: int
    cache_write_tokens: int
    output_tokens: int
    reasoning_tokens: int
    # cached_input_tokens / input_tokens, over requests that reported a cached count.
    cache_hit_rate: float | None
    estimated_cost_usd: float | None
    average_cost_usd: float | None
    # Input-side estimates for priced requests. Savings compare against every
    # input token billed at the uncached rate; net = read savings - write overhead.
    estimated_input_cost_usd: float | None
    estimated_uncached_input_cost_usd: float | None
    estimated_cache_read_savings_usd: float | None
    estimated_cache_write_overhead_usd: float | None
    estimated_cache_savings_usd: float | None
    average_latency_ms: int | None
    average_first_token_ms: int | None
    first_request_at: str | None
    last_request_at: str | None


class ModelRates(BaseModel):
    input: float
    cached_input: float | None
    cache_write: float | None
    output: float
    long_context_above_input_tokens: int | None


class ModelUsage(UsageSummary):
    model: str
    display_name: str | None
    rates: ModelRates | None


class RouteUsage(UsageSummary):
    """routing_mode is manual, auto, fallback, or legacy (recorded before Phase 2.2)."""

    routing_mode: str
    task_mode: str | None
    requested_model: str
    model: str
    display_name: str | None


class ConversationUsage(UsageSummary):
    conversation_id: str | None
    title: str | None
    deleted: bool


class DailyUsage(BaseModel):
    date: str
    requests: int
    input_tokens: int
    output_tokens: int
    cached_input_tokens: int
    estimated_cost_usd: float | None


class RecentRequest(BaseModel):
    request_id: str
    started_at: str
    conversation_id: str | None
    conversation_title: str | None
    model: str | None
    requested_model: str | None
    selection: str | None
    task_mode: str | None
    routing_mode: str | None
    routing_reason: str | None
    first_token_ms: int | None
    purpose: str
    status: str
    source: str
    usage_reported: bool
    priced: bool
    input_tokens: int | None
    cached_input_tokens: int | None
    cache_write_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    duration_ms: int | None
    estimated_cost_usd: float | None
    estimated_cache_savings_usd: float | None


class PricingSummary(BaseModel):
    currency: str
    tier: str
    source_url: str | None
    verified_at: str | None
    notes: str | None
    error: str | None
    models: list[str]


class UsageReport(BaseModel):
    generated_at: str
    days: int | None
    window_start: str | None
    pricing: PricingSummary
    totals: UsageSummary
    by_model: list[ModelUsage]
    by_route: list[RouteUsage]
    by_conversation: list[ConversationUsage]
    daily: list[DailyUsage]
    recent: list[RecentRequest]
