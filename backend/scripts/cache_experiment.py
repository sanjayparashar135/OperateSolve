"""Manual prompt-caching experiment. Makes a few small, billed OpenAI calls.

Run from the backend directory:

    .\\.venv\\Scripts\\python.exe -m scripts.cache_experiment
    .\\.venv\\Scripts\\python.exe -m scripts.cache_experiment --mode repeat --calls 4
    .\\.venv\\Scripts\\python.exe -m scripts.cache_experiment --mode new-question
    .\\.venv\\Scripts\\python.exe -m scripts.cache_experiment --model nano

It sends a ~2,600-token synthetic runbook (above Luna's 1,024-token minimum)
through the same provider, instructions, and cache settings the app uses, then
prints cached tokens, cache writes, input cost against an uncached baseline,
and time to first token for each call. Nothing is written to the database.

Modes:
  conversation  Each call appends the previous answer and a new question, like
                a chat. Calls after the first should read the earlier turns
                from cache.
  repeat        The identical prompt every call.
  new-question  Same runbook, different question, no history. On Luna this is
                expected to miss: the implicit breakpoint sits after the
                question, so the shared runbook alone was never written.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
import uuid
from dataclasses import dataclass

from app.config import get_settings
from app.services.chat_service import INSTRUCTIONS
from app.services.llm import prompt_cache
from app.services.llm.base import ChatTurn, LLMDelta, LLMDone, LLMError, LLMUsage
from app.services.llm.openai_provider import OpenAIResponsesProvider
from app.services.llm.registry import MODEL_SPECS
from app.services.pricing import PricingConfigError, estimate_cost, load_pricing

QUESTIONS = [
    "Which service should page the database on-call first if ledger lag grows? One sentence.",
    "What is the first check for checkout-web 5xx errors? One sentence.",
    "Who owns auth-gateway, and what is its rollback step? One sentence.",
    "Which dashboard shows notification-worker backlog? One sentence.",
    "When should search-indexer be paused? One sentence.",
    "What is the escalation path for inventory-sync? One sentence.",
    "What latency SLO does payments-api have? One sentence.",
    "How is reporting-batch rolled back? One sentence.",
    "Which services share the primary Postgres cluster? One sentence.",
    "What should an operator post in the first status update? One sentence.",
]

SERVICES = [
    ("payments-api", "Payments", "p99 under 300 ms", "primary Postgres"),
    ("ledger-writer", "Ledger", "replication lag under 5 s", "primary Postgres"),
    ("checkout-web", "Storefront", "5xx rate under 0.5%", "edge cache"),
    ("auth-gateway", "Identity", "login success above 99.5%", "session store"),
    ("notification-worker", "Messaging", "queue age under 2 min", "message bus"),
    ("search-indexer", "Discovery", "index freshness under 10 min", "search cluster"),
    ("inventory-sync", "Supply", "sync drift under 1%", "primary Postgres"),
    ("reporting-batch", "Analytics", "nightly run under 90 min", "warehouse"),
    ("pricing-engine", "Pricing", "p95 under 150 ms", "rules cache"),
    ("fraud-scorer", "Risk", "decision time under 80 ms", "feature store"),
]


def runbook(run_tag: str | None) -> str:
    lines = []
    if run_tag:
        lines.append(f"Experiment run {run_tag}.")
    lines += [
        "OPERATESOLVE REFERENCE RUNBOOK (synthetic, for a caching experiment).",
        "Use this runbook to answer the questions that follow. Keep answers short.",
        "",
    ]
    for name, team, slo, dependency in SERVICES:
        lines += [
            f"## {name}",
            f"- Owner: the {team} team. Primary on-call rotates weekly; secondary is the {team} lead.",
            f"- SLO: {slo}, measured over a rolling 28-day window. Error budget alerts fire at 2% and 5% burn per hour.",
            f"- Dependency: {dependency}. If {dependency} is degraded, treat {name} symptoms as downstream first.",
            f"- Dashboards: '{name} overview' for golden signals, '{name} deep dive' for per-endpoint latency and saturation.",
            f"- First checks: confirm the last deploy time, compare error rate with the previous 24 hours, check {dependency} health, then check recent config changes.",
            f"- Mitigation: scale out by two replicas, shed non-critical traffic with the {name} kill switch, or fail over to the secondary region if both zones are affected.",
            f"- Rollback: redeploy the previous release tag through the pipeline; do not hot-patch {name} in production.",
            f"- Escalation: page {team} secondary after 15 minutes without mitigation, then the incident commander after 30 minutes.",
            f"- Status update: state customer impact for {name}, the mitigation in progress, and the next update time.",
            "",
        ]
    return "\n".join(lines)


@dataclass
class CallResult:
    usage: LLMUsage | None
    first_token_ms: int | None
    total_ms: int
    text: str
    model: str | None


async def one_call(provider, turns, model_id: str, effort: str | None) -> CallResult:
    started = time.perf_counter()
    first_token_ms: int | None = None
    parts: list[str] = []
    async for event in provider.stream(turns, model=model_id, instructions=INSTRUCTIONS, reasoning_effort=effort):
        if isinstance(event, LLMDelta) and event.text:
            if first_token_ms is None:
                first_token_ms = int((time.perf_counter() - started) * 1000)
            parts.append(event.text)
        elif isinstance(event, LLMError):
            raise RuntimeError(event.message)
        elif isinstance(event, LLMDone):
            return CallResult(event.usage, first_token_ms, int((time.perf_counter() - started) * 1000), "".join(parts), event.model)
    raise RuntimeError("The stream ended without a final event.")


def usd(value) -> str:
    return "n/a" if value is None else f"${float(value):.7f}"


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="luna", help="Registry key or model id (default: luna)")
    parser.add_argument("--mode", choices=["conversation", "repeat", "new-question"], default="conversation")
    parser.add_argument("--calls", type=int, default=3, help="Number of calls, 2-10 (default: 3)")
    parser.add_argument("--pause", type=float, default=2.0, help="Seconds between calls (default: 2)")
    parser.add_argument(
        "--reuse-prefix",
        action="store_true",
        help="Omit the per-run tag so a rerun within the cache lifetime can hit the previous run's cache",
    )
    args = parser.parse_args()
    calls = max(2, min(args.calls, 10))

    settings = get_settings()
    if not settings.llm_configured:
        print("OPENAI_API_KEY is not set in backend/.env.", file=sys.stderr)
        return 2
    spec = next((s for s in MODEL_SPECS if args.model.lower() in {s.key, s.model_id}), None)
    if spec is None:
        print(f"Unknown model '{args.model}'. Use one of: {', '.join(s.key for s in MODEL_SPECS)}.", file=sys.stderr)
        return 2
    try:
        pricing = load_pricing(settings.pricing_file).find(spec.model_id)
    except PricingConfigError as exc:
        print(f"Pricing unavailable ({exc}); costs will be n/a.")
        pricing = None

    # Short answers and the lowest reasoning effort keep each call well under $0.001.
    provider = OpenAIResponsesProvider(settings.model_copy(update={"openai_max_output_tokens": 200}))
    effort = spec.reasoning_efforts[0]
    policy = prompt_cache.policy_for(spec.model_id)
    prefix = runbook(None if args.reuse_prefix else uuid.uuid4().hex[:8])

    print(f"Model {spec.model_id} · mode {args.mode} · {calls} calls · reasoning effort {effort}")
    print(f"Cache parameters sent: {prompt_cache.request_params(spec.model_id) or 'none'}")
    if policy.min_prefix_tokens:
        print(f"Documented minimum cacheable prefix: {policy.min_prefix_tokens} tokens")
    print()

    header = f"{'call':>4} {'input':>6} {'cached':>6} {'writes':>6} {'hit':>5} {'input cost':>11} {'uncached':>11} {'saved':>11} {'ttft':>7} {'total':>7}"
    print(header)
    print("-" * len(header))

    history: list[ChatTurn] = []
    results: list[tuple[CallResult, object]] = []
    for index in range(calls):
        question = QUESTIONS[0] if args.mode == "repeat" else QUESTIONS[index % len(QUESTIONS)]
        if args.mode == "conversation":
            content = f"{prefix}\n\nQuestion: {question}" if index == 0 else question
            turns = [*history, ChatTurn("user", content)]
        else:
            turns = [ChatTurn("user", f"{prefix}\n\nQuestion: {question}")]
        try:
            result = await one_call(provider, turns, spec.model_id, effort)
        except RuntimeError as exc:
            print(f"Call {index + 1} failed: {exc}", file=sys.stderr)
            return 1
        if args.mode == "conversation":
            history = [*turns, ChatTurn("assistant", result.text)]

        usage = result.usage
        cost = None
        if usage and usage.reported and pricing:
            cost = estimate_cost(
                pricing,
                input_tokens=usage.input_tokens,
                output_tokens=usage.output_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                cache_write_tokens=usage.cache_write_tokens,
            )
        results.append((result, cost))
        inp = usage.input_tokens if usage else None
        cached = usage.cached_input_tokens if usage else None
        writes = usage.cache_write_tokens if usage else None
        hit = f"{cached / inp:.0%}" if inp and cached is not None else "n/a"
        print(
            f"{index + 1:>4} {inp if inp is not None else 'n/a':>6} {cached if cached is not None else 'n/a':>6} "
            f"{writes if writes is not None else 'n/a':>6} {hit:>5} {usd(cost.input if cost else None):>11} "
            f"{usd(cost.uncached_baseline_input if cost else None):>11} {usd(cost.cache_savings if cost else None):>11} "
            f"{(str(result.first_token_ms) + ' ms') if result.first_token_ms is not None else 'n/a':>7} {str(result.total_ms) + ' ms':>7}"
        )
        if index < calls - 1:
            await asyncio.sleep(args.pause)

    first, rest = results[0], results[1:]
    later_cached = sum((r.usage.cached_input_tokens or 0) for r, _ in rest if r.usage)
    later_input = sum((r.usage.input_tokens or 0) for r, _ in rest if r.usage)
    ttfts = [r.first_token_ms for r, _ in rest if r.first_token_ms is not None]
    print()
    print(f"First call time to first token: {first[0].first_token_ms} ms")
    if ttfts:
        print(f"Later calls average time to first token: {round(sum(ttfts) / len(ttfts))} ms")
    if later_input:
        print(f"Later calls cache hit rate: {later_cached / later_input:.0%} of input tokens")
    net = sum((c.cache_savings for _, c in results if c), start=0)
    print(f"Estimated net input savings across all calls: {usd(net)} (negative means write premiums exceeded read discounts)")
    if (first[0].usage and first[0].usage.input_tokens or 0) < (policy.min_prefix_tokens or 0):
        print("Warning: the first prompt is below the documented minimum, so it cannot be cached.")
    if later_input and later_cached == 0:
        print()
        print("No cache hits. Common reasons:")
        print("- new-question mode on Luna: expected; the shared runbook ends inside the user message, not at a breakpoint.")
        print("- The calls landed on different cache machines; caching is best-effort. Rerun once.")
        print("- More than the cache lifetime passed between calls, or the model or reasoning effort changed.")
        print("- The model does not support caching for this request (check the model page).")
    print()
    print("Costs are estimates from backend/pricing.json; the OpenAI billing dashboard is authoritative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
