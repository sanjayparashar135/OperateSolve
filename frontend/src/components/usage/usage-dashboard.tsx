"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import {
  ArrowLeftIcon,
  CircleAlertIcon,
  CoinsIcon,
  ExternalLinkIcon,
  InfoIcon,
  RefreshCwIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { getUsage, isAbortError } from "@/lib/api";
import {
  formatCompact,
  formatCount,
  formatDateTime,
  formatDay,
  formatLatency,
  formatPercent,
  formatRate,
  formatUsd,
  isSameModel,
} from "@/lib/format";
import type { DailyUsage, RouteUsage, UsageReport, UsageSummary } from "@/lib/types";
import { cn } from "@/lib/utils";

const RANGES: Array<{ label: string; days: number | null }> = [
  { label: "7 days", days: 7 },
  { label: "30 days", days: 30 },
  { label: "90 days", days: 90 },
  { label: "All time", days: null },
];

export function UsageDashboard() {
  const [days, setDays] = useState<number | null>(30);
  const [reloadKey, setReloadKey] = useState(0);
  const [report, setReport] = useState<UsageReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    getUsage(days, controller.signal)
      .then((next) => {
        setReport(next);
        setError(null);
      })
      .catch((reason: unknown) => {
        if (isAbortError(reason)) return;
        setError(reason instanceof Error ? reason.message : "Could not load usage.");
      })
      .finally(() => {
        if (!controller.signal.aborted) setLoading(false);
      });
    return () => controller.abort();
  }, [days, reloadKey]);

  function selectRange(next: number | null) {
    if (next === days) return;
    setLoading(true);
    setDays(next);
  }

  function refresh() {
    setLoading(true);
    setReloadKey((key) => key + 1);
  }

  const totals = report?.totals;

  return (
    <main className="usage-page">
      <div className="usage-inner">
        <Link href="/" className="back-link">
          <ArrowLeftIcon />
          Back to chats
        </Link>

        <header className="usage-head">
          <div>
            <p className="eyebrow">Observability</p>
            <h1>Usage &amp; cost</h1>
            <p className="usage-lede">
              Tokens reported by the OpenAI API for requests this OperateSolve backend sent with your API key. This is
              your OpenAI API spend, not Cursor usage.
            </p>
          </div>
          <div className="usage-controls">
            <div className="segmented" role="group" aria-label="Time range">
              {RANGES.map((range) => (
                <button
                  key={range.label}
                  type="button"
                  aria-pressed={days === range.days}
                  className={cn(days === range.days && "active")}
                  onClick={() => selectRange(range.days)}
                >
                  {range.label}
                </button>
              ))}
            </div>
            <Button type="button" variant="outline" className="refresh-button" onClick={refresh} disabled={loading}>
              <RefreshCwIcon className={cn(loading && "animate-spin")} />
              Refresh
            </Button>
          </div>
        </header>

        {error ? (
          <div className="banner warn" role="alert">
            {error} Check that the API is running, then refresh.
          </div>
        ) : null}

        {report ? <Notices report={report} /> : null}

        {!report && loading ? (
          <div className="kpi-grid">
            {Array.from({ length: 5 }, (_, index) => (
              <Skeleton key={index} className="h-[118px] w-full rounded-2xl" />
            ))}
          </div>
        ) : null}

        {report && totals ? (
          <div className={cn("usage-body", loading && "is-loading")}>
            <Kpis totals={totals} />
            {totals.requests === 0 ? (
              <section className="usage-card empty-usage">
                <CoinsIcon />
                <p>No model requests recorded in this range.</p>
                <p className="muted">Send a chat message, then refresh. Each streamed reply records one request.</p>
              </section>
            ) : (
              <>
                <DailyChart daily={report.daily} />
                <CacheCard report={report} />
                <ModelTable report={report} />
                <RouteTable report={report} />
                <ConversationTable report={report} />
                <RecentTable report={report} />
              </>
            )}
            <p className="usage-foot">
              {report.window_start ? `Since ${formatDateTime(report.window_start)} · ` : "All recorded requests · "}
              Generated {formatDateTime(report.generated_at)} · Daily buckets use UTC dates
            </p>
          </div>
        ) : null}
      </div>
    </main>
  );
}

function Notices({ report }: { report: UsageReport }) {
  const { pricing, totals } = report;
  return (
    <div className="notice-stack">
      <div className="banner notice" role="note">
        <InfoIcon />
        <p>
          <strong>Costs are estimates.</strong> Each request&apos;s reported tokens are multiplied by the rates in{" "}
          <code>backend/pricing.json</code> ({pricing.tier} tier
          {pricing.verified_at ? `, verified ${pricing.verified_at}` : ""}
          {pricing.source_url ? (
            <>
              {" against "}
              <a href={pricing.source_url} target="_blank" rel="noreferrer">
                OpenAI pricing <ExternalLinkIcon className="inline-icon" />
              </a>
            </>
          ) : null}
          ). Your OpenAI billing dashboard is the source of truth.
        </p>
      </div>
      {pricing.error ? (
        <div className="banner warn" role="alert">
          <CircleAlertIcon />
          <p>
            Pricing config could not be loaded, so costs are unavailable: {pricing.error}
          </p>
        </div>
      ) : null}
      {totals.requests_missing_usage > 0 ? (
        <div className="banner caution" role="note">
          <CircleAlertIcon />
          <p>
            <strong>{formatCount(totals.requests_missing_usage)}</strong> request
            {totals.requests_missing_usage === 1 ? "" : "s"} finished without usage from OpenAI (stopped, failed, or
            interrupted). They are counted as requests but contribute no tokens or cost, so totals may be low.
          </p>
        </div>
      ) : null}
      {totals.requests_unpriced > 0 && !pricing.error ? (
        <div className="banner caution" role="note">
          <CircleAlertIcon />
          <p>
            <strong>{formatCount(totals.requests_unpriced)}</strong> request
            {totals.requests_unpriced === 1 ? " uses a model" : "s use models"} with no rate in{" "}
            <code>backend/pricing.json</code>. Their tokens are counted but not costed.
          </p>
        </div>
      ) : null}
      {totals.requests_backfilled > 0 ? (
        <div className="banner subtle" role="note">
          <InfoIcon />
          <p>
            {formatCount(totals.requests_backfilled)} request{totals.requests_backfilled === 1 ? " was" : "s were"}{" "}
            imported from Phase 1 messages. Those rows have no cached-token detail and only the final attempt of each
            retried message.
          </p>
        </div>
      ) : null}
    </div>
  );
}

function Kpis({ totals }: { totals: UsageSummary }) {
  return (
    <div className="kpi-grid">
      <article className="kpi featured">
        <p className="kpi-label">
          Estimated API cost <span className="est-badge">Estimate</span>
        </p>
        <p className="kpi-value">{formatUsd(totals.estimated_cost_usd)}</p>
        <p className="kpi-sub">
          {totals.estimated_cost_usd == null
            ? "No priced usage yet"
            : `${formatCount(totals.requests_with_usage - totals.requests_unpriced)} priced requests · USD`}
        </p>
      </article>
      <article className="kpi">
        <p className="kpi-label">Requests</p>
        <p className="kpi-value">{formatCount(totals.requests)}</p>
        <p className="kpi-sub">
          {formatCount(totals.requests_with_usage)} with usage
          {totals.requests_missing_usage > 0 ? (
            <span className="warn-text"> · {formatCount(totals.requests_missing_usage)} missing</span>
          ) : null}
        </p>
      </article>
      <article className="kpi">
        <p className="kpi-label">Input tokens</p>
        <p className="kpi-value" title={formatCount(totals.input_tokens)}>
          {formatCompact(totals.input_tokens)}
        </p>
        <p className="kpi-sub">
          {formatCompact(totals.cached_input_tokens)} cached
          {totals.cache_hit_rate != null ? ` (${formatPercent(totals.cache_hit_rate)})` : ""}
          {totals.cache_write_tokens > 0 ? ` · ${formatCompact(totals.cache_write_tokens)} cache writes` : ""}
        </p>
      </article>
      <article className="kpi">
        <p className="kpi-label">Output tokens</p>
        <p className="kpi-value" title={formatCount(totals.output_tokens)}>
          {formatCompact(totals.output_tokens)}
        </p>
        <p className="kpi-sub">{formatCompact(totals.reasoning_tokens)} reasoning (included)</p>
      </article>
      <article className="kpi">
        <p className="kpi-label">Average latency</p>
        <p className="kpi-value">{formatLatency(totals.average_latency_ms)}</p>
        <p className="kpi-sub">Full stream, completed requests</p>
      </article>
    </div>
  );
}

function DailyChart({ daily }: { daily: DailyUsage[] }) {
  const [metric, setMetric] = useState<"cost" | "tokens">("tokens");
  const values = daily.map((day) =>
    metric === "cost" ? (day.estimated_cost_usd ?? 0) : day.input_tokens + day.output_tokens,
  );
  const max = Math.max(...values, 0);
  const labelEvery = Math.max(1, Math.ceil(daily.length / 7));

  return (
    <section className="usage-card">
      <header className="card-head">
        <div>
          <h2>Daily activity</h2>
          <p>{metric === "cost" ? "Estimated USD per UTC day" : "Input + output tokens per UTC day"}</p>
        </div>
        <div className="segmented small" role="group" aria-label="Chart metric">
          <button type="button" aria-pressed={metric === "tokens"} className={cn(metric === "tokens" && "active")} onClick={() => setMetric("tokens")}>
            Tokens
          </button>
          <button type="button" aria-pressed={metric === "cost"} className={cn(metric === "cost" && "active")} onClick={() => setMetric("cost")}>
            Est. cost
          </button>
        </div>
      </header>
      <div className="chart" role="img" aria-label={`Daily ${metric === "cost" ? "estimated cost" : "tokens"}`}>
        {daily.map((day, index) => {
          const value = values[index];
          const height = max > 0 ? Math.max((value / max) * 100, value > 0 ? 3 : 0) : 0;
          const tip = `${formatDay(day.date)}: ${formatCount(day.requests)} requests · ${formatCount(
            day.input_tokens,
          )} in / ${formatCount(day.output_tokens)} out · ${formatUsd(day.estimated_cost_usd)} est.`;
          return (
            <div key={day.date} className="chart-col" title={tip}>
              <div className="chart-track">
                <div className="chart-bar" style={{ height: `${height}%` }} />
              </div>
              <span className="chart-label">{index % labelEvery === 0 || index === daily.length - 1 ? formatDay(day.date) : ""}</span>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function CacheCard({ report }: { report: UsageReport }) {
  const { totals } = report;
  return (
    <section className="usage-card">
      <header className="card-head">
        <div>
          <h2>
            Prompt caching <span className="est-badge">Estimate</span>
          </h2>
          <p>
            OpenAI caches repeated prompt prefixes automatically. Hit rate is cached-read input tokens ÷ input tokens.
            Savings compare the actual input cost with the same tokens billed at the uncached input rate: reads save
            money, while cache writes on Luna cost 1.25× input, so a first call that only writes costs more than uncached.
            {totals.requests_cached_unknown > 0
              ? ` ${formatCount(totals.requests_cached_unknown)} imported request${
                  totals.requests_cached_unknown === 1 ? " has" : "s have"
                } no cached detail and ${totals.requests_cached_unknown === 1 ? "is" : "are"} left out of the hit rate.`
              : ""}
          </p>
        </div>
      </header>
      <div className="kpi-grid">
        <article className="kpi">
          <p className="kpi-label">Cache hit rate</p>
          <p className="kpi-value">{formatPercent(totals.cache_hit_rate)}</p>
          <p className="kpi-sub">Cached reads ÷ input tokens</p>
        </article>
        <article className="kpi">
          <p className="kpi-label">Cached-read tokens</p>
          <p className="kpi-value" title={formatCount(totals.cached_input_tokens)}>
            {formatCompact(totals.cached_input_tokens)}
          </p>
          <p className="kpi-sub">Billed at the cached-input rate</p>
        </article>
        <article className="kpi">
          <p className="kpi-label">Cache-write tokens</p>
          <p className="kpi-value" title={formatCount(totals.cache_write_tokens)}>
            {formatCompact(totals.cache_write_tokens)}
          </p>
          <p className="kpi-sub">Billed at the cache-write rate where one applies</p>
        </article>
        <article className="kpi featured">
          <p className="kpi-label">Net input savings</p>
          <p className="kpi-value">{formatUsd(totals.estimated_cache_savings_usd)}</p>
          <p className="kpi-sub">
            {formatUsd(totals.estimated_cache_read_savings_usd)} saved on reads −{" "}
            {formatUsd(totals.estimated_cache_write_overhead_usd)} write overhead
          </p>
        </article>
        <article className="kpi">
          <p className="kpi-label">Input cost</p>
          <p className="kpi-value">{formatUsd(totals.estimated_input_cost_usd)}</p>
          <p className="kpi-sub">vs {formatUsd(totals.estimated_uncached_input_cost_usd)} if nothing were cached</p>
        </article>
      </div>
      <div className="table-wrap cache-table">
        <table className="usage-table">
          <thead>
            <tr>
              <th>Model</th>
              <th className="num">Input</th>
              <th className="num">Cached reads</th>
              <th className="num">Cache writes</th>
              <th className="num">Hit rate</th>
              <th className="num">Avg first token</th>
              <th className="num">Input cost</th>
              <th className="num">Uncached baseline</th>
              <th className="num">Read savings</th>
              <th className="num">Write overhead</th>
              <th className="num">Net savings</th>
            </tr>
          </thead>
          <tbody>
            {report.by_model.map((row) => (
              <tr key={row.model}>
                <td className="mono">
                  {row.display_name ? <span className="strong">{row.display_name} </span> : null}
                  <span className={cn(row.display_name && "muted")}>{row.model}</span>
                </td>
                <td className="num">{formatCount(row.input_tokens)}</td>
                <td className="num">{formatCount(row.cached_input_tokens)}</td>
                <td className="num">{formatCount(row.cache_write_tokens)}</td>
                <td className="num">{formatPercent(row.cache_hit_rate)}</td>
                <td className="num">{formatLatency(row.average_first_token_ms)}</td>
                <td className="num">{formatUsd(row.estimated_input_cost_usd)}</td>
                <td className="num muted">{formatUsd(row.estimated_uncached_input_cost_usd)}</td>
                <td className="num">{formatUsd(row.estimated_cache_read_savings_usd)}</td>
                <td className="num">{formatUsd(row.estimated_cache_write_overhead_usd)}</td>
                <td className={cn("num strong", (row.estimated_cache_savings_usd ?? 0) < 0 && "warn-text")}>
                  {formatUsd(row.estimated_cache_savings_usd)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

const ROUTE_LABELS: Record<RouteUsage["routing_mode"], string> = {
  manual: "Manual",
  auto: "Auto",
  fallback: "Auto fallback",
  legacy: "Before routing",
};

function ModelTable({ report }: { report: UsageReport }) {
  return (
    <section className="usage-card">
      <header className="card-head">
        <div>
          <h2>By model</h2>
          <p>
            Compare cost and speed per model. Latency is the full stream; first token is time to the first text.
            Rates are USD per 1M tokens from the backend pricing config.
          </p>
        </div>
      </header>
      <div className="table-wrap">
        <table className="usage-table">
          <thead>
            <tr>
              <th>Model</th>
              <th className="num">Requests</th>
              <th className="num">Input</th>
              <th className="num">Cached</th>
              <th className="num">Cache writes</th>
              <th className="num">Output</th>
              <th className="num">Avg first token</th>
              <th className="num">Avg latency</th>
              <th className="num">Avg cost / req</th>
              <th className="num">Est. cost</th>
              <th>Rates in / cached / write / out</th>
            </tr>
          </thead>
          <tbody>
            {report.by_model.map((row) => (
              <tr key={row.model}>
                <td className="mono">
                  {row.display_name ? <span className="strong">{row.display_name} </span> : null}
                  <span className={cn(row.display_name && "muted")}>{row.model}</span>
                </td>
                <td className="num">
                  {formatCount(row.requests)}
                  <MissingNote count={row.requests_missing_usage} />
                </td>
                <td className="num">{formatCount(row.input_tokens)}</td>
                <td className="num">{formatCount(row.cached_input_tokens)}</td>
                <td className="num">{formatCount(row.cache_write_tokens)}</td>
                <td className="num">{formatCount(row.output_tokens)}</td>
                <td className="num">{formatLatency(row.average_first_token_ms)}</td>
                <td className="num">{formatLatency(row.average_latency_ms)}</td>
                <td className="num">{formatUsd(row.average_cost_usd)}</td>
                <td className="num strong">{row.rates ? formatUsd(row.estimated_cost_usd) : <span className="warn-text">unpriced</span>}</td>
                <td className="mono muted">
                  {row.rates ? (
                    `${formatRate(row.rates.input)} / ${formatRate(row.rates.cached_input)} / ${formatRate(
                      row.rates.cache_write,
                    )} / ${formatRate(row.rates.output)}`
                  ) : (
                    <span className="warn-text">Not in pricing config</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function RouteTable({ report }: { report: UsageReport }) {
  return (
    <section className="usage-card">
      <header className="card-head">
        <div>
          <h2>Routing decisions</h2>
          <p>
            One row per route taken. Auto maps Simple → Nano, General → Luna, Detailed → Mini; a fallback means the Auto
            target was unavailable and Luna answered instead.
          </p>
        </div>
      </header>
      <div className="table-wrap">
        <table className="usage-table">
          <thead>
            <tr>
              <th>Route</th>
              <th>Requested → answered by</th>
              <th className="num">Requests</th>
              <th className="num">Input</th>
              <th className="num">Output</th>
              <th className="num">Avg first token</th>
              <th className="num">Avg latency</th>
              <th className="num">Est. cost</th>
            </tr>
          </thead>
          <tbody>
            {report.by_route.map((row) => (
              <tr key={`${row.routing_mode}:${row.task_mode ?? ""}:${row.requested_model}:${row.model}`}>
                <td className="nowrap">
                  <span className={cn("status-pill", row.routing_mode === "fallback" ? "stopped" : row.routing_mode === "legacy" ? "" : "complete")}>
                    {ROUTE_LABELS[row.routing_mode] ?? row.routing_mode}
                  </span>
                  {row.task_mode ? <span className="muted">{capitalize(row.task_mode)}</span> : null}
                </td>
                <td className="mono">
                  {isSameModel(row.requested_model, row.model) ? row.model : `${row.requested_model} → ${row.model}`}
                </td>
                <td className="num">
                  {formatCount(row.requests)}
                  <MissingNote count={row.requests_missing_usage} />
                </td>
                <td className="num">{formatCount(row.input_tokens)}</td>
                <td className="num">{formatCount(row.output_tokens)}</td>
                <td className="num">{formatLatency(row.average_first_token_ms)}</td>
                <td className="num">{formatLatency(row.average_latency_ms)}</td>
                <td className="num strong">{formatUsd(row.estimated_cost_usd)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function capitalize(text: string): string {
  return text.charAt(0).toUpperCase() + text.slice(1);
}

function ConversationTable({ report }: { report: UsageReport }) {
  return (
    <section className="usage-card">
      <header className="card-head">
        <div>
          <h2>By conversation</h2>
          <p>Deleted conversations keep their usage so totals stay complete</p>
        </div>
      </header>
      <div className="table-wrap">
        <table className="usage-table">
          <thead>
            <tr>
              <th>Conversation</th>
              <th className="num">Requests</th>
              <th className="num">Input</th>
              <th className="num">Cached</th>
              <th className="num">Output</th>
              <th className="num">Est. cost</th>
              <th>Last request</th>
            </tr>
          </thead>
          <tbody>
            {report.by_conversation.map((row) => (
              <tr key={row.conversation_id ?? "none"}>
                <td className="title-cell">
                  {row.deleted || !row.conversation_id ? (
                    <span className="muted">Deleted conversation</span>
                  ) : (
                    <Link href={`/?c=${row.conversation_id}`}>{row.title}</Link>
                  )}
                </td>
                <td className="num">
                  {formatCount(row.requests)}
                  <MissingNote count={row.requests_missing_usage} />
                </td>
                <td className="num">{formatCount(row.input_tokens)}</td>
                <td className="num">{formatCount(row.cached_input_tokens)}</td>
                <td className="num">{formatCount(row.output_tokens)}</td>
                <td className="num strong">{formatUsd(row.estimated_cost_usd)}</td>
                <td className="nowrap muted">{formatDateTime(row.last_request_at)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function RecentTable({ report }: { report: UsageReport }) {
  return (
    <section className="usage-card">
      <header className="card-head">
        <div>
          <h2>Recent requests</h2>
          <p>Latest {report.recent.length} model calls, newest first</p>
        </div>
      </header>
      <div className="table-wrap">
        <table className="usage-table">
          <thead>
            <tr>
              <th>Time</th>
              <th>Conversation</th>
              <th>Model</th>
              <th>Status</th>
              <th className="num">Input</th>
              <th className="num">Cached</th>
              <th className="num">Cache writes</th>
              <th className="num">Output</th>
              <th className="num">First token</th>
              <th className="num">Duration</th>
              <th className="num">Est. cost</th>
            </tr>
          </thead>
          <tbody>
            {report.recent.map((row) => (
              <tr key={row.request_id} title={`Request ${row.request_id}`}>
                <td className="nowrap">{formatDateTime(row.started_at)}</td>
                <td className="title-cell">
                  {row.purpose === "incident_analysis" ? (
                    <span className="muted">Incident analysis</span>
                  ) : row.purpose === "parallel_experiment" ? (
                    <span className="muted">LLM experiment</span>
                  ) : (
                    row.conversation_title ?? <span className="muted">Deleted conversation</span>
                  )}
                </td>
                <td className="mono" title={row.routing_reason ?? undefined}>
                  <div className="route-cell">
                    <span>{row.model ?? "—"}</span>
                    {row.routing_mode ? (
                      <span className="muted">
                        {ROUTE_LABELS[row.routing_mode]}
                        {row.task_mode ? ` · ${capitalize(row.task_mode)}` : ""}
                        {row.requested_model && !isSameModel(row.requested_model, row.model) ? ` · asked ${row.requested_model}` : ""}
                      </span>
                    ) : null}
                  </div>
                </td>
                <td>
                  <span className={cn("status-pill", row.status)}>{row.status}</span>
                  {row.purpose === "summarization" ? <span className="status-pill imported">summary</span> : null}
                  {row.purpose === "incident_analysis" ? (
                    <span className="status-pill imported">incident</span>
                  ) : null}
                  {row.purpose === "parallel_experiment" ? (
                    <span className="status-pill imported">experiment</span>
                  ) : null}
                  {row.source === "backfill" ? <span className="status-pill imported">imported</span> : null}
                </td>
                {row.usage_reported ? (
                  <>
                    <td className="num">{formatCount(row.input_tokens)}</td>
                    <td className="num">{row.cached_input_tokens == null ? "n/a" : formatCount(row.cached_input_tokens)}</td>
                    <td className="num">{row.cache_write_tokens == null ? "n/a" : formatCount(row.cache_write_tokens)}</td>
                    <td className="num">{formatCount(row.output_tokens)}</td>
                  </>
                ) : (
                  <td className="num warn-text" colSpan={4}>
                    usage missing
                  </td>
                )}
                <td className="num">{formatLatency(row.first_token_ms)}</td>
                <td className="num">{formatLatency(row.duration_ms)}</td>
                <td className="num strong">
                  {row.estimated_cost_usd != null ? formatUsd(row.estimated_cost_usd) : row.usage_reported ? "unpriced" : "—"}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function MissingNote({ count }: { count: number }) {
  if (count === 0) return null;
  return (
    <span className="missing-note" title={`${count} without usage`}>
      {" "}
      ({count} missing)
    </span>
  );
}
