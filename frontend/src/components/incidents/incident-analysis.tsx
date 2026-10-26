"use client";

import { useState, type FormEvent } from "react";
import Link from "next/link";
import { ArrowLeftIcon, LoaderCircleIcon, ShieldAlertIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { analyzeIncident, isAbortError } from "@/lib/api";
import { formatCount, formatLatency, formatUsd } from "@/lib/format";
import type { IncidentAnalyzeResponse, IncidentSeverity } from "@/lib/types";
import { cn } from "@/lib/utils";

const SEVERITY_LABEL: Record<IncidentSeverity, string> = {
  P1: "Critical",
  P2: "High",
  P3: "Medium",
  P4: "Low",
};

export function IncidentAnalysisPanel() {
  const [incidentId, setIncidentId] = useState("INC-1001");
  const [description, setDescription] = useState(
    "Payments p95 latency rose from 120ms to 4.5s after the 14:00 deploy. Checkout errors are elevated in us-east-1; auth and catalog look healthy.",
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<IncidentAnalyzeResponse | null>(null);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    const id = incidentId.trim();
    const text = description.trim();
    if (!id || !text) {
      setError("Incident ID and description are required.");
      return;
    }
    const controller = new AbortController();
    setLoading(true);
    setError(null);
    try {
      const next = await analyzeIncident(id, text, controller.signal);
      setResult(next);
    } catch (reason: unknown) {
      if (isAbortError(reason)) return;
      setResult(null);
      setError(reason instanceof Error ? reason.message : "Could not analyze the incident.");
    } finally {
      setLoading(false);
    }
  }

  const analysis = result?.analysis;

  return (
    <main className="incident-page">
      <div className="incident-inner">
        <Link href="/" className="back-link">
          <ArrowLeftIcon />
          Back to chats
        </Link>

        <header className="incident-head">
          <div>
            <p className="eyebrow">Phase 2.5A</p>
            <h1>Incident analysis</h1>
            <p className="incident-lede">
              Submit a fictional incident. The model returns a schema-enforced assessment — severity,
              probable cause, and recommended actions — without changing chat behavior.
            </p>
          </div>
        </header>

        <form className="incident-form" onSubmit={(event) => void onSubmit(event)}>
          <label className="field">
            <span>Incident ID</span>
            <Input
              value={incidentId}
              onChange={(event) => setIncidentId(event.target.value)}
              maxLength={120}
              placeholder="INC-1001"
              disabled={loading}
              autoComplete="off"
            />
          </label>
          <label className="field">
            <span>Description</span>
            <textarea
              className="incident-textarea"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              maxLength={20000}
              rows={8}
              placeholder="Symptoms, timeline, affected services…"
              disabled={loading}
            />
          </label>
          <div className="incident-actions">
            <Button type="submit" disabled={loading}>
              {loading ? (
                <>
                  <LoaderCircleIcon className="spin" />
                  Analyzing…
                </>
              ) : (
                <>
                  <ShieldAlertIcon />
                  Analyze incident
                </>
              )}
            </Button>
          </div>
        </form>

        {error ? (
          <div className="incident-error" role="alert">
            <p>{error}</p>
          </div>
        ) : null}

        {loading && !result ? (
          <div className="incident-loading" aria-live="polite">
            <LoaderCircleIcon className="spin" />
            Waiting for structured output…
          </div>
        ) : null}

        {analysis ? (
          <section className="incident-result" aria-label="Analysis result">
            <div className="incident-result-head">
              <div>
                <p className="eyebrow">Structured result</p>
                <h2>{analysis.incident_id}</h2>
              </div>
              <span className={cn("severity-pill", analysis.severity.toLowerCase())}>
                {analysis.severity} · {SEVERITY_LABEL[analysis.severity]}
              </span>
            </div>

            <dl className="incident-fields">
              <div>
                <dt>Summary</dt>
                <dd>{analysis.summary}</dd>
              </div>
              <div>
                <dt>Probable cause</dt>
                <dd>{analysis.probable_cause}</dd>
              </div>
              <div>
                <dt>Recommended actions</dt>
                <dd>
                  <ol>
                    {analysis.recommended_actions.map((action) => (
                      <li key={action}>{action}</li>
                    ))}
                  </ol>
                </dd>
              </div>
              <div>
                <dt>Human approval</dt>
                <dd>{analysis.requires_human_approval ? "Required before acting" : "Not required"}</dd>
              </div>
            </dl>

            {result ? (
              <p className="incident-meta">
                {result.model ?? "unknown model"}
                {" · "}
                {formatLatency(result.latency_ms)}
                {result.input_tokens != null ? ` · in ${formatCount(result.input_tokens)}` : ""}
                {result.output_tokens != null ? ` · out ${formatCount(result.output_tokens)}` : ""}
                {result.cached_input_tokens != null
                  ? ` · cached ${formatCount(result.cached_input_tokens)}`
                  : ""}
                {result.estimated_cost_usd != null
                  ? ` · ~${formatUsd(result.estimated_cost_usd)}`
                  : ""}
              </p>
            ) : null}
          </section>
        ) : null}
      </div>
    </main>
  );
}
