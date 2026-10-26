"use client";

import { useState, type FormEvent } from "react";
import Link from "next/link";
import { ArrowLeftIcon, LoaderCircleIcon, TimerIcon, WorkflowIcon } from "lucide-react";
import { Button } from "@/components/ui/button";
import { isAbortError, runParallelExperiment } from "@/lib/api";
import { formatCount, formatLatency, formatUsd } from "@/lib/format";
import type { ExperimentMode, ParallelExperimentResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

const DEFAULT_INCIDENT =
  "Payments p95 latency rose from 120ms to 4.5s after the 14:00 deploy. Checkout errors are elevated in us-east-1; auth and catalog look healthy. On-call suspects a connection-pool change in the payments service.";

type RunState = {
  sequential: ParallelExperimentResponse | null;
  parallel: ParallelExperimentResponse | null;
};

export function ParallelExperimentPanel() {
  const [description, setDescription] = useState(DEFAULT_INCIDENT);
  const [running, setRunning] = useState<ExperimentMode | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [runs, setRuns] = useState<RunState>({ sequential: null, parallel: null });
  const [activeMode, setActiveMode] = useState<ExperimentMode | null>(null);

  async function runMode(mode: ExperimentMode) {
    const text = description.trim();
    if (!text) {
      setError("Incident description is required.");
      return;
    }
    const controller = new AbortController();
    setRunning(mode);
    setError(null);
    setActiveMode(mode);
    try {
      const next = await runParallelExperiment(text, mode, controller.signal);
      setRuns((prev) => ({ ...prev, [mode]: next }));
    } catch (reason: unknown) {
      if (isAbortError(reason)) return;
      setError(reason instanceof Error ? reason.message : "Could not run the experiment.");
    } finally {
      setRunning(null);
    }
  }

  function onSubmit(event: FormEvent) {
    event.preventDefault();
  }

  const sequential = runs.sequential;
  const parallel = runs.parallel;
  const active = activeMode ? runs[activeMode] : null;
  const hasComparison = sequential != null || parallel != null;

  return (
    <main className="incident-page experiment-page">
      <div className="incident-inner experiment-inner">
        <Link href="/" className="back-link">
          <ArrowLeftIcon />
          Back to chats
        </Link>

        <header className="incident-head">
          <div>
            <p className="eyebrow">Phase 2.5B</p>
            <h1>LLM Experiments</h1>
            <p className="incident-lede">
              Run three independent incident tasks sequentially or in parallel. Parallel mode can reduce
              wall-clock latency; token usage and estimated cost stay similar because each task still makes
              its own billed call.
            </p>
          </div>
        </header>

        <form className="incident-form" onSubmit={onSubmit}>
          <label className="field">
            <span>Fictional incident description</span>
            <textarea
              className="incident-textarea"
              value={description}
              onChange={(event) => setDescription(event.target.value)}
              maxLength={20000}
              rows={7}
              placeholder="Symptoms, timeline, affected services…"
              disabled={running != null}
            />
          </label>
          <div className="experiment-actions">
            <Button
              type="button"
              variant="outline"
              disabled={running != null}
              onClick={() => void runMode("sequential")}
            >
              {running === "sequential" ? (
                <>
                  <LoaderCircleIcon className="spin" />
                  Running sequential…
                </>
              ) : (
                <>
                  <TimerIcon />
                  Run sequential
                </>
              )}
            </Button>
            <Button type="button" disabled={running != null} onClick={() => void runMode("parallel")}>
              {running === "parallel" ? (
                <>
                  <LoaderCircleIcon className="spin" />
                  Running parallel…
                </>
              ) : (
                <>
                  <WorkflowIcon />
                  Run parallel
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

        {hasComparison ? (
          <section className="incident-result experiment-compare" aria-label="Performance comparison">
            <div className="incident-result-head">
              <div>
                <p className="eyebrow">Comparison</p>
                <h2>Sequential vs parallel</h2>
                <p className="experiment-note">
                  Each column is a separate execution run. Parallelism does not reduce token cost — it only
                  overlaps independent calls.
                </p>
              </div>
            </div>
            <div className="experiment-table-wrap">
              <table className="experiment-table">
                <thead>
                  <tr>
                    <th>Metric</th>
                    <th>Sequential</th>
                    <th>Parallel</th>
                  </tr>
                </thead>
                <tbody>
                  <tr>
                    <td>Total latency</td>
                    <td>{sequential ? formatLatency(sequential.totals.latency_ms) : "—"}</td>
                    <td>{parallel ? formatLatency(parallel.totals.latency_ms) : "—"}</td>
                  </tr>
                  <tr>
                    <td>Input tokens</td>
                    <td>{sequential ? formatCount(sequential.totals.input_tokens) : "—"}</td>
                    <td>{parallel ? formatCount(parallel.totals.input_tokens) : "—"}</td>
                  </tr>
                  <tr>
                    <td>Output tokens</td>
                    <td>{sequential ? formatCount(sequential.totals.output_tokens) : "—"}</td>
                    <td>{parallel ? formatCount(parallel.totals.output_tokens) : "—"}</td>
                  </tr>
                  <tr>
                    <td>Estimated cost</td>
                    <td>{sequential ? formatUsd(sequential.totals.estimated_cost_usd) : "—"}</td>
                    <td>{parallel ? formatUsd(parallel.totals.estimated_cost_usd) : "—"}</td>
                  </tr>
                  <tr>
                    <td>Successful tasks</td>
                    <td>
                      {sequential
                        ? `${sequential.totals.successful_tasks} / ${sequential.tasks.length}`
                        : "—"}
                    </td>
                    <td>
                      {parallel ? `${parallel.totals.successful_tasks} / ${parallel.tasks.length}` : "—"}
                    </td>
                  </tr>
                </tbody>
              </table>
            </div>
          </section>
        ) : null}

        {active ? (
          <section className="incident-result" aria-label={`${active.mode} run results`}>
            <div className="incident-result-head">
              <div>
                <p className="eyebrow">
                  {active.mode === "sequential" ? "Sequential run" : "Parallel run"}
                </p>
                <h2>Task results</h2>
                <p className="incident-meta">
                  model {active.model} · total {formatLatency(active.totals.latency_ms)} ·{" "}
                  {formatCount(active.totals.input_tokens)} in / {formatCount(active.totals.output_tokens)}{" "}
                  out · {formatUsd(active.totals.estimated_cost_usd)}
                </p>
              </div>
              {sequential && parallel ? (
                <div className="experiment-mode-toggle" role="group" aria-label="Show run results">
                  <button
                    type="button"
                    className={cn("experiment-mode-btn", activeMode === "sequential" && "active")}
                    onClick={() => setActiveMode("sequential")}
                  >
                    Sequential
                  </button>
                  <button
                    type="button"
                    className={cn("experiment-mode-btn", activeMode === "parallel" && "active")}
                    onClick={() => setActiveMode("parallel")}
                  >
                    Parallel
                  </button>
                </div>
              ) : null}
            </div>

            <div className="experiment-tasks">
              {active.tasks.map((task) => (
                <article key={`${active.mode}-${task.task_id}`} className="experiment-task">
                  <header className="experiment-task-head">
                    <h3>{task.task_name}</h3>
                    <span className={cn("status-pill", task.status)}>{task.status}</span>
                  </header>
                  {task.status === "complete" ? (
                    <p className="experiment-task-body">{task.result}</p>
                  ) : (
                    <p className="experiment-task-error">{task.error ?? "Task failed."}</p>
                  )}
                  <p className="incident-meta">
                    {formatLatency(task.latency_ms)} · {formatCount(task.input_tokens)} in /{" "}
                    {formatCount(task.output_tokens)} out · {formatUsd(task.estimated_cost_usd)}
                  </p>
                </article>
              ))}
            </div>
          </section>
        ) : running ? (
          <div className="incident-loading">
            <LoaderCircleIcon className="spin" />
            Running {running} experiment…
          </div>
        ) : null}
      </div>
    </main>
  );
}
