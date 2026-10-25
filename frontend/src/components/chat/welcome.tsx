import { ArrowUpRightIcon } from "lucide-react";

export const SUGGESTIONS = [
  {
    eyebrow: "Incident",
    title: "Triage a checkout failure",
    detail: "Sequence the first checks and name the owners.",
    prompt:
      "Latency on the checkout API jumped after the 02:10 deploy. Error rate is 4%, payments are timing out, and the last change touched the inventory client. Give me a triage sequence, the signals to confirm, and a first customer status update.",
  },
  {
    eyebrow: "Change",
    title: "Review a deploy window",
    detail: "Turn a change into a go/no-go checklist.",
    prompt:
      "We are deploying the payments service tomorrow at 06:00 UTC for 30 minutes. It includes a schema migration and a feature flag. Draft a change-window checklist covering pre-checks, rollout, abort criteria, and who communicates status.",
  },
  {
    eyebrow: "Runbook",
    title: "Find the gaps",
    detail: "Compare the steps with what on-call actually needs.",
    prompt:
      "Review this runbook outline for a database failover: page the DBA, promote the replica, update the connection string, and watch error rate. What is missing before this is safe for a new on-call operator?",
  },
  {
    eyebrow: "Status",
    title: "Write the update",
    detail: "Make an internal note safe for customers.",
    prompt:
      "Turn these notes into a customer-ready status update: investigating elevated checkout errors since 02:14 UTC, payments delayed, no evidence of data loss, next update in 30 minutes. Keep it calm, specific, and short.",
  },
] as const;

export function Welcome({
  disabled,
  onPrompt,
}: {
  disabled: boolean;
  onPrompt: (prompt: string) => void;
}) {
  return (
    <section className="welcome">
      <p className="eyebrow">Operations copilot</p>
      <h1>What needs to be solved?</h1>
      <p className="lede">
        Triage incidents, pressure-test a change, and draft the update your stakeholders are waiting on.
      </p>
      <div className="suggestion-grid">
        {SUGGESTIONS.map((item) => (
          <button
            key={item.title}
            type="button"
            className="suggestion"
            disabled={disabled}
            onClick={() => onPrompt(item.prompt)}
          >
            <span className="eyebrow">{item.eyebrow}</span>
            <span className="suggestion-title">
              {item.title}
              <ArrowUpRightIcon />
            </span>
            <span className="suggestion-detail">{item.detail}</span>
          </button>
        ))}
      </div>
    </section>
  );
}
