import type { Message } from "@/lib/types";
import { formatCount, formatLatency } from "@/lib/format";
import { cn } from "@/lib/utils";

const ROUTE_LABELS = { manual: "manual", auto: "auto", fallback: "auto · fallback" } as const;

export function Metrics({ message }: { message: Message }) {
  if (message.role !== "assistant" || message.status === "streaming") return null;
  const usage =
    message.input_tokens == null && message.output_tokens == null
      ? "usage unavailable"
      : `${formatCount(message.input_tokens)} in${
          message.cached_input_tokens ? ` (${formatCount(message.cached_input_tokens)} cached)` : ""
        } · ${formatCount(message.output_tokens)} out`;
  const route = message.routing_mode ? ROUTE_LABELS[message.routing_mode] : null;

  return (
    <p className="metrics" title={message.request_id ? `Request ${message.request_id}` : undefined}>
      {route ? (
        <span
          className={cn("route-tag", message.routing_mode === "fallback" && "fallback")}
          title={message.routing_reason ?? undefined}
        >
          {route}
        </span>
      ) : null}
      <span>{message.model ?? "model unavailable"}</span>
      <span aria-hidden="true">·</span>
      <span>{formatLatency(message.latency_ms)}</span>
      <span aria-hidden="true">·</span>
      <span>{usage}</span>
    </p>
  );
}
