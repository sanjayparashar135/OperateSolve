import type { Message } from "@/lib/types";
import { formatCount, formatLatency } from "@/lib/format";

export function Metrics({ message }: { message: Message }) {
  if (message.role !== "assistant" || message.status === "streaming") return null;
  const usage =
    message.input_tokens == null && message.output_tokens == null
      ? "usage unavailable"
      : `${formatCount(message.input_tokens)} in · ${formatCount(message.output_tokens)} out`;

  return (
    <p className="metrics" title={message.request_id ? `Request ${message.request_id}` : undefined}>
      <span>{message.model ?? "model unavailable"}</span>
      <span aria-hidden="true">·</span>
      <span>{formatLatency(message.latency_ms)}</span>
      <span aria-hidden="true">·</span>
      <span>{usage}</span>
    </p>
  );
}
