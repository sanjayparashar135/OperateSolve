import type { Conversation } from "@/lib/types";

export function formatLatency(ms: number | null): string {
  if (ms == null) return "—";
  if (ms < 1000) return `${ms} ms`;
  const seconds = ms / 1000;
  return `${seconds.toFixed(seconds < 10 ? 1 : 0)} s`;
}

export function formatCount(value: number | null): string {
  if (value == null) return "—";
  return new Intl.NumberFormat().format(value);
}

export function formatCompact(value: number | null): string {
  if (value == null) return "—";
  if (Math.abs(value) < 10_000) return new Intl.NumberFormat().format(value);
  return new Intl.NumberFormat(undefined, { notation: "compact", maximumFractionDigits: 1 }).format(value);
}

export function formatUsd(value: number | null): string {
  if (value == null) return "—";
  if (value === 0) return "$0.00";
  const abs = Math.abs(value);
  const digits = abs >= 1 ? 2 : abs >= 0.01 ? 4 : 6;
  if (abs < 0.000001) return "< $0.000001";
  return new Intl.NumberFormat(undefined, {
    style: "currency",
    currency: "USD",
    minimumFractionDigits: 2,
    maximumFractionDigits: digits,
  }).format(value);
}

export function formatRate(value: number | null): string {
  if (value == null) return "input rate";
  return `$${value.toLocaleString(undefined, { maximumFractionDigits: 4 })}`;
}

/** True when the provider answered with the requested model or one of its dated snapshots. */
export function isSameModel(requested: string | null, actual: string | null): boolean {
  if (!requested || !actual) return requested === actual;
  return actual === requested || new RegExp(`^${requested.replace(/[.]/g, "\\.")}-\\d{4}-\\d{2}-\\d{2}$`).test(actual);
}

export function formatDateTime(iso: string | null): string {
  if (!iso) return "—";
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "—";
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(date);
}

export function formatDay(isoDate: string): string {
  const date = new Date(`${isoDate}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return isoDate;
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", timeZone: "UTC" }).format(date);
}

export function formatWhen(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return "";
  const now = new Date();
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const startYesterday = startToday - 86_400_000;
  if (date.getTime() >= startToday) {
    return new Intl.DateTimeFormat(undefined, { hour: "numeric", minute: "2-digit" }).format(date);
  }
  if (date.getTime() >= startYesterday) return "Yesterday";
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric" }).format(date);
}

export function groupConversations(items: Conversation[]): Array<{ label: string; items: Conversation[] }> {
  const now = new Date();
  const startToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const startYesterday = startToday - 86_400_000;
  const groups: Record<"Today" | "Yesterday" | "Earlier", Conversation[]> = {
    Today: [],
    Yesterday: [],
    Earlier: [],
  };
  for (const item of items) {
    const time = new Date(item.updated_at).getTime();
    if (time >= startToday) groups.Today.push(item);
    else if (time >= startYesterday) groups.Yesterday.push(item);
    else groups.Earlier.push(item);
  }
  return (["Today", "Yesterday", "Earlier"] as const)
    .filter((label) => groups[label].length > 0)
    .map((label) => ({ label, items: groups[label] }));
}

export function upsertConversation(items: Conversation[], conversation: Conversation): Conversation[] {
  const next = items.filter((item) => item.id !== conversation.id);
  next.unshift(conversation);
  return next;
}
