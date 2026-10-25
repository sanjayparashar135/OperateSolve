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
