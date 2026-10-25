import type {
  Conversation,
  ConversationDetail,
  Health,
  Message,
  StreamHandlers,
} from "@/lib/types";

export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://127.0.0.1:8000";

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

export function isAbortError(error: unknown): boolean {
  return error instanceof Error && error.name === "AbortError";
}

export function listChats(signal?: AbortSignal): Promise<Conversation[]> {
  return request<Conversation[]>("/api/chats", { signal });
}

export function createChat(): Promise<Conversation> {
  return request<Conversation>("/api/chats", { method: "POST", body: JSON.stringify({}) });
}

export function getChat(id: string, signal?: AbortSignal): Promise<ConversationDetail> {
  return request<ConversationDetail>(`/api/chats/${id}`, { signal });
}

export function renameChat(id: string, title: string): Promise<Conversation> {
  return request<Conversation>(`/api/chats/${id}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

export function deleteChat(id: string): Promise<void> {
  return request<void>(`/api/chats/${id}`, { method: "DELETE" });
}

export function getHealth(signal?: AbortSignal): Promise<Health> {
  return request<Health>("/api/health", { signal });
}

export async function streamMessage(
  conversationId: string,
  content: string,
  clientRequestId: string,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<boolean> {
  return streamRequest(
    `/api/chats/${conversationId}/messages`,
    { content, client_request_id: clientRequestId },
    handlers,
    signal,
  );
}

export async function retryMessage(
  conversationId: string,
  messageId: string,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<boolean> {
  return streamRequest(
    `/api/chats/${conversationId}/messages/${messageId}/retry`,
    undefined,
    handlers,
    signal,
  );
}

async function streamRequest(
  path: string,
  body: unknown,
  handlers: StreamHandlers,
  signal: AbortSignal,
): Promise<boolean> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  });
  if (!response.ok) {
    throw new ApiError(await readError(response), response.status);
  }
  if (!response.body) {
    throw new ApiError("The API returned an empty stream.", response.status);
  }
  return readEventStream(response, handlers);
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      Accept: "application/json",
      ...(init?.body ? { "Content-Type": "application/json" } : {}),
      ...init?.headers,
    },
  });
  if (!response.ok) {
    throw new ApiError(await readError(response), response.status);
  }
  if (response.status === 204) {
    return undefined as T;
  }
  return (await response.json()) as T;
}

async function readError(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { detail?: unknown };
    if (typeof body.detail === "string") return body.detail;
    if (Array.isArray(body.detail)) {
      const text = body.detail
        .map((item) => (item && typeof item === "object" && "msg" in item ? String(item.msg) : ""))
        .filter(Boolean)
        .join(" ");
      if (text) return text;
    }
  } catch {
    // The body was not JSON.
  }
  return `Request failed (${response.status}).`;
}

async function readEventStream(response: Response, handlers: StreamHandlers): Promise<boolean> {
  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let terminal = false;

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const blocks = buffer.split("\n\n");
    buffer = blocks.pop() ?? "";
    for (const block of blocks) {
      const data = block
        .split("\n")
        .filter((line) => line.startsWith("data: "))
        .map((line) => line.slice(6))
        .join("\n");
      if (!data) continue;
      const event: unknown = JSON.parse(data);
      if (!event || typeof event !== "object" || !("type" in event)) continue;
      const record = event as Record<string, unknown>;
      if (record.type === "meta") {
        handlers.onMeta(event as Parameters<StreamHandlers["onMeta"]>[0]);
      } else if (record.type === "delta" && typeof record.text === "string") {
        handlers.onDelta(record.text);
      } else if (record.type === "done" && record.message && typeof record.message === "object") {
        terminal = true;
        handlers.onDone(record.message as Message);
      } else if (record.type === "error" && record.message && typeof record.message === "object") {
        terminal = true;
        handlers.onError({
          error: typeof record.error === "string" ? record.error : "The response failed.",
          message: record.message as Message,
        });
      }
    }
  }

  return terminal;
}
