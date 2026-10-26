import type { NextRequest } from "next/server";
import { proxyToBackend } from "@/lib/backend-proxy";

type RouteContext = { params: Promise<{ conversationId: string; messageId: string }> };

/** Streaming chat retry — App Router proxy (rewrites buffer SSE). */
export async function POST(request: NextRequest, context: RouteContext) {
  const { conversationId, messageId } = await context.params;
  return proxyToBackend(request, `/api/chats/${conversationId}/messages/${messageId}/retry`);
}
