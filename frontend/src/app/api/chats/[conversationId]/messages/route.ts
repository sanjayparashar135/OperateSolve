import type { NextRequest } from "next/server";
import { proxyToBackend } from "@/lib/backend-proxy";

type RouteContext = { params: Promise<{ conversationId: string }> };

/** Streaming chat send — App Router proxy (rewrites buffer SSE). */
export async function POST(request: NextRequest, context: RouteContext) {
  const { conversationId } = await context.params;
  return proxyToBackend(request, `/api/chats/${conversationId}/messages`);
}
