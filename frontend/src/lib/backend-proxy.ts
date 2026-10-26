import type { NextRequest } from "next/server";

const BACKEND_ORIGIN = (process.env.BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

/** Hop-by-hop / encoding headers must not be forwarded when proxying a stream. */
const DROP_REQUEST_HEADERS = new Set([
  "connection",
  "content-length",
  "host",
  "keep-alive",
  "proxy-authenticate",
  "proxy-authorization",
  "te",
  "trailer",
  "transfer-encoding",
  "upgrade",
]);

const DROP_RESPONSE_HEADERS = new Set([
  "connection",
  "content-encoding",
  "content-length",
  "keep-alive",
  "transfer-encoding",
]);

function backendUrl(pathWithQuery: string): string {
  const path = pathWithQuery.startsWith("/") ? pathWithQuery : `/${pathWithQuery}`;
  return `${BACKEND_ORIGIN}${path}`;
}

function forwardRequestHeaders(request: NextRequest): Headers {
  const headers = new Headers();
  request.headers.forEach((value, key) => {
    if (!DROP_REQUEST_HEADERS.has(key.toLowerCase())) {
      headers.set(key, value);
    }
  });
  return headers;
}

function forwardResponseHeaders(upstream: Response): Headers {
  const headers = new Headers();
  upstream.headers.forEach((value, key) => {
    if (!DROP_RESPONSE_HEADERS.has(key.toLowerCase())) {
      headers.set(key, value);
    }
  });
  // Encourage intermediaries not to buffer SSE.
  if ((upstream.headers.get("content-type") ?? "").includes("text/event-stream")) {
    headers.set("Cache-Control", "no-cache, no-transform");
    headers.set("X-Accel-Buffering", "no");
  }
  return headers;
}

/**
 * Proxy a request to the local FastAPI process, streaming the body when present.
 * Used for chat SSE so responses are not buffered the way Next.js rewrites are.
 */
export async function proxyToBackend(request: NextRequest, apiPath: string): Promise<Response> {
  const search = request.nextUrl.search;
  const url = backendUrl(`${apiPath}${search}`);
  const method = request.method.toUpperCase();
  const hasBody = method !== "GET" && method !== "HEAD";

  const init: RequestInit & { duplex?: "half" } = {
    method,
    headers: forwardRequestHeaders(request),
    body: hasBody ? request.body : undefined,
    cache: "no-store",
    signal: request.signal,
  };
  if (hasBody) {
    init.duplex = "half";
  }
  const upstream = await fetch(url, init);

  return new Response(upstream.body, {
    status: upstream.status,
    statusText: upstream.statusText,
    headers: forwardResponseHeaders(upstream),
  });
}
