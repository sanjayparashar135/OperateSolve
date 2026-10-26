import type { NextConfig } from "next";

/**
 * Hostname-only origins allowed to reach the Next.js dev server (no scheme/port).
 * Override with ALLOWED_DEV_ORIGINS (comma-separated) for a temporary remote hostname.
 * Do not use wildcards that accept arbitrary sites.
 */
function allowedDevOrigins(): string[] {
  const fromEnv = (process.env.ALLOWED_DEV_ORIGINS ?? "")
    .split(",")
    .map((value) => value.trim())
    .filter(Boolean);
  // Cloudflare Quick Tunnel hostnames: https://*.trycloudflare.com
  const defaults = ["*.trycloudflare.com"];
  return [...new Set([...defaults, ...fromEnv])];
}

const backendOrigin = (process.env.BACKEND_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

const nextConfig: NextConfig = {
  cacheComponents: true,
  partialPrefetching: true,
  allowedDevOrigins: allowedDevOrigins(),
  async rewrites() {
    // Forward browser /api/* to local FastAPI. Query strings are preserved by Next.js.
    // Chat SSE endpoints are handled by App Router route handlers instead (rewrites buffer streams).
    return [
      {
        source: "/api/:path*",
        destination: `${backendOrigin}/api/:path*`,
      },
    ];
  },
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
};

export default nextConfig;
