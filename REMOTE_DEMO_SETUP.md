# OperateSolve remote demo (Cloudflare Tunnel)

Temporary, authenticated access from your Windows development PC. Only the Next.js UI is exposed. FastAPI, SQLite, ChromaDB, and `OPENAI_API_KEY` stay on localhost.

## Architecture

```text
Remote browser
  → Cloudflare (Access email OTP + Tunnel)
  → http://127.0.0.1:3000  (Next.js)
       ├─ UI pages
       └─ /api/*  →  http://127.0.0.1:8000/api/*  (server-side only)
            FastAPI + SQLite + Chroma + OpenAI key
```

- Browser uses **relative** `/api/*` URLs (no backend host in client env).
- Next.js **rewrites** forward most `/api/*` calls to FastAPI.
- Chat **SSE** send/retry use App Router route handlers (rewrites buffer event streams).
- Bind FastAPI to **`127.0.0.1:8000` only**. Do not publish port 8000.

## Prerequisites

1. OperateSolve runs locally (`start-dev.ps1` or manual uvicorn + `npm run dev`).
2. [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/) installed and on `PATH`.
3. Do **not** put tunnel tokens, credentials, or demo URLs in Git (see `.gitignore`).

## Start the apps (local)

```powershell
powershell -ExecutionPolicy Bypass -File .\start-dev.ps1
```

Confirm:

```powershell
# Backend (loopback only)
Invoke-RestMethod http://127.0.0.1:8000/api/health

# Same-origin proxy through Next.js
Invoke-RestMethod http://127.0.0.1:3000/api/health
```

Optional for demos: set `EXPOSE_API_DOCS=false` in `backend/.env` and restart the API so `/docs`, `/redoc`, and `/openapi.json` are disabled even on loopback.

## Cloudflare Quick Tunnel (temporary URL)

Tunnel **only** the frontend:

```powershell
cloudflared tunnel --url http://127.0.0.1:3000
```

Copy the `https://….trycloudflare.com` URL from the cloudflared output. Do not commit it.

`next.config.ts` already allows `*.trycloudflare.com` for `next dev`. For a different hostname, set server-only:

```powershell
# hostname only, no https:// and no port
$env:ALLOWED_DEV_ORIGINS="your-demo-host.example.com"
```

Then restart the frontend.

## Email authentication (Cloudflare Access)

Anyone with a Quick Tunnel URL can open the app unless you add Access.

**Recommended for a real demo:** use a **named** Cloudflare Tunnel on a hostname you control, then create a Cloudflare Zero Trust **Access Application** for that hostname with **One-time PIN (email)** (or your IdP). Access runs at the edge before traffic reaches Next.js, so UI pages and proxied `/api/*` are both protected.

Quick Tunnels alone do not give you a durable Access policy on a stable hostname. Treat a bare Quick Tunnel URL as a shared secret and prefer Access on a named tunnel when showing the app to others.

## Streaming / SSE limitations

| Layer | Chat streaming (`text/event-stream`) |
| --- | --- |
| Local browser → Next `/api` → FastAPI | Supported. SSE send/retry use streaming route handlers. |
| **Cloudflare Quick Tunnel** (`*.trycloudflare.com`) | **Not supported.** The edge buffers SSE; chat tokens may arrive all at once or stall. Non-streaming features (conversations list, usage, incidents, experiments, Knowledge Search/Ask) still work. |
| **Named Cloudflare Tunnel** | SSE is supported per Cloudflare docs. Use this if remote chat streaming matters. |

Do not switch OperateSolve to polling for the demo. Prefer a named tunnel when you need live token streaming remotely.

## What must stay private

| Asset | Exposure |
| --- | --- |
| Next.js `:3000` | Only public surface (via tunnel + Access) |
| FastAPI `:8000` | `127.0.0.1` only — never tunnel this port |
| `OPENAI_API_KEY` | `backend/.env` only |
| SQLite / Chroma | Local disk under `backend/data/` |
| `/docs`, `/redoc`, `/openapi.json` | Not under `/api`; not rewritten. Optionally disable with `EXPOSE_API_DOCS=false` |

Do not set `CORS_ORIGINS=*`. Do not add tunnel origins to CORS (browser calls are same-origin to Next.js). Do not put backend URLs in `NEXT_PUBLIC_*` variables.

## Stop

1. Stop `cloudflared` (Ctrl+C).
2. Close the API and frontend windows from `start-dev.ps1`.

## Local verification checklist

1. `Invoke-RestMethod http://127.0.0.1:3000/api/health` succeeds while the backend is up.
2. With the backend stopped, the same call fails (proxy depends on FastAPI).
3. Chat streaming still works at http://127.0.0.1:3000 (local, no tunnel).
4. Knowledge, usage, incidents, and experiments load via `/api/...` in the Network tab (same origin as the page).
