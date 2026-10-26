# Phase 2 completion

OperateSolve Phase 2 (cost, routing, caching, context, structured analysis, and a parallel-execution learning experiment) is complete. Phase 1 chat behavior is preserved.

## Features completed

| Phase | Feature |
| --- | --- |
| 2.1 | Usage & cost dashboard (`/usage`), `llm_usage` persistence, `pricing.json` estimates |
| 2.2 | Model registry, Auto/Luna/Mini/Nano selection, deterministic router, availability checks |
| 2.3 | Prompt-cache request settings, cache hit/savings on the dashboard, `cache_experiment.py` |
| 2.4 | Model-aware input budgets, sliding-window history, incremental summarization |
| 2.5A | Schema-enforced incident analysis (`/incidents`, `POST /api/incidents/analyze`) |
| 2.5B | Sequential vs parallel LLM experiment (`/experiments`, `POST /api/experiments/parallel`) |

## Architecture summary

```text
UI (Next.js) → FastAPI routes → services → LLM provider (OpenAI Responses)
                              ↘ SQLite (conversations, messages, summaries, llm_usage)
```

- **Chat path:** preference → `ModelRouter` → context window (+ optional summary) → streaming provider → one `llm_usage` row (`purpose=chat`).
- **Side paths:** summarization, incident Structured Outputs, and the parallel experiment each record their own `purpose` and never rewrite chat messages.
- **Cost:** token counts from the provider; USD from `pricing.json` on read. No second database.

## Test results (2026-10-26)

| Check | Result |
| --- | --- |
| Backend `pytest` | **140 passed** |
| Frontend `npm run typecheck` | Pass |
| Frontend `npm run lint` | Pass |
| Frontend `npm run build` | Pass (`/`, `/incidents`, `/experiments`, `/usage`) |

Live OpenAI calls are not part of automated CI; use the manual steps in the README for billed smoke checks.

## Known limitations

- Costs are estimates; OpenAI billing is authoritative. Standard tier only.
- Mini and Nano are deprecated (shutdown 2026-12-11); Auto Simple/Detailed fall back to Luna after that.
- Context budgeting uses a characters÷4 heuristic; provider-reported usage remains the cost source of truth.
- Incident analysis and the parallel experiment use the baseline `OPENAI_MODEL` only (no per-request model picker).
- Parallel experiment is a learning demo: three fixed tasks, no production workflow, no batch API.
- Parallelism reduces wall-clock latency for independent calls; total tokens and cost stay comparable.

## Recommended future improvements

1. **Intelligent model routing** — route by prompt complexity, latency budget, or cost target instead of (or in addition to) the Simple/General/Detailed switch.
2. **Batch / async jobs** — OpenAI Batch for offline incident backfills where latency does not matter.
3. **RAG / runbooks** — retrieve internal docs into the cache-friendly prompt prefix.
4. **Agent / tool use** — constrained tools for status pages, deploy history, and ticketing (with human approval gates).
5. **Auth and multi-tenant isolation** — operators, API keys, and per-tenant cache keys.
6. **Stronger evaluation** — golden incident sets and quality scores beside cost/latency.

Do not treat 2.5B as a production orchestration layer; keep experiments behind `/experiments` until product requirements are clear.
