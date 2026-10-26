# Phase 3 Completion — Knowledge RAG

OperateSolve Phase 3 adds a local knowledge RAG stack over synthetic Northstar Commerce Markdown: ingest → chunk → embed → retrieve → grounded answer with citations. Phase 3.3 adds hybrid retrieval (semantic + BM25 + RRF) and evaluation against the Phase 3.1/3.2 baseline.

**Default retrieval mode remains `semantic`.** That recommendation is based on measured eval results below, not assumptions.

## End-to-end pipeline

```text
Source Markdown
  → load + content-hash
  → chunk (heading-aware, overlap)
  → OpenAI embeddings (text-embedding-3-small)
  → ChromaDB (cosine) + BM25 JSON sidecar (synced on ingest)
  → retrieve: semantic | hybrid (semantic ∥ BM25 → RRF → optional local rerank)
  → select within RAG context budget
  → one structured LLM call (GroundedAnswerPayload)
  → citation scrub / validate → UI
```

Ask Knowledge never uses extra LLM calls for retrieval or reranking.

## Ingestion, chunking, embeddings, storage

| Piece | Location | Notes |
| --- | --- | --- |
| Sources | `backend/data/knowledge/source/*.md` | 10 synthetic ops docs |
| Loader | `app/services/rag/loader.py` | Stable `document_id` = filename stem |
| Chunking | `app/services/rag/chunking.py` | Token estimate + overlap; section headings preserved |
| Embeddings | `app/services/rag/embeddings.py` | Batched OpenAI; usage purpose=`embedding` |
| Vector store | `app/services/rag/store.py` | Persistent Chroma, cosine space |
| BM25 index | `app/services/rag/bm25.py` | `rank-bm25` + JSON sidecar; rebuilt when Chroma mutates |
| Ingest CLI | `scripts/ingest_knowledge.py` | Idempotent via content hash; prunes deleted sources |

BM25 stays synchronized by rebuilding from Chroma after ingest upserts/deletes (and on API startup if counts drift).

## Semantic vs hybrid retrieval

| Mode | Behavior |
| --- | --- |
| `semantic` | Query embed → Chroma top_k (Phase 3.1/3.2 baseline) |
| `hybrid` | Parallel semantic candidates + BM25 candidates → Reciprocal Rank Fusion (default k=60) → top_k |

Score fields are kept distinct on each hit:

- **similarity / distance** — cosine from Chroma
- **bm25_score** — keyword relevance (or term-hit fallback when classic BM25 IDF collapses on tiny corpora)
- **rrf_score** — fusion rank score

Config (`backend/.env.example`):

- `RAG_RETRIEVAL_MODE` (default `semantic`)
- `RAG_SEMANTIC_CANDIDATES`, `RAG_BM25_CANDIDATES`
- `RAG_RRF_K` (default 60)
- `BM25_INDEX_PATH`
- `RAG_RERANKER_ENABLED` / `RAG_RERANKER`

API: `POST /api/knowledge/search` and `POST /api/knowledge/ask` accept optional `retrieval_mode`.

UI: http://localhost:3000/knowledge — Semantic / Hybrid selector; hits show method and scores.

## Fusion / reranking design

**RRF** merges ranked lists and dedupes by `chunk_id`. It is not a learned reranker.

**Reranker interface** (`app/services/rag/rerank.py`):

- `NoOpReranker` — default when disabled
- `LexicalOverlapReranker` — local Jaccard token overlap (query-dependent, no paid API, no heavy ML)

Reranking stays **disabled by default**. Lexical overlap is available for experiments (`RAG_RERANKER_ENABLED=true` or eval `--modes …,hybrid_rerank`). Do not claim quality wins without measuring.

## Grounded answers and citations

Flow in `app/services/rag/answer.py`:

1. Retrieve (mode from request or `RAG_RETRIEVAL_MODE`)
2. Select chunks within `RAG_CONTEXT_TOKEN_BUDGET` (`RAG_MIN_SIMILARITY` applies to semantic-participating hits only)
3. One Structured Outputs call → `GroundedAnswerPayload`
4. Scrub fabricated `[S#]` labels; return validated citations

Citation validation checks **reference integrity**, not factual grounding of every claim.

## Retrieval evaluation (measured)

Dataset: `backend/data/knowledge/eval/retrieval_eval.json` (`northstar-retrieval-v2`)

- Preserves original q01–q13
- Adds q14–q27 (incident IDs, acronyms, API paths, error codes, paraphrases, cross-doc, unanswerable, exact keywords)

Runner: `python -m scripts.eval_retrieval --modes semantic,hybrid,hybrid_rerank`

**Run conditions:** populated Chroma (91 chunks) + BM25 (91 chunks), `text-embedding-3-small`, k=5, identical queries/labels. Measured 2026-10-26.

| Mode | Recall@5 | MRR@5 | Precision@5* | Mean latency | Embedding tokens |
| --- | --- | --- | --- | --- | --- |
| semantic | **24/25 = 0.960** | **0.883** | **0.750** (n=4) | 256 ms | 367 (~$0.000007) |
| hybrid | 24/25 = 0.960 | 0.890 | 0.521 (n=4) | 243 ms | 367 (~$0.000007) |
| hybrid_rerank | **25/25 = 1.000** | 0.737 | 0.508 (n=4) | 251 ms | 367 (~$0.000007) |

\*Precision@K only for items with `relevant_document_ids` (4 questions). Most items lack graded relevance — do not over-read Precision.

Notable per-question outcomes:

- **q20** (OMS error-code enumeration): MISS for semantic and hybrid; HIT only with lexical rerank (MRR 0.33).
- Hybrid’s MRR edge over semantic is small (+0.007); labeled Precision is worse.
- Lexical rerank raises recall to 1.0 but **hurts MRR** (~0.15 drop) by promoting weak keyword overlap over stronger fused ranks.

### Default strategy recommendation

Keep **`RAG_RETRIEVAL_MODE=semantic`** as the server default until a stronger reranker or tuned candidate counts improve hybrid without degrading ranking quality. Use the UI **Hybrid** toggle when debugging exact IDs/error codes. Leave **`RAG_RERANKER_ENABLED=false`**.

## Answer evaluation

Runner: `python -m scripts.eval_rag_answer --modes semantic,hybrid [--retrieval-only]`

Live generation is optional (billed). Metrics:

- Retrieval Recall@5 / MRR@5
- Citation validity (reference integrity)
- Fact coverage via **substring** checks (`expected_answer_facts`) — **not** LLM-as-judge and **not** proof of factual accuracy
- Abstention on `unanswerable` items
- Generation token usage / estimated cost when live

Automated pytest does not call live OpenAI for these scripts.

## Tests and checks (Phase 3.3)

- Backend: **184 passed** (`pytest`), including `tests/test_rag_hybrid.py` (BM25, RRF, dedupe, mode switch, stale/missing index, reranker, API modes)
- Frontend: typecheck, lint, production build — passed
- No live OpenAI in automated tests

## Known limitations and production considerations

1. **Corpus size** — Demo knowledge base is small; classic BM25 IDF can collapse to zero; keyword hit-count fallback is used when scores are flat.
2. **Chunk granularity** — Current store has ~9 chunks/doc from prior ingest settings; re-ingest with `--force` after changing chunk size if you need larger chunks.
3. **Hybrid noise** — RRF can surface keyword-adjacent but less useful docs; Precision@5 on labeled items dropped vs semantic in this run.
4. **Lexical rerank** — Weak signal; improves recall here at the cost of MRR. Not a cross-encoder.
5. **No chat memory RAG** — Knowledge ask is single-shot; not wired into conversational chat history.
6. **No agents / MCP / LangChain** — Intentional for Phase 3.
7. **Citation ≠ grounding** — Valid `[S#]` does not prove every sentence is supported.
8. **Costs** — Each search/ask embeds the query; ask adds one generation call. Prefer `--retrieval-only` for routine eval.

## Manual verification

```powershell
cd backend
.\.venv\Scripts\python.exe -m scripts.ingest_knowledge
.\.venv\Scripts\python.exe -m scripts.eval_retrieval --modes semantic,hybrid,hybrid_rerank
.\.venv\Scripts\python.exe -m scripts.eval_rag_answer --retrieval-only --modes semantic,hybrid
# Optional live answers (billed):
.\.venv\Scripts\python.exe -m scripts.eval_rag_answer --modes semantic,hybrid
```

UI: open http://localhost:3000/knowledge, toggle Semantic vs Hybrid, run Search and Ask on `INC-2407` and a paraphrase like “checkout failing because the DB pool looks full”.
