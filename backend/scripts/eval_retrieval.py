"""Compare semantic vs hybrid (and optional rerank) retrieval quality.

Requires a populated Chroma collection and BM25 index (run ingest_knowledge first).
Makes one embedding API call per question per mode (shared embedder; billed).

    .\\.venv\\Scripts\\python.exe -m scripts.eval_retrieval
    .\\.venv\\Scripts\\python.exe -m scripts.eval_retrieval --modes semantic,hybrid --json
    .\\.venv\\Scripts\\python.exe -m scripts.eval_retrieval --modes semantic,hybrid,hybrid_rerank

Reports only measured outcomes; it does not invent scores.
Precision@K is reported only for items with relevant_document_ids.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from app.config import BACKEND_DIR, get_settings
from app.services.pricing import estimate_cost, load_pricing
from app.services.rag.factory import build_bm25_index, build_vector_store, resolve_path
from app.services.rag.embeddings import OpenAIEmbeddingProvider
from app.services.rag.rerank import LexicalOverlapReranker, NoOpReranker
from app.services.rag.retrieve import RetrievalError, retrieve

DEFAULT_EVAL = BACKEND_DIR / "data" / "knowledge" / "eval" / "retrieval_eval.json"
MODE_CHOICES = ("semantic", "hybrid", "hybrid_rerank")


def recall_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Binary recall: 1.0 if any expected document appears in the top k retrieved docs."""

    if not expected_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    return 1.0 if top.intersection(expected_ids) else 0.0


def mrr_at_k(retrieved_ids: list[str], expected_ids: list[str], k: int) -> float:
    """Mean reciprocal rank of the first expected document within top k (0 if none)."""

    if not expected_ids:
        return 0.0
    expected = set(expected_ids)
    for index, doc_id in enumerate(retrieved_ids[:k], start=1):
        if doc_id in expected:
            return 1.0 / index
    return 0.0


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float | None:
    """Fraction of top-k retrieved docs that are labeled relevant. None if no labels."""

    if not relevant_ids:
        return None
    relevant = set(relevant_ids)
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    hits = sum(1 for doc_id in top if doc_id in relevant)
    return hits / len(top)


def _estimate_embedding_cost(settings, tokens: int | None) -> float | None:
    if tokens is None or tokens <= 0:
        return None
    try:
        pricing = load_pricing(settings.pricing_file)
    except Exception:
        return None
    model_pricing = pricing.find(settings.embedding_model)
    if model_pricing is None:
        return None
    return float(
        estimate_cost(
            model_pricing,
            input_tokens=tokens,
            output_tokens=0,
        ).total
    )


async def _run_mode(
    *,
    mode_label: str,
    questions: list[dict],
    k: int,
    store,
    embedder,
    bm25_index,
    settings,
) -> dict:
    if mode_label == "semantic":
        retrieval_mode = "semantic"
        reranker = NoOpReranker()
    elif mode_label == "hybrid":
        retrieval_mode = "hybrid"
        reranker = NoOpReranker()
    elif mode_label == "hybrid_rerank":
        retrieval_mode = "hybrid"
        reranker = LexicalOverlapReranker()
    else:
        raise ValueError(f"Unknown mode: {mode_label}")

    rows = []
    recall_hits = 0
    recall_scored = 0
    mrr_scores: list[float] = []
    precision_scores: list[float] = []
    latencies: list[int] = []
    embed_tokens_total = 0
    embed_tokens_known = False

    for item in questions:
        query = item["query"]
        expected = list(item.get("expected_document_ids") or [])
        relevant = list(item.get("relevant_document_ids") or [])
        unanswerable = bool(item.get("unanswerable"))
        row: dict = {
            "id": item.get("id"),
            "query": query,
            "category": item.get("category"),
            "expected_document_ids": expected,
            "unanswerable": unanswerable,
        }

        if unanswerable or not expected:
            row.update(
                {
                    "retrieved_document_ids": [],
                    "recall_hit": None,
                    "mrr": None,
                    "precision": None,
                    "skipped": True,
                    "reason": "unanswerable_or_no_expected_docs",
                    "note": "Recall/MRR not scored for unanswerable items.",
                }
            )
            rows.append(row)
            continue

        try:
            started = time.perf_counter()
            result = await retrieve(
                query,
                store=store,
                embedder=embedder,
                bm25_index=bm25_index,
                mode=retrieval_mode,
                top_k=k,
                semantic_candidates=settings.rag_semantic_candidates,
                bm25_candidates=settings.rag_bm25_candidates,
                rrf_k=settings.rag_rrf_k,
                reranker=reranker,
            )
            wall_ms = int((time.perf_counter() - started) * 1000)
        except RetrievalError as exc:
            recall_scored += 1
            row.update(
                {
                    "retrieved_document_ids": [],
                    "recall_hit": False,
                    "mrr": 0.0,
                    "precision": None,
                    "error": exc.message,
                    "latency_ms": None,
                }
            )
            rows.append(row)
            continue

        retrieved_docs: list[str] = []
        for hit in result.hits:
            if hit.document_id not in retrieved_docs:
                retrieved_docs.append(hit.document_id)

        recall = recall_at_k(retrieved_docs, expected, k)
        mrr = mrr_at_k(retrieved_docs, expected, k)
        precision = precision_at_k(retrieved_docs, relevant, k) if relevant else None

        recall_hits += int(recall)
        recall_scored += 1
        mrr_scores.append(mrr)
        if precision is not None:
            precision_scores.append(precision)
        latencies.append(result.latency_ms or wall_ms)
        if result.embedding_tokens is not None:
            embed_tokens_total += result.embedding_tokens
            embed_tokens_known = True

        row.update(
            {
                "retrieved_document_ids": retrieved_docs,
                "top_chunk_ids": [h.chunk_id for h in result.hits],
                "recall_hit": bool(recall),
                "mrr": mrr,
                "precision": precision,
                "precision_note": None
                if relevant
                else "Precision@K not calculated (no relevant_document_ids labels).",
                "latency_ms": result.latency_ms or wall_ms,
                "retrieval_mode": result.mode,
                "reranker": result.reranker,
                "embedding_tokens": result.embedding_tokens,
            }
        )
        rows.append(row)

    embed_cost = _estimate_embedding_cost(
        settings, embed_tokens_total if embed_tokens_known else None
    )
    return {
        "mode": mode_label,
        "k": k,
        "questions_scored": recall_scored,
        "metrics": {
            "recall_at_k": {
                "value": (recall_hits / recall_scored) if recall_scored else 0.0,
                "hits": recall_hits,
                "total": recall_scored,
                "kind": "automated",
            },
            "mrr_at_k": {
                "value": (sum(mrr_scores) / len(mrr_scores)) if mrr_scores else 0.0,
                "scored": len(mrr_scores),
                "kind": "automated",
            },
            "precision_at_k": {
                "value": (sum(precision_scores) / len(precision_scores)) if precision_scores else None,
                "scored": len(precision_scores),
                "kind": "automated",
                "note": (
                    "Only items with relevant_document_ids are included. "
                    "Most questions lack graded relevance labels."
                ),
            },
            "latency_ms": {
                "mean": (sum(latencies) / len(latencies)) if latencies else None,
                "p50": sorted(latencies)[len(latencies) // 2] if latencies else None,
                "samples": len(latencies),
                "kind": "measured",
            },
            "embedding_tokens": {
                "total": embed_tokens_total if embed_tokens_known else None,
                "estimated_cost_usd": embed_cost,
                "kind": "provider_reported" if embed_tokens_known else "unavailable",
            },
        },
        "results": rows,
    }


async def run_eval(*, eval_path: Path, k: int, modes: list[str]) -> dict:
    settings = get_settings()
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    questions = payload.get("questions") or []
    store = build_vector_store(settings)
    bm25_index = build_bm25_index(settings)
    embedder = OpenAIEmbeddingProvider(settings, repository=None)

    if store.count() == 0:
        raise SystemExit(
            "Vector store is empty. Ingest documents first:\n"
            "  .\\.venv\\Scripts\\python.exe -m scripts.ingest_knowledge"
        )
    if bm25_index.size == 0 and store.count() > 0:
        bm25_index.rebuild_from_store(store)
    elif bm25_index.size != store.count():
        bm25_index.rebuild_from_store(store)

    mode_reports = []
    for mode in modes:
        mode_reports.append(
            await _run_mode(
                mode_label=mode,
                questions=questions,
                k=k,
                store=store,
                embedder=embedder,
                bm25_index=bm25_index,
                settings=settings,
            )
        )

    return {
        "dataset": payload.get("name"),
        "eval_path": str(eval_path),
        "k": k,
        "embedding_model": settings.embedding_model,
        "store_chunks": store.count(),
        "bm25_chunks": bm25_index.size,
        "modes": mode_reports,
        "comparison_note": (
            "Compare modes under identical queries and labels. "
            "Default RAG_RETRIEVAL_MODE should stay semantic until hybrid wins on measured metrics."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate knowledge retrieval (semantic vs hybrid).")
    parser.add_argument("--eval", type=str, default=str(DEFAULT_EVAL), help="Path to eval JSON.")
    parser.add_argument("--k", type=int, default=5, help="Cutoff for Recall/MRR/Precision@K.")
    parser.add_argument(
        "--modes",
        type=str,
        default="semantic,hybrid",
        help="Comma-separated: semantic,hybrid,hybrid_rerank",
    )
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    args = parser.parse_args(argv)
    eval_path = resolve_path(args.eval) if not Path(args.eval).is_absolute() else Path(args.eval)
    if not eval_path.is_file():
        print(f"Eval file not found: {eval_path}", file=sys.stderr)
        return 1

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        if mode not in MODE_CHOICES:
            print(f"Unknown mode {mode!r}. Choose from {MODE_CHOICES}", file=sys.stderr)
            return 1

    report = asyncio.run(run_eval(eval_path=eval_path, k=args.k, modes=modes))
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"Dataset: {report['dataset']}")
    print(f"Store chunks: {report['store_chunks']} · BM25 chunks: {report['bm25_chunks']}")
    for mode_report in report["modes"]:
        metrics = mode_report["metrics"]
        print(f"\n=== {mode_report['mode']} ===")
        print(
            f"Recall@{mode_report['k']}: {metrics['recall_at_k']['hits']}/"
            f"{metrics['recall_at_k']['total']} = {metrics['recall_at_k']['value']:.3f}"
        )
        print(f"MRR@{mode_report['k']}: {metrics['mrr_at_k']['value']:.3f}")
        prec = metrics["precision_at_k"]
        if prec["value"] is None:
            print(f"Precision@{mode_report['k']}: n/a ({prec['note']})")
        else:
            print(
                f"Precision@{mode_report['k']}: {prec['value']:.3f} "
                f"over {prec['scored']} labeled questions"
            )
        lat = metrics["latency_ms"]
        if lat["mean"] is not None:
            print(f"Latency ms: mean={lat['mean']:.1f} p50={lat['p50']}")
        emb = metrics["embedding_tokens"]
        if emb["total"] is not None:
            cost = emb["estimated_cost_usd"]
            cost_s = f" ~${cost:.6f}" if cost is not None else ""
            print(f"Embedding tokens: {emb['total']}{cost_s}")
        for row in mode_report["results"]:
            if row.get("skipped"):
                print(f"  [SKIP] {row['id']}: {row.get('reason')}")
                continue
            mark = "HIT" if row.get("recall_hit") else "MISS"
            print(
                f"  [{mark}] {row['id']}: {row.get('retrieved_document_ids')} "
                f"mrr={row.get('mrr')} lat={row.get('latency_ms')}ms"
            )
            if row.get("error"):
                print(f"         error: {row['error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
