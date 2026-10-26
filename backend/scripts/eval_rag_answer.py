"""Phase 3.2/3.3 grounded RAG answer evaluation.

Requires a populated Chroma (+ BM25) collection. Live generation needs OPENAI_API_KEY.
Automated unit tests do not call this script with live generation.

Reports (clearly labeled):
  - Retrieval Recall@5 / MRR@5 — deterministic
  - Citation validity — deterministic reference integrity
  - Answer fact coverage — deterministic substring checks against
    expected_answer_facts (NOT proof of factual accuracy / not an LLM judge)
  - Abstention — for unanswerable items, insufficient_context must be true
  - Generation token usage and estimated cost when live answers run

    .\\.venv\\Scripts\\python.exe -m scripts.eval_rag_answer --retrieval-only
    .\\.venv\\Scripts\\python.exe -m scripts.eval_rag_answer --modes semantic,hybrid --retrieval-only
    .\\.venv\\Scripts\\python.exe -m scripts.eval_rag_answer --modes semantic,hybrid
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

from app.config import BACKEND_DIR, get_settings
from app.db.repository import Repository
from app.main import build_pricing, build_registry
from app.services.llm.factory import build_provider
from app.services.rag.answer import RagAnswerError, RagAnswerService, extract_citation_refs
from app.services.rag.embeddings import OpenAIEmbeddingProvider
from app.services.rag.factory import build_bm25_index, build_rag_reranker, build_vector_store, resolve_path
from app.services.rag.retrieve import RetrievalError, retrieve
from scripts.eval_retrieval import mrr_at_k, recall_at_k

DEFAULT_EVAL = BACKEND_DIR / "data" / "knowledge" / "eval" / "retrieval_eval.json"
CITATION_PATTERN = re.compile(r"\[(S\d+)\]", re.IGNORECASE)
MODE_CHOICES = ("semantic", "hybrid")


def fact_coverage(answer: str, facts: list[str]) -> dict:
    """Deterministic: fraction of expected fact substrings found (case-insensitive)."""

    text = (answer or "").lower()
    if not facts:
        return {"matched": 0, "total": 0, "ratio": 1.0, "missing": []}
    matched = [fact for fact in facts if fact.lower() in text]
    missing = [fact for fact in facts if fact.lower() not in text]
    return {
        "matched": len(matched),
        "total": len(facts),
        "ratio": len(matched) / len(facts),
        "missing": missing,
    }


def citation_validity(answer: str, citation_refs: list[str]) -> dict:
    """Deterministic: every [S#] in the answer must appear in citation_refs."""

    allowed = {ref.upper() for ref in citation_refs}
    claimed = [m.group(1).upper() for m in CITATION_PATTERN.finditer(answer or "")]
    invalid = sorted({label for label in claimed if label not in allowed})
    return {
        "claimed": claimed,
        "allowed": sorted(allowed),
        "invalid": invalid,
        "valid": len(invalid) == 0,
    }


async def _run_mode(
    *,
    mode: str,
    questions: list[dict],
    k: int,
    retrieval_only: bool,
    store,
    embedder,
    bm25_index,
    settings,
    answer_service: RagAnswerService | None,
) -> dict:
    rows = []
    recall_hits = 0
    mrr_scores: list[float] = []
    citation_ok = 0
    citation_checked = 0
    fact_scores: list[float] = []
    abstention_ok = 0
    abstention_total = 0
    answerable_answered = 0
    answerable_total = 0
    gen_tokens_in = 0
    gen_tokens_out = 0
    gen_cost = 0.0
    gen_cost_known = False
    embed_tokens_total = 0
    embed_tokens_known = False

    for item in questions:
        query = item["query"]
        expected = list(item.get("expected_document_ids") or [])
        facts = list(item.get("expected_answer_facts") or [])
        unanswerable = bool(item.get("unanswerable"))
        row: dict = {
            "id": item.get("id"),
            "query": query,
            "category": item.get("category"),
            "expected_document_ids": expected,
            "unanswerable": unanswerable,
            "checks": {
                "recall": "automated",
                "mrr": "automated",
                "citation_validity": "automated",
                "fact_coverage": "automated_substring_not_factual_proof",
                "abstention": "automated",
                "subjective_quality": "not_evaluated",
            },
        }

        try:
            retrieval = await retrieve(
                query,
                store=store,
                embedder=embedder,
                bm25_index=bm25_index,
                mode=mode,
                top_k=k,
                semantic_candidates=settings.rag_semantic_candidates,
                bm25_candidates=settings.rag_bm25_candidates,
                rrf_k=settings.rag_rrf_k,
                reranker=None,
            )
        except RetrievalError as exc:
            row["retrieval_error"] = exc.message
            row["retrieved_document_ids"] = []
            row["recall_hit"] = False
            row["mrr"] = 0.0
            rows.append(row)
            continue

        retrieved_docs: list[str] = []
        for hit in retrieval.hits:
            if hit.document_id not in retrieved_docs:
                retrieved_docs.append(hit.document_id)
        row["retrieved_document_ids"] = retrieved_docs
        row["retrieval_latency_ms"] = retrieval.latency_ms
        if retrieval.embedding_tokens is not None:
            embed_tokens_total += retrieval.embedding_tokens
            embed_tokens_known = True

        if unanswerable or not expected:
            row["recall_hit"] = None
            row["mrr"] = None
            row["recall_skipped"] = True
        else:
            score = recall_at_k(retrieved_docs, expected, k)
            mrr = mrr_at_k(retrieved_docs, expected, k)
            recall_hits += int(score)
            mrr_scores.append(mrr)
            row["recall_hit"] = bool(score)
            row["mrr"] = mrr

        if retrieval_only or answer_service is None:
            rows.append(row)
            continue

        try:
            response = await answer_service.ask(query, top_k=k, retrieval_mode=mode)
        except RagAnswerError as exc:
            row["answer_error"] = {"message": exc.message, "code": exc.code}
            rows.append(row)
            continue

        refs = [c.ref for c in response.citations]
        validity = citation_validity(response.answer, refs)
        claimed = extract_citation_refs(response.answer, [])
        if any(label not in {r.upper() for r in refs} for label in claimed):
            validity["valid"] = False
            validity["invalid"] = sorted(
                set(validity["invalid"])
                | {label for label in claimed if label not in {r.upper() for r in refs}}
            )

        citation_checked += 1
        citation_ok += int(validity["valid"])
        coverage = fact_coverage(response.answer, facts) if not unanswerable else {
            "matched": 0,
            "total": 0,
            "ratio": 1.0,
            "missing": [],
        }
        if not unanswerable and facts:
            fact_scores.append(coverage["ratio"])

        if unanswerable:
            abstention_total += 1
            abstention_ok += int(response.insufficient_context)
        else:
            answerable_total += 1
            answerable_answered += int(not response.insufficient_context)

        if response.input_tokens:
            gen_tokens_in += response.input_tokens
        if response.output_tokens:
            gen_tokens_out += response.output_tokens
        if response.estimated_cost_usd is not None:
            gen_cost += response.estimated_cost_usd
            gen_cost_known = True

        row.update(
            {
                "insufficient_context": response.insufficient_context,
                "citation_refs": refs,
                "citation_valid": validity["valid"],
                "citation_invalid": validity["invalid"],
                "fact_coverage": coverage,
                "model": response.model,
                "latency_ms": response.latency_ms,
                "estimated_cost_usd": response.estimated_cost_usd,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "retrieval_mode": response.retrieval_mode,
                "answer_preview": response.answer[:240],
            }
        )
        rows.append(row)

    recall_total = sum(1 for row in rows if row.get("recall_hit") is not None)
    return {
        "mode": mode,
        "k": k,
        "questions": len(questions),
        "metrics": {
            "recall_at_k": {
                "value": (recall_hits / recall_total) if recall_total else 0.0,
                "hits": recall_hits,
                "total": recall_total,
                "kind": "automated",
            },
            "mrr_at_k": {
                "value": (sum(mrr_scores) / len(mrr_scores)) if mrr_scores else 0.0,
                "scored": len(mrr_scores),
                "kind": "automated",
            },
            "citation_validity": {
                "value": (citation_ok / citation_checked) if citation_checked else None,
                "ok": citation_ok,
                "checked": citation_checked,
                "kind": "automated",
            },
            "answer_fact_coverage": {
                "value": (sum(fact_scores) / len(fact_scores)) if fact_scores else None,
                "questions_scored": len(fact_scores),
                "kind": "automated_substring",
                "note": (
                    "Substring presence only — not an LLM-as-judge score and not proof "
                    "of factual accuracy."
                ),
            },
            "abstention_on_unanswerable": {
                "value": (abstention_ok / abstention_total) if abstention_total else None,
                "ok": abstention_ok,
                "total": abstention_total,
                "kind": "automated",
            },
            "answered_when_answerable": {
                "value": (answerable_answered / answerable_total) if answerable_total else None,
                "ok": answerable_answered,
                "total": answerable_total,
                "kind": "automated",
            },
            "generation_tokens": {
                "input": gen_tokens_in if not retrieval_only else None,
                "output": gen_tokens_out if not retrieval_only else None,
                "estimated_cost_usd": gen_cost if gen_cost_known else None,
                "kind": "provider_reported" if not retrieval_only else "skipped",
            },
            "embedding_tokens": {
                "total": embed_tokens_total if embed_tokens_known else None,
                "kind": "provider_reported" if embed_tokens_known else "unavailable",
            },
        },
        "results": rows,
    }


async def run_eval(*, eval_path: Path, k: int, retrieval_only: bool, modes: list[str]) -> dict:
    settings = get_settings()
    payload = json.loads(eval_path.read_text(encoding="utf-8"))
    questions = payload.get("questions") or []
    store = build_vector_store(settings)
    bm25_index = build_bm25_index(settings)
    repo = Repository(settings.database_path)
    embedder = OpenAIEmbeddingProvider(settings, repository=repo)

    if store.count() == 0:
        raise SystemExit(
            "Vector store is empty. Ingest documents first:\n"
            "  .\\.venv\\Scripts\\python.exe -m scripts.ingest_knowledge"
        )
    if bm25_index.size == 0 and store.count() > 0:
        bm25_index.rebuild_from_store(store)
    elif bm25_index.size != store.count():
        bm25_index.rebuild_from_store(store)

    answer_service: RagAnswerService | None = None
    if not retrieval_only:
        if not settings.llm_configured:
            raise SystemExit(
                "OPENAI_API_KEY is required for answer evaluation (or pass --retrieval-only)."
            )
        answer_service = RagAnswerService(
            repository=repo,
            provider=build_provider(settings),
            settings=settings,
            registry=build_registry(settings),
            store=store,
            embedder=embedder,
            pricing=build_pricing(settings),
            bm25_index=bm25_index,
            reranker=build_rag_reranker(settings),
        )

    mode_reports = []
    for mode in modes:
        mode_reports.append(
            await _run_mode(
                mode=mode,
                questions=questions,
                k=k,
                retrieval_only=retrieval_only,
                store=store,
                embedder=embedder,
                bm25_index=bm25_index,
                settings=settings,
                answer_service=answer_service,
            )
        )

    return {
        "dataset": payload.get("name"),
        "eval_path": str(eval_path),
        "run_mode": "retrieval_only" if retrieval_only else "retrieval_and_answer",
        "k": k,
        "store_chunks": store.count(),
        "bm25_chunks": bm25_index.size,
        "embedding_model": settings.embedding_model,
        "modes": mode_reports,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate grounded RAG answers.")
    parser.add_argument("--eval", type=str, default=str(DEFAULT_EVAL), help="Path to eval JSON.")
    parser.add_argument("--k", type=int, default=5, help="Top-K / Recall@K cutoff.")
    parser.add_argument("--json", action="store_true", help="Print full JSON report.")
    parser.add_argument(
        "--retrieval-only",
        action="store_true",
        help="Skip LLM generation; report retrieval metrics only (still embeds queries).",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default="semantic,hybrid",
        help="Comma-separated retrieval modes: semantic,hybrid",
    )
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

    report = asyncio.run(
        run_eval(
            eval_path=eval_path,
            k=args.k,
            retrieval_only=args.retrieval_only,
            modes=modes,
        )
    )
    if args.json:
        print(json.dumps(report, indent=2))
        return 0

    print(f"Dataset: {report['dataset']} ({report['run_mode']})")
    print(f"Store chunks: {report['store_chunks']} · BM25 chunks: {report['bm25_chunks']}")
    for mode_report in report["modes"]:
        metrics = mode_report["metrics"]
        print(f"\n=== {mode_report['mode']} ===")
        print(
            f"Recall@{mode_report['k']}: {metrics['recall_at_k']['hits']}/"
            f"{metrics['recall_at_k']['total']} = {metrics['recall_at_k']['value']:.3f} [automated]"
        )
        print(f"MRR@{mode_report['k']}: {metrics['mrr_at_k']['value']:.3f} [automated]")
        if not args.retrieval_only:
            cv = metrics["citation_validity"]
            if cv["checked"]:
                print(f"Citation validity: {cv['ok']}/{cv['checked']} = {cv['value']:.3f} [automated]")
            fc = metrics["answer_fact_coverage"]
            if fc["value"] is not None:
                print(
                    f"Answer fact coverage: {fc['value']:.3f} "
                    f"over {fc['questions_scored']} questions [automated substring — not factual proof]"
                )
            ab = metrics["abstention_on_unanswerable"]
            if ab["total"]:
                print(
                    f"Abstention (unanswerable): {ab['ok']}/{ab['total']} = {ab['value']:.3f} [automated]"
                )
            an = metrics["answered_when_answerable"]
            if an["total"]:
                print(
                    f"Answered when answerable: {an['ok']}/{an['total']} = {an['value']:.3f} [automated]"
                )
            gen = metrics["generation_tokens"]
            if gen["input"] is not None:
                cost = gen["estimated_cost_usd"]
                cost_s = f" ~${cost:.6f}" if cost is not None else ""
                print(f"Generation tokens: in={gen['input']} out={gen['output']}{cost_s}")
            print("Subjective fluency/quality: not evaluated")
        for row in mode_report["results"]:
            if row.get("recall_skipped"):
                mark = "SKIP"
            else:
                mark = "HIT" if row.get("recall_hit") else "MISS"
            extra = ""
            if row.get("insufficient_context"):
                extra = " abstain"
            elif row.get("citation_valid") is False:
                extra = f" bad_cites={row.get('citation_invalid')}"
            print(f"  [{mark}] {row['id']}: {row.get('retrieved_document_ids')}{extra}")
            if row.get("retrieval_error"):
                print(f"         retrieval_error: {row['retrieval_error']}")
            if row.get("answer_error"):
                print(f"         answer_error: {row['answer_error']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
