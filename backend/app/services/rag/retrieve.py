"""Knowledge retrieval: semantic baseline, BM25 keyword, and hybrid RRF fusion."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

from app.services.rag.bm25 import Bm25Index
from app.services.rag.embeddings import EmbeddingError, EmbeddingProvider
from app.services.rag.fusion import reciprocal_rank_fusion
from app.services.rag.models import SearchHit
from app.services.rag.rerank import NoOpReranker, Reranker
from app.services.rag.store import ChromaVectorStore, VectorStoreError

logger = logging.getLogger(__name__)

RETRIEVAL_MODES = frozenset({"semantic", "hybrid"})


class RetrievalError(RuntimeError):
    def __init__(self, message: str, *, status_code: int = 500, code: str = "retrieval_error") -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code


@dataclass(frozen=True)
class RetrievalResult:
    hits: list[SearchHit]
    mode: str
    top_k: int
    latency_ms: int
    semantic_candidates: int = 0
    bm25_candidates: int = 0
    reranker: str = "noop"
    embedding_tokens: int | None = None


def normalize_retrieval_mode(mode: str | None, *, default: str = "semantic") -> str:
    value = (mode or default).strip().lower()
    if value not in RETRIEVAL_MODES:
        raise RetrievalError(
            f"retrieval_mode must be one of: {', '.join(sorted(RETRIEVAL_MODES))}.",
            status_code=400,
            code="invalid_retrieval_mode",
        )
    return value


async def semantic_search(
    query: str,
    *,
    store: ChromaVectorStore,
    embedder: EmbeddingProvider,
    top_k: int = 5,
) -> list[SearchHit]:
    """Embed the query and return the top_k nearest chunks (Phase 3.1 baseline)."""

    result = await retrieve(
        query,
        store=store,
        embedder=embedder,
        mode="semantic",
        top_k=top_k,
    )
    return result.hits


async def retrieve(
    query: str,
    *,
    store: ChromaVectorStore,
    embedder: EmbeddingProvider,
    bm25_index: Bm25Index | None = None,
    mode: str = "semantic",
    top_k: int = 5,
    semantic_candidates: int | None = None,
    bm25_candidates: int | None = None,
    rrf_k: int = 60,
    reranker: Reranker | None = None,
) -> RetrievalResult:
    """Run semantic or hybrid retrieval; optionally apply a local reranker."""

    cleaned = (query or "").strip()
    if not cleaned:
        raise RetrievalError("Query must not be empty.", status_code=400, code="empty_query")
    if top_k < 1 or top_k > 50:
        raise RetrievalError("top_k must be between 1 and 50.", status_code=400, code="invalid_top_k")

    mode = normalize_retrieval_mode(mode)
    sem_n = semantic_candidates if semantic_candidates is not None else max(top_k, 20)
    bm25_n = bm25_candidates if bm25_candidates is not None else max(top_k, 20)
    if sem_n < 1 or sem_n > 100:
        raise RetrievalError(
            "semantic_candidates must be between 1 and 100.",
            status_code=400,
            code="invalid_semantic_candidates",
        )
    if bm25_n < 1 or bm25_n > 100:
        raise RetrievalError(
            "bm25_candidates must be between 1 and 100.",
            status_code=400,
            code="invalid_bm25_candidates",
        )
    if rrf_k < 1:
        raise RetrievalError("rrf_k must be at least 1.", status_code=400, code="invalid_rrf_k")

    active_reranker: Reranker = reranker or NoOpReranker()
    started = time.perf_counter()

    try:
        if store.count() == 0:
            logger.info("knowledge_search_empty query_len=%s mode=%s", len(cleaned), mode)
            return RetrievalResult(
                hits=[],
                mode=mode,
                top_k=top_k,
                latency_ms=int((time.perf_counter() - started) * 1000),
                reranker=active_reranker.name,
            )

        if mode == "semantic":
            hits, embed_tokens = await _semantic_hits(
                cleaned, store=store, embedder=embedder, top_k=top_k
            )
            hits = _annotate_semantic(hits)
            latency_ms = int((time.perf_counter() - started) * 1000)
            logger.info(
                "knowledge_search mode=semantic query_len=%s top_k=%s hits=%s latency_ms=%s",
                len(cleaned),
                top_k,
                len(hits),
                latency_ms,
            )
            return RetrievalResult(
                hits=hits,
                mode=mode,
                top_k=top_k,
                latency_ms=latency_ms,
                semantic_candidates=top_k,
                reranker="noop",
                embedding_tokens=embed_tokens,
            )

        # Hybrid: run semantic embedding search and BM25 in parallel.
        semantic_task = asyncio.create_task(
            _semantic_hits(cleaned, store=store, embedder=embedder, top_k=sem_n)
        )
        bm25_task = asyncio.create_task(
            asyncio.to_thread(_bm25_hits, cleaned, bm25_index, bm25_n)
        )
        (semantic_hits, embed_tokens), bm25_hits = await asyncio.gather(semantic_task, bm25_task)
        semantic_hits = _annotate_semantic(semantic_hits)

        fused = reciprocal_rank_fusion(semantic_hits, bm25_hits, k=rrf_k, top_k=max(top_k, sem_n))
        # Pull a larger fused pool into the reranker, then cut to top_k.
        pool = fused[: max(top_k, min(len(fused), max(sem_n, bm25_n)))]
        ranked = active_reranker.rerank(cleaned, pool, top_k=top_k)
        # Ensure final ranks are 1..n after rerank.
        ranked = [
            SearchHit(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                document_title=hit.document_title,
                source=hit.source,
                document_type=hit.document_type,
                section_heading=hit.section_heading,
                content=hit.content,
                distance=hit.distance,
                similarity=hit.similarity,
                rank=index,
                bm25_score=hit.bm25_score,
                rrf_score=hit.rrf_score,
                semantic_rank=hit.semantic_rank,
                bm25_rank=hit.bm25_rank,
                retrieval_method=hit.retrieval_method if hit.retrieval_method != "semantic" else (
                    "hybrid" if hit.bm25_rank is not None else "semantic"
                ),
            )
            for index, hit in enumerate(ranked, start=1)
        ]

        latency_ms = int((time.perf_counter() - started) * 1000)
        logger.info(
            "knowledge_search mode=hybrid query_len=%s top_k=%s semantic=%s bm25=%s "
            "fused=%s reranker=%s latency_ms=%s",
            len(cleaned),
            top_k,
            len(semantic_hits),
            len(bm25_hits),
            len(ranked),
            active_reranker.name,
            latency_ms,
        )
        return RetrievalResult(
            hits=ranked,
            mode=mode,
            top_k=top_k,
            latency_ms=latency_ms,
            semantic_candidates=len(semantic_hits),
            bm25_candidates=len(bm25_hits),
            reranker=active_reranker.name,
            embedding_tokens=embed_tokens,
        )
    except EmbeddingError as exc:
        raise RetrievalError(str(exc), status_code=502, code="embedding_failed") from exc
    except VectorStoreError as exc:
        raise RetrievalError(str(exc), status_code=500, code="vector_store_error") from exc


async def _semantic_hits(
    query: str,
    *,
    store: ChromaVectorStore,
    embedder: EmbeddingProvider,
    top_k: int,
) -> tuple[list[SearchHit], int | None]:
    embedded = await embedder.embed_texts([query])
    if not embedded.vectors:
        raise RetrievalError("Embedding provider returned no vector.", code="embed_empty")
    hits = store.search(embedded.vectors[0], top_k=top_k)
    return hits, embedded.input_tokens


def _bm25_hits(query: str, index: Bm25Index | None, top_k: int) -> list[SearchHit]:
    if index is None:
        logger.warning("bm25_unavailable reason=no_index")
        return []
    if index.size == 0:
        # Attempt lazy load from disk (stale process / first request after restart).
        if not index.load() or index.size == 0:
            logger.warning("bm25_unavailable reason=empty_index")
            return []
    try:
        return index.search(query, top_k=top_k)
    except Exception:
        logger.exception("bm25_search_failed")
        return []


def _annotate_semantic(hits: list[SearchHit]) -> list[SearchHit]:
    annotated: list[SearchHit] = []
    for hit in hits:
        annotated.append(
            SearchHit(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                document_title=hit.document_title,
                source=hit.source,
                document_type=hit.document_type,
                section_heading=hit.section_heading,
                content=hit.content,
                distance=hit.distance,
                similarity=hit.similarity,
                rank=hit.rank,
                bm25_score=hit.bm25_score,
                rrf_score=hit.rrf_score,
                semantic_rank=hit.rank,
                bm25_rank=hit.bm25_rank,
                retrieval_method="semantic",
            )
        )
    return annotated
