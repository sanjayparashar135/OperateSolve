"""Optional lightweight reranking after hybrid fusion.

Genuine query-dependent reranking is distinct from sorting by RRF score.
Heavy cross-encoders are intentionally avoided. The default production path
keeps reranking disabled (`rag_reranker_enabled=false`).

`LexicalOverlapReranker` reorders candidates by query↔chunk token overlap
(Jaccard). It is local, free, and query-dependent, but it is a weak signal —
do not claim quality gains unless measured on the eval set with it enabled.
"""

from __future__ import annotations

import logging
from typing import Protocol

from app.services.rag.bm25 import tokenize
from app.services.rag.models import SearchHit

logger = logging.getLogger(__name__)


class Reranker(Protocol):
    """Reorder retrieval candidates for a query without extra paid API calls."""

    name: str

    def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        ...


class NoOpReranker:
    """Pass-through; preserves hybrid/RRF order."""

    name = "noop"

    def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        return list(hits[:top_k])


class LexicalOverlapReranker:
    """Query-dependent local reranker using Jaccard token overlap.

    This is not a neural cross-encoder. Prefer measuring before enabling.
    """

    name = "lexical_overlap"

    def rerank(self, query: str, hits: list[SearchHit], *, top_k: int) -> list[SearchHit]:
        query_tokens = set(tokenize(query))
        if not query_tokens or not hits:
            return list(hits[:top_k])

        scored: list[tuple[float, int, SearchHit]] = []
        for index, hit in enumerate(hits):
            chunk_tokens = set(tokenize(hit.content))
            if not chunk_tokens:
                overlap = 0.0
            else:
                intersection = len(query_tokens & chunk_tokens)
                union = len(query_tokens | chunk_tokens)
                overlap = intersection / union if union else 0.0
            scored.append((overlap, index, hit))

        # Higher overlap first; stable tie-break by prior rank (index).
        scored.sort(key=lambda item: (-item[0], item[1]))
        reranked: list[SearchHit] = []
        for rank, (overlap, _, hit) in enumerate(scored[:top_k], start=1):
            reranked.append(
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
                    rank=rank,
                    bm25_score=hit.bm25_score,
                    rrf_score=hit.rrf_score,
                    semantic_rank=hit.semantic_rank,
                    bm25_rank=hit.bm25_rank,
                    retrieval_method=hit.retrieval_method,
                )
            )
            logger.debug(
                "rerank_lexical chunk_id=%s overlap=%.4f prior_rank=%s new_rank=%s",
                hit.chunk_id,
                overlap,
                hit.rank,
                rank,
            )
        return reranked


def build_reranker(*, enabled: bool, name: str = "lexical_overlap") -> Reranker:
    if not enabled:
        return NoOpReranker()
    key = (name or "lexical_overlap").strip().lower()
    if key in {"lexical_overlap", "lexical", "jaccard"}:
        return LexicalOverlapReranker()
    if key in {"noop", "none", "off"}:
        return NoOpReranker()
    logger.warning("unknown_reranker name=%s falling_back=noop", name)
    return NoOpReranker()
