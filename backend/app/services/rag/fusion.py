"""Reciprocal Rank Fusion (RRF) for combining ranked retrieval lists."""

from __future__ import annotations

from app.services.rag.models import SearchHit


def reciprocal_rank_fusion(
    *lists: list[SearchHit],
    k: int = 60,
    top_k: int = 5,
) -> list[SearchHit]:
    """Fuse ranked hit lists with RRF; dedupe by chunk_id; preserve score provenance.

    RRF score(d) = Σ 1 / (k + rank_i(d)) over lists where d appears.
    Cosine similarity/distance and BM25 scores are carried from the contributing lists.
    """

    if k < 1:
        raise ValueError("RRF k must be at least 1.")
    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    scores: dict[str, float] = {}
    merged: dict[str, SearchHit] = {}

    for hits in lists:
        for hit in hits:
            chunk_id = hit.chunk_id
            contribution = 1.0 / (k + hit.rank)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + contribution
            existing = merged.get(chunk_id)
            if existing is None:
                merged[chunk_id] = hit
            else:
                merged[chunk_id] = _merge_hits(existing, hit)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]
    fused: list[SearchHit] = []
    for rank, (chunk_id, rrf_score) in enumerate(ordered, start=1):
        base = merged[chunk_id]
        methods = set()
        if base.semantic_rank is not None or base.retrieval_method == "semantic":
            methods.add("semantic")
        if base.bm25_rank is not None or base.retrieval_method == "bm25":
            methods.add("bm25")
        if base.retrieval_method == "hybrid":
            methods.update({"semantic", "bm25"})
        method = "hybrid" if len(methods) > 1 else (next(iter(methods)) if methods else "hybrid")
        fused.append(
            SearchHit(
                chunk_id=base.chunk_id,
                document_id=base.document_id,
                document_title=base.document_title,
                source=base.source,
                document_type=base.document_type,
                section_heading=base.section_heading,
                content=base.content,
                distance=base.distance,
                similarity=base.similarity,
                rank=rank,
                bm25_score=base.bm25_score,
                rrf_score=rrf_score,
                semantic_rank=base.semantic_rank,
                bm25_rank=base.bm25_rank,
                retrieval_method=method,
            )
        )
    return fused


def _merge_hits(left: SearchHit, right: SearchHit) -> SearchHit:
    """Combine provenance from two hits for the same chunk_id."""

    # Prefer real cosine similarity when one side is BM25-only (similarity 0, distance 1).
    if right.semantic_rank is not None and left.semantic_rank is None:
        distance, similarity, semantic_rank = right.distance, right.similarity, right.semantic_rank
    elif left.semantic_rank is not None:
        distance, similarity, semantic_rank = left.distance, left.similarity, left.semantic_rank
    else:
        # Both semantic or neither: keep the better similarity.
        if right.similarity >= left.similarity:
            distance, similarity, semantic_rank = right.distance, right.similarity, right.semantic_rank
        else:
            distance, similarity, semantic_rank = left.distance, left.similarity, left.semantic_rank

    if right.bm25_rank is not None and left.bm25_rank is None:
        bm25_score, bm25_rank = right.bm25_score, right.bm25_rank
    elif left.bm25_rank is not None and right.bm25_rank is None:
        bm25_score, bm25_rank = left.bm25_score, left.bm25_rank
    elif left.bm25_score is not None and right.bm25_score is not None:
        if right.bm25_score >= left.bm25_score:
            bm25_score, bm25_rank = right.bm25_score, right.bm25_rank
        else:
            bm25_score, bm25_rank = left.bm25_score, left.bm25_rank
    else:
        bm25_score = left.bm25_score if left.bm25_score is not None else right.bm25_score
        bm25_rank = left.bm25_rank if left.bm25_rank is not None else right.bm25_rank

    return SearchHit(
        chunk_id=left.chunk_id,
        document_id=left.document_id or right.document_id,
        document_title=left.document_title or right.document_title,
        source=left.source or right.source,
        document_type=left.document_type or right.document_type,
        section_heading=left.section_heading or right.section_heading,
        content=left.content or right.content,
        distance=distance,
        similarity=similarity,
        rank=min(left.rank, right.rank),
        bm25_score=bm25_score,
        rrf_score=None,
        semantic_rank=semantic_rank,
        bm25_rank=bm25_rank,
        retrieval_method="hybrid",
    )
