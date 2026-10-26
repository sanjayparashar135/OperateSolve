"""Knowledge base search and grounded answers (Phase 3.1–3.3)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request

from app.db.repository import Repository
from app.schemas.knowledge import (
    KnowledgeAskRequest,
    KnowledgeAskResponse,
    KnowledgeSearchHit,
    KnowledgeSearchRequest,
    KnowledgeSearchResponse,
)
from app.services.rag.answer import RagAnswerError, RagAnswerService
from app.services.rag.factory import (
    build_bm25_index,
    build_embedder,
    build_rag_reranker,
    build_vector_store,
)
from app.services.rag.retrieve import RetrievalError, normalize_retrieval_mode, retrieve

router = APIRouter(prefix="/api", tags=["knowledge"])


def _ensure_rag(request: Request) -> tuple[object, object, object]:
    settings = request.app.state.settings
    store = getattr(request.app.state, "vector_store", None)
    embedder = getattr(request.app.state, "embedder", None)
    bm25_index = getattr(request.app.state, "bm25_index", None)
    if store is None:
        store = build_vector_store(settings)
        request.app.state.vector_store = store
    if embedder is None:
        repo = Repository(settings.database_path)
        embedder = build_embedder(settings, repository=repo)
        request.app.state.embedder = embedder
    if bm25_index is None:
        bm25_index = build_bm25_index(settings)
        request.app.state.bm25_index = bm25_index
        # Heal empty/stale sidecar from an already-populated Chroma store.
        if bm25_index.size == 0 and store.count() > 0:
            bm25_index.rebuild_from_store(store)
        elif bm25_index.size != store.count() and store.count() > 0:
            bm25_index.rebuild_from_store(store)
    return store, embedder, bm25_index


def get_rag_answer_service(request: Request) -> RagAnswerService:
    settings = request.app.state.settings
    store, embedder, bm25_index = _ensure_rag(request)
    reranker = getattr(request.app.state, "rag_reranker", None)
    if reranker is None:
        reranker = build_rag_reranker(settings)
        request.app.state.rag_reranker = reranker
    return RagAnswerService(
        repository=Repository(settings.database_path),
        provider=request.app.state.provider,
        settings=settings,
        registry=request.app.state.registry,
        store=store,
        embedder=embedder,
        pricing=request.app.state.pricing,
        bm25_index=bm25_index,
        reranker=reranker,
    )


@router.post("/knowledge/search", response_model=KnowledgeSearchResponse)
async def search_knowledge(
    payload: KnowledgeSearchRequest,
    request: Request,
) -> KnowledgeSearchResponse:
    settings = request.app.state.settings
    store, embedder, bm25_index = _ensure_rag(request)
    reranker = getattr(request.app.state, "rag_reranker", None)
    if reranker is None:
        reranker = build_rag_reranker(settings)
        request.app.state.rag_reranker = reranker

    try:
        mode = normalize_retrieval_mode(
            payload.retrieval_mode,
            default=settings.rag_retrieval_mode,
        )
        result = await retrieve(
            payload.query,
            store=store,
            embedder=embedder,
            bm25_index=bm25_index,
            mode=mode,
            top_k=payload.top_k,
            semantic_candidates=settings.rag_semantic_candidates,
            bm25_candidates=settings.rag_bm25_candidates,
            rrf_k=settings.rag_rrf_k,
            reranker=reranker if mode == "hybrid" else None,
        )
    except RetrievalError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"message": exc.message, "code": exc.code},
        ) from exc

    return KnowledgeSearchResponse(
        query=payload.query.strip(),
        top_k=payload.top_k,
        embedding_model=getattr(embedder, "model", settings.embedding_model),
        hit_count=len(result.hits),
        retrieval_mode=result.mode,
        reranker=result.reranker,
        semantic_candidates=result.semantic_candidates,
        bm25_candidates=result.bm25_candidates,
        retrieval_latency_ms=result.latency_ms,
        hits=[
            KnowledgeSearchHit(
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
                semantic_rank=hit.semantic_rank,
                bm25_rank=hit.bm25_rank,
                retrieval_method=hit.retrieval_method,
            )
            for hit in result.hits
        ],
    )


@router.post("/knowledge/ask", response_model=KnowledgeAskResponse)
async def ask_knowledge(
    payload: KnowledgeAskRequest,
    service: RagAnswerService = Depends(get_rag_answer_service),
) -> KnowledgeAskResponse:
    try:
        return await service.ask(
            payload.question,
            top_k=payload.top_k,
            retrieval_mode=payload.retrieval_mode,
        )
    except RagAnswerError as exc:
        raise HTTPException(
            status_code=exc.status_code,
            detail={"message": exc.message, "code": exc.code},
        ) from exc
