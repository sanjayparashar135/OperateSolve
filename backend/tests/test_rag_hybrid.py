"""Phase 3.3 hybrid retrieval tests (no live OpenAI calls)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.rag.bm25 import Bm25Index, tokenize
from app.services.rag.embeddings import FakeEmbeddingProvider
from app.services.rag.fusion import reciprocal_rank_fusion
from app.services.rag.ingest import ingest_directory
from app.services.rag.models import DocumentChunk, SearchHit
from app.services.rag.rerank import LexicalOverlapReranker, NoOpReranker, build_reranker
from app.services.rag.retrieve import RetrievalError, retrieve
from app.services.rag.store import ChromaVectorStore
from tests.conftest import make_settings
from tests.fakes import FakeProvider


@pytest.fixture
def knowledge_root(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir()
    (root / "oms-runbook.md").write_text(
        """# OMS Database Connection Exhaustion Runbook

**Document type:** runbook

## Symptoms

OMS returns HTTP 503 with code oms_overloaded when the Postgres pool is saturated.
Incident INC-2407 documented this failure mode.

## Recovery procedure

Do not scale OMS replicas first. Terminate idle in transaction sessions, then reduce pool size.
""",
        encoding="utf-8",
    )
    (root / "oms-api.md").write_text(
        """# OMS API Specification

**Document type:** api_spec

## Endpoints

POST /v1/orders requires Idempotency-Key header.
Error code inventory_unavailable means IRS hard reserve failed.
Error code payment_upstream_error means AcmePay failure.
""",
        encoding="utf-8",
    )
    (root / "architecture.md").write_text(
        """# Architecture Overview

**Document type:** architecture

## Core services

Order Management API depends on Payment Gateway Adapter (PGA) and Inventory Reservation Service (IRS).
""",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(
        tmp_path / "chroma",
        collection_name="test_hybrid",
        embedding_model="fake-embed",
        embedding_dimensions=32,
    )


@pytest.fixture
def embedder() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(dimensions=32)


@pytest.fixture
def bm25_index(tmp_path: Path) -> Bm25Index:
    return Bm25Index(tmp_path / "bm25" / "index.json")


def _hit(
    chunk_id: str,
    *,
    rank: int,
    document_id: str = "doc",
    content: str = "content",
    similarity: float = 0.5,
    bm25_score: float | None = None,
    method: str = "semantic",
) -> SearchHit:
    return SearchHit(
        chunk_id=chunk_id,
        document_id=document_id,
        document_title=document_id,
        source=f"{document_id}.md",
        document_type="general",
        section_heading="Section",
        content=content,
        distance=1.0 - similarity,
        similarity=similarity,
        rank=rank,
        bm25_score=bm25_score,
        semantic_rank=rank if method in {"semantic", "hybrid"} else None,
        bm25_rank=rank if method in {"bm25", "hybrid"} else None,
        retrieval_method=method,
    )


def test_tokenize_keeps_incident_ids():
    tokens = tokenize("Recover INC-2407 oms_overloaded via POST /v1/orders")
    assert "inc-2407" in tokens
    assert "oms_overloaded" in tokens
    assert "v1/orders" in tokens or "orders" in tokens


def test_bm25_index_search_and_persist(tmp_path: Path):
    index = Bm25Index(tmp_path / "bm25.json")
    chunks = [
        DocumentChunk(
            chunk_id="a#0",
            document_id="a",
            document_title="A",
            source="a.md",
            document_type="runbook",
            section_heading="Symptoms",
            content="OMS returns oms_overloaded during INC-2407 connection exhaustion",
            chunk_index=0,
            content_hash="h1",
        ),
        DocumentChunk(
            chunk_id="b#0",
            document_id="b",
            document_title="B",
            source="b.md",
            document_type="api_spec",
            section_heading="Endpoints",
            content="POST /v1/orders Idempotency-Key inventory_unavailable",
            chunk_index=0,
            content_hash="h2",
        ),
    ]
    index.rebuild(chunks)
    index.save()

    hits = index.search("INC-2407 oms_overloaded", top_k=2)
    assert hits
    assert hits[0].chunk_id == "a#0"
    assert hits[0].bm25_score is not None and hits[0].bm25_score > 0
    assert hits[0].retrieval_method == "bm25"

    exact = index.search("Idempotency-Key inventory_unavailable", top_k=1)
    assert exact[0].chunk_id == "b#0"

    reloaded = Bm25Index(tmp_path / "bm25.json")
    assert reloaded.load()
    assert reloaded.size == 2
    assert reloaded.search("INC-2407", top_k=1)[0].chunk_id == "a#0"


def test_bm25_missing_index_returns_empty(tmp_path: Path):
    index = Bm25Index(tmp_path / "missing.json")
    assert index.search("anything", top_k=5) == []


def test_bm25_stale_index_returns_empty(tmp_path: Path):
    index = Bm25Index(tmp_path / "stale.json")
    index.rebuild(
        [
            DocumentChunk(
                chunk_id="a#0",
                document_id="a",
                document_title="A",
                source="a.md",
                document_type="general",
                section_heading="",
                content="hello world",
                chunk_index=0,
                content_hash="h",
            )
        ]
    )
    index._stale = True
    assert index.search("hello", top_k=5) == []


def test_rrf_fusion_dedupes_and_ranks():
    semantic = [
        _hit("a#0", rank=1, similarity=0.9, method="semantic"),
        _hit("b#0", rank=2, similarity=0.8, method="semantic"),
        _hit("c#0", rank=3, similarity=0.7, method="semantic"),
    ]
    bm25 = [
        _hit("b#0", rank=1, bm25_score=4.0, method="bm25"),
        _hit("a#0", rank=2, bm25_score=2.0, method="bm25"),
        _hit("d#0", rank=3, bm25_score=1.0, method="bm25"),
    ]
    fused = reciprocal_rank_fusion(semantic, bm25, k=60, top_k=4)
    ids = [hit.chunk_id for hit in fused]
    assert len(ids) == len(set(ids))
    assert ids[0] in {"a#0", "b#0"}
    # b appears rank 2 semantic + rank 1 bm25 → strong RRF
    assert "b#0" in ids[:2]
    b = next(hit for hit in fused if hit.chunk_id == "b#0")
    assert b.rrf_score is not None
    assert b.semantic_rank == 2
    assert b.bm25_rank == 1
    assert b.retrieval_method == "hybrid"


def test_rrf_preserves_distinct_score_types():
    semantic = [_hit("x#0", rank=1, similarity=0.95, method="semantic")]
    bm25 = [_hit("x#0", rank=1, bm25_score=3.5, method="bm25")]
    fused = reciprocal_rank_fusion(semantic, bm25, k=60, top_k=1)[0]
    assert fused.similarity == pytest.approx(0.95)
    assert fused.bm25_score == pytest.approx(3.5)
    assert fused.rrf_score == pytest.approx(2.0 / 61.0)


@pytest.mark.asyncio
async def test_hybrid_retrieval_mode_switch(
    knowledge_root: Path,
    store: ChromaVectorStore,
    embedder: FakeEmbeddingProvider,
    bm25_index: Bm25Index,
):
    ingest_directory(
        knowledge_root,
        store,
        embedder,
        chunk_size_tokens=120,
        overlap_tokens=20,
        bm25_index=bm25_index,
    )
    assert bm25_index.size == store.count()

    semantic = await retrieve(
        "INC-2407 oms_overloaded",
        store=store,
        embedder=embedder,
        bm25_index=bm25_index,
        mode="semantic",
        top_k=3,
    )
    hybrid = await retrieve(
        "INC-2407 oms_overloaded",
        store=store,
        embedder=embedder,
        bm25_index=bm25_index,
        mode="hybrid",
        top_k=3,
        semantic_candidates=5,
        bm25_candidates=5,
        rrf_k=60,
    )
    assert semantic.mode == "semantic"
    assert hybrid.mode == "hybrid"
    assert semantic.hits
    assert hybrid.hits
    assert any(hit.rrf_score is not None for hit in hybrid.hits)
    assert hybrid.bm25_candidates >= 1


@pytest.mark.asyncio
async def test_hybrid_with_missing_bm25_falls_back_to_semantic_side(
    knowledge_root: Path,
    store: ChromaVectorStore,
    embedder: FakeEmbeddingProvider,
    tmp_path: Path,
):
    ingest_directory(knowledge_root, store, embedder, chunk_size_tokens=120, overlap_tokens=20)
    empty = Bm25Index(tmp_path / "empty-bm25.json")
    result = await retrieve(
        "database connection exhaustion",
        store=store,
        embedder=embedder,
        bm25_index=empty,
        mode="hybrid",
        top_k=3,
    )
    assert result.hits
    assert result.bm25_candidates == 0


@pytest.mark.asyncio
async def test_invalid_retrieval_mode(
    store: ChromaVectorStore,
    embedder: FakeEmbeddingProvider,
    bm25_index: Bm25Index,
):
    with pytest.raises(RetrievalError) as exc:
        await retrieve(
            "query",
            store=store,
            embedder=embedder,
            bm25_index=bm25_index,
            mode="magic",
            top_k=3,
        )
    assert exc.value.code == "invalid_retrieval_mode"


def test_ingest_keeps_bm25_in_sync(
    knowledge_root: Path,
    store: ChromaVectorStore,
    embedder: FakeEmbeddingProvider,
    bm25_index: Bm25Index,
):
    report = ingest_directory(
        knowledge_root,
        store,
        embedder,
        chunk_size_tokens=120,
        overlap_tokens=20,
        bm25_index=bm25_index,
    )
    assert report.documents_indexed == 3
    assert bm25_index.size == store.count()

    (knowledge_root / "architecture.md").unlink()
    ingest_directory(
        knowledge_root,
        store,
        embedder,
        chunk_size_tokens=120,
        overlap_tokens=20,
        bm25_index=bm25_index,
    )
    assert store.chunk_ids_for_document("architecture") == []
    assert bm25_index.size == store.count()
    assert all(r.document_id != "architecture" for r in bm25_index._records)


def test_lexical_reranker_is_query_dependent_not_rrf_resort():
    hits = [
        _hit("a#0", rank=1, content="unrelated shipping tracking labels", similarity=0.9, method="hybrid"),
        _hit("b#0", rank=2, content="INC-2407 oms_overloaded connection pool", similarity=0.5, method="hybrid"),
    ]
    # Give a the higher RRF so a noop would keep a first.
    hits[0] = SearchHit(
        **{**hits[0].__dict__, "rrf_score": 0.05},
    )
    hits[1] = SearchHit(
        **{**hits[1].__dict__, "rrf_score": 0.01},
    )
    reranked = LexicalOverlapReranker().rerank(
        "INC-2407 oms_overloaded",
        hits,
        top_k=2,
    )
    assert reranked[0].chunk_id == "b#0"
    assert NoOpReranker().rerank("q", hits, top_k=2)[0].chunk_id == "a#0"
    assert build_reranker(enabled=False).name == "noop"
    assert build_reranker(enabled=True, name="lexical_overlap").name == "lexical_overlap"


def test_knowledge_search_api_hybrid_mode(
    tmp_path: Path,
    knowledge_root: Path,
    store: ChromaVectorStore,
    embedder: FakeEmbeddingProvider,
    bm25_index: Bm25Index,
):
    ingest_directory(
        knowledge_root,
        store,
        embedder,
        chunk_size_tokens=120,
        overlap_tokens=20,
        bm25_index=bm25_index,
    )
    settings = make_settings(
        tmp_path,
        embedding_dimensions=32,
        embedding_model="fake-embed",
        rag_retrieval_mode="semantic",
    )
    app = create_app(
        settings,
        provider=FakeProvider(),
        embedder=embedder,
        vector_store=store,
        bm25_index=bm25_index,
    )
    with TestClient(app) as client:
        semantic = client.post(
            "/api/knowledge/search",
            json={"query": "INC-2407 oms_overloaded", "top_k": 3, "retrieval_mode": "semantic"},
        )
        hybrid = client.post(
            "/api/knowledge/search",
            json={"query": "INC-2407 oms_overloaded", "top_k": 3, "retrieval_mode": "hybrid"},
        )
    assert semantic.status_code == 200
    assert hybrid.status_code == 200
    assert semantic.json()["retrieval_mode"] == "semantic"
    assert hybrid.json()["retrieval_mode"] == "hybrid"
    assert hybrid.json()["hit_count"] >= 1
    hit = hybrid.json()["hits"][0]
    assert "rrf_score" in hit
    assert hit.get("retrieval_method") in {"semantic", "bm25", "hybrid"}


def test_eval_metric_helpers():
    from scripts.eval_retrieval import mrr_at_k, precision_at_k, recall_at_k

    assert recall_at_k(["a", "b"], ["b"], 5) == 1.0
    assert mrr_at_k(["x", "b", "a"], ["a", "b"], 5) == 0.5
    assert precision_at_k(["a", "x", "b"], ["a", "b"], 3) == pytest.approx(2 / 3)
    assert precision_at_k(["a"], [], 5) is None
