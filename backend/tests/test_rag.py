"""Phase 3.1 RAG foundation tests (no live OpenAI calls)."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from app.services.rag.chunking import chunk_document
from app.services.rag.embeddings import FakeEmbeddingProvider
from app.services.rag.ingest import ingest_directory
from app.services.rag.loader import DocumentLoadError, content_hash, load_markdown_directory, load_markdown_file
from app.services.rag.models import LoadedDocument
from app.services.rag.retrieve import RetrievalError, semantic_search
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

## Recovery procedure

Do not scale OMS replicas first. Terminate idle in transaction sessions, then reduce pool size.

## Prevention

Keep replicas times pool size under seventy percent of max_connections.
""",
        encoding="utf-8",
    )
    (root / "architecture.md").write_text(
        """# Northstar Architecture Overview

**Document type:** architecture

## Core services

Order Management API depends on Payment Gateway Adapter and Inventory Reservation Service.

## Kafka topics

order.confirmed is consumed by Shipping Orchestrator and Notification Hub.
""",
        encoding="utf-8",
    )
    return root


@pytest.fixture
def store(tmp_path: Path) -> ChromaVectorStore:
    return ChromaVectorStore(
        tmp_path / "chroma",
        collection_name="test_knowledge",
        embedding_model="fake-embed",
        embedding_dimensions=32,
    )


@pytest.fixture
def embedder() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(dimensions=32)


def test_load_markdown_metadata_and_stable_ids(knowledge_root: Path):
    docs = load_markdown_directory(knowledge_root)
    assert len(docs) == 2
    by_id = {doc.document_id: doc for doc in docs}
    assert "oms-runbook" in by_id
    assert by_id["oms-runbook"].title.startswith("OMS Database")
    assert by_id["oms-runbook"].document_type == "runbook"
    assert by_id["architecture"].document_type == "architecture"
    again = load_markdown_file(knowledge_root / "oms-runbook.md", knowledge_root)
    assert again.document_id == by_id["oms-runbook"].document_id
    assert again.content_hash == by_id["oms-runbook"].content_hash


def test_empty_and_malformed_documents(tmp_path: Path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "empty.md").write_text("   \n", encoding="utf-8")
    with pytest.raises(DocumentLoadError, match="empty"):
        load_markdown_file(root / "empty.md", root)

    (root / "ok.md").write_text("# Title\n\nBody text for a valid document.\n", encoding="utf-8")
    (root / "empty2.md").write_text("", encoding="utf-8")
    docs = load_markdown_directory(root)
    assert [doc.document_id for doc in docs] == ["ok"]


def test_chunking_overlap_and_stable_ids():
    body = "\n\n".join(f"Paragraph {i} with enough words to estimate several tokens." for i in range(40))
    doc = LoadedDocument(
        document_id="long-doc",
        title="Long Doc",
        source="long-doc.md",
        document_type="general",
        content=f"# Long Doc\n\n## Section A\n\n{body}",
        content_hash=content_hash("x"),
    )
    chunks = chunk_document(doc, chunk_size_tokens=80, overlap_tokens=20)
    assert len(chunks) >= 2
    assert chunks[0].chunk_id == "long-doc#0000"
    assert chunks[1].chunk_id == "long-doc#0001"
    assert all(chunk.section_heading for chunk in chunks)
    assert all(chunk.document_title == "Long Doc" for chunk in chunks)
    # Overlap: later chunk text should share a suffix/prefix region with the prior body.
    assert any(
        token in chunks[1].content
        for token in chunks[0].content.split()[-15:]
        if len(token) > 4
    )


def test_ingest_idempotent_and_skip_unchanged(knowledge_root: Path, store: ChromaVectorStore, embedder: FakeEmbeddingProvider):
    report1 = ingest_directory(knowledge_root, store, embedder, chunk_size_tokens=120, overlap_tokens=20)
    assert report1.documents_indexed == 2
    assert report1.chunks_upserted > 0
    assert store.count() == report1.chunks_upserted
    calls_after_first = len(embedder.calls)

    report2 = ingest_directory(knowledge_root, store, embedder, chunk_size_tokens=120, overlap_tokens=20)
    assert report2.documents_indexed == 0
    assert report2.documents_skipped == 2
    assert len(embedder.calls) == calls_after_first


def test_changed_document_reindexes_and_removes_stale_chunks(
    knowledge_root: Path,
    store: ChromaVectorStore,
    embedder: FakeEmbeddingProvider,
):
    ingest_directory(knowledge_root, store, embedder, chunk_size_tokens=120, overlap_tokens=20)
    before = store.count()
    old_ids = set(store.chunk_ids_for_document("oms-runbook"))
    assert old_ids

    path = knowledge_root / "oms-runbook.md"
    path.write_text(
        path.read_text(encoding="utf-8")
        + "\n\n## Extra section\n\nNew recovery tip about connection budgets and HikariCP.\n",
        encoding="utf-8",
    )
    report = ingest_directory(knowledge_root, store, embedder, chunk_size_tokens=120, overlap_tokens=20)
    assert report.documents_indexed == 1
    assert report.chunks_deleted >= len(old_ids)
    new_ids = set(store.chunk_ids_for_document("oms-runbook"))
    assert new_ids
    assert new_ids != old_ids or store.count() != before


def test_deleted_source_prunes_chunks(knowledge_root: Path, store: ChromaVectorStore, embedder: FakeEmbeddingProvider):
    ingest_directory(knowledge_root, store, embedder, chunk_size_tokens=120, overlap_tokens=20)
    assert store.chunk_ids_for_document("architecture")
    (knowledge_root / "architecture.md").unlink()
    report = ingest_directory(knowledge_root, store, embedder, chunk_size_tokens=120, overlap_tokens=20)
    assert report.documents_removed == 1
    assert store.chunk_ids_for_document("architecture") == []


def test_embedding_failure_surfaces(knowledge_root: Path, store: ChromaVectorStore, embedder: FakeEmbeddingProvider):
    from app.services.rag.ingest import IngestionError

    embedder.fail = True
    with pytest.raises(IngestionError):
        ingest_directory(knowledge_root, store, embedder, chunk_size_tokens=120, overlap_tokens=20)
    assert store.count() == 0


@pytest.mark.asyncio
async def test_semantic_search_metadata_and_ranking(
    knowledge_root: Path,
    store: ChromaVectorStore,
    embedder: FakeEmbeddingProvider,
):
    ingest_directory(knowledge_root, store, embedder, chunk_size_tokens=120, overlap_tokens=20)
    hits = await semantic_search(
        "OMS database connection exhaustion recovery without scaling replicas",
        store=store,
        embedder=embedder,
        top_k=3,
    )
    assert hits
    assert hits[0].rank == 1
    assert hits[0].chunk_id
    assert hits[0].document_id
    assert hits[0].document_title
    assert hits[0].section_heading is not None
    assert isinstance(hits[0].distance, float)
    assert isinstance(hits[0].similarity, float)
    assert hits[0].document_id == "oms-runbook"


@pytest.mark.asyncio
async def test_empty_knowledge_base_search(store: ChromaVectorStore, embedder: FakeEmbeddingProvider):
    hits = await semantic_search("anything", store=store, embedder=embedder, top_k=5)
    assert hits == []


@pytest.mark.asyncio
async def test_search_rejects_empty_query(store: ChromaVectorStore, embedder: FakeEmbeddingProvider):
    with pytest.raises(RetrievalError) as exc:
        await semantic_search("  ", store=store, embedder=embedder, top_k=5)
    assert exc.value.status_code == 400


def test_knowledge_search_api(tmp_path: Path, knowledge_root: Path, store: ChromaVectorStore, embedder: FakeEmbeddingProvider):
    ingest_directory(knowledge_root, store, embedder, chunk_size_tokens=120, overlap_tokens=20)
    settings = make_settings(
        tmp_path,
        embedding_dimensions=32,
        embedding_model="fake-embed",
        chroma_persist_directory=str(tmp_path / "chroma-api"),
    )
    # Point the app at the already-populated store/embedder.
    app = create_app(settings, provider=FakeProvider(), embedder=embedder, vector_store=store)
    with TestClient(app) as client:
        response = client.post(
            "/api/knowledge/search",
            json={"query": "database connection exhaustion OMS", "top_k": 3},
        )
    assert response.status_code == 200
    body = response.json()
    assert body["hit_count"] >= 1
    assert body["embedding_model"] == "fake-embed"
    hit = body["hits"][0]
    assert hit["document_id"]
    assert "distance" in hit
    assert "similarity" in hit
    assert hit["chunk_id"]


def test_knowledge_search_api_empty_store(tmp_path: Path):
    settings = make_settings(tmp_path, embedding_dimensions=32, embedding_model="fake-embed")
    store = ChromaVectorStore(
        tmp_path / "empty-chroma",
        collection_name="empty",
        embedding_model="fake-embed",
        embedding_dimensions=32,
    )
    embedder = FakeEmbeddingProvider(dimensions=32)
    app = create_app(settings, provider=FakeProvider(), embedder=embedder, vector_store=store)
    with TestClient(app) as client:
        response = client.post("/api/knowledge/search", json={"query": "no docs yet", "top_k": 5})
    assert response.status_code == 200
    assert response.json()["hit_count"] == 0
    assert response.json()["hits"] == []


@pytest.mark.asyncio
async def test_fake_embedder_failure_on_search(store: ChromaVectorStore):
    # Empty store short-circuits before embed — seed one chunk first.
    working = FakeEmbeddingProvider(dimensions=32)
    doc = LoadedDocument(
        document_id="seed",
        title="Seed",
        source="seed.md",
        document_type="general",
        content="# Seed\n\nConnection pool text.",
        content_hash="abc",
    )
    chunks = chunk_document(doc, chunk_size_tokens=100, overlap_tokens=10)
    vectors = working.embed_texts_sync([c.content for c in chunks]).vectors
    store.upsert_chunks(chunks, vectors)

    embedder = FakeEmbeddingProvider(dimensions=32)
    embedder.fail = True
    with pytest.raises(RetrievalError) as exc:
        await semantic_search("query", store=store, embedder=embedder, top_k=3)
    assert exc.value.code == "embedding_failed"


def test_content_hash_changes_with_edits():
    a = content_hash("hello")
    b = content_hash("hello")
    c = content_hash("hello!")
    assert a == b
    assert a != c

def test_recall_at_k_helper():
    from scripts.eval_retrieval import recall_at_k

    assert recall_at_k(["a", "b", "c"], ["c"], 5) == 1.0
    assert recall_at_k(["a", "b"], ["c"], 5) == 0.0
    assert recall_at_k(["a", "b", "c"], ["c"], 2) == 0.0
