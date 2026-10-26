"""Build embedding, vector store, BM25 index, and reranker from Settings."""

from __future__ import annotations

from pathlib import Path

from app.config import BACKEND_DIR, Settings
from app.db.repository import Repository
from app.services.rag.bm25 import Bm25Index
from app.services.rag.embeddings import EmbeddingProvider, FakeEmbeddingProvider, OpenAIEmbeddingProvider
from app.services.rag.rerank import Reranker, build_reranker
from app.services.rag.store import ChromaVectorStore


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else BACKEND_DIR / path


def build_embedder(
    settings: Settings,
    *,
    repository: Repository | None = None,
    override: EmbeddingProvider | None = None,
) -> EmbeddingProvider:
    if override is not None:
        return override
    return OpenAIEmbeddingProvider(settings, repository=repository)


def build_vector_store(settings: Settings) -> ChromaVectorStore:
    return ChromaVectorStore(
        resolve_path(settings.chroma_persist_directory),
        collection_name=settings.chroma_collection_name,
        embedding_model=settings.embedding_model,
        embedding_dimensions=settings.embedding_dimensions,
    )


def build_bm25_index(settings: Settings, *, load: bool = True) -> Bm25Index:
    index = Bm25Index(resolve_path(settings.bm25_index_path))
    if load:
        index.load()
    return index


def build_rag_reranker(settings: Settings) -> Reranker:
    return build_reranker(
        enabled=settings.rag_reranker_enabled,
        name=settings.rag_reranker,
    )


def build_fake_embedder(settings: Settings) -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(
        dimensions=settings.embedding_dimensions,
        model="fake-embed",
    )
