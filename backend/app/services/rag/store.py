"""Persistent ChromaDB vector store for document chunks."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from app.services.rag.models import DocumentChunk, SearchHit

logger = logging.getLogger(__name__)

COLLECTION_METADATA_KEYS = ("embedding_model", "embedding_dimensions")


class VectorStoreError(RuntimeError):
    pass


class ChromaVectorStore:
    """Thin wrapper around a persistent Chroma collection."""

    def __init__(
        self,
        persist_directory: str | Path,
        *,
        collection_name: str = "operatesolve_knowledge",
        embedding_model: str,
        embedding_dimensions: int,
    ) -> None:
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.embedding_dimensions = embedding_dimensions
        self._client = None
        self._collection = None

    def _ensure(self):
        if self._collection is not None:
            return self._collection
        try:
            import chromadb
            from chromadb.config import Settings as ChromaSettings
        except ImportError as exc:
            raise VectorStoreError(
                "chromadb is not installed. Run: pip install -r requirements.txt"
            ) from exc

        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(self.persist_directory),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.embedding_model,
                "embedding_dimensions": self.embedding_dimensions,
            },
        )
        self._validate_collection(self._collection)
        return self._collection

    def _validate_collection(self, collection) -> None:
        meta = dict(collection.metadata or {})
        stored_model = meta.get("embedding_model")
        stored_dims = meta.get("embedding_dimensions")
        if stored_model and stored_model != self.embedding_model:
            raise VectorStoreError(
                f"Collection embedding model mismatch: stored={stored_model!r} "
                f"configured={self.embedding_model!r}. Re-ingest or use a new chroma path."
            )
        if stored_dims is not None and int(stored_dims) != self.embedding_dimensions:
            raise VectorStoreError(
                f"Collection embedding dimensions mismatch: stored={stored_dims} "
                f"configured={self.embedding_dimensions}. Re-ingest or use a new chroma path."
            )

    def count(self) -> int:
        return int(self._ensure().count())

    def document_hashes(self) -> dict[str, str]:
        """Map document_id -> content_hash from stored chunk metadata."""

        collection = self._ensure()
        total = collection.count()
        if total == 0:
            return {}
        result = collection.get(include=["metadatas"])
        hashes: dict[str, str] = {}
        for meta in result.get("metadatas") or []:
            if not meta:
                continue
            doc_id = meta.get("document_id")
            content_hash = meta.get("content_hash")
            if doc_id and content_hash and doc_id not in hashes:
                hashes[str(doc_id)] = str(content_hash)
        return hashes

    def chunk_ids_for_document(self, document_id: str) -> list[str]:
        collection = self._ensure()
        result = collection.get(where={"document_id": document_id}, include=[])
        return list(result.get("ids") or [])

    def delete_document(self, document_id: str) -> int:
        ids = self.chunk_ids_for_document(document_id)
        if not ids:
            return 0
        self._ensure().delete(ids=ids)
        logger.info("vector_delete document_id=%s chunks=%s", document_id, len(ids))
        return len(ids)

    def upsert_chunks(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("Chunk and vector counts must match.")
        if not chunks:
            return
        for vector in vectors:
            if len(vector) != self.embedding_dimensions:
                raise VectorStoreError(
                    f"Vector length {len(vector)} does not match configured "
                    f"dimensions {self.embedding_dimensions}."
                )

        collection = self._ensure()
        collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=vectors,
            documents=[chunk.content for chunk in chunks],
            metadatas=[_chunk_metadata(chunk, self.embedding_model, self.embedding_dimensions) for chunk in chunks],
        )
        logger.info("vector_upsert chunks=%s document_id=%s", len(chunks), chunks[0].document_id)

    def list_chunks(self) -> list[DocumentChunk]:
        """Return every stored chunk with metadata (for BM25 rebuild)."""

        collection = self._ensure()
        total = collection.count()
        if total == 0:
            return []
        result = collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        chunks: list[DocumentChunk] = []
        for chunk_id, document, meta in zip(ids, documents, metadatas, strict=True):
            meta = meta or {}
            chunks.append(
                DocumentChunk(
                    chunk_id=str(chunk_id),
                    document_id=str(meta.get("document_id") or ""),
                    document_title=str(meta.get("document_title") or ""),
                    source=str(meta.get("source") or ""),
                    document_type=str(meta.get("document_type") or ""),
                    section_heading=str(meta.get("section_heading") or ""),
                    content=str(document or ""),
                    chunk_index=int(meta.get("chunk_index") or 0),
                    content_hash=str(meta.get("content_hash") or ""),
                )
            )
        return chunks

    def search(
        self,
        query_vector: list[float],
        *,
        top_k: int = 5,
    ) -> list[SearchHit]:
        if top_k < 1:
            raise VectorStoreError("top_k must be at least 1.")
        if len(query_vector) != self.embedding_dimensions:
            raise VectorStoreError(
                f"Query vector length {len(query_vector)} does not match "
                f"configured dimensions {self.embedding_dimensions}."
            )

        collection = self._ensure()
        if collection.count() == 0:
            return []

        result = collection.query(
            query_embeddings=[query_vector],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]

        hits: list[SearchHit] = []
        for rank, (chunk_id, document, meta, distance) in enumerate(
            zip(ids, documents, metadatas, distances, strict=True),
            start=1,
        ):
            meta = meta or {}
            dist = float(distance)
            # Cosine space in Chroma: distance is 1 - cosine_similarity.
            similarity = 1.0 - dist
            hits.append(
                SearchHit(
                    chunk_id=str(chunk_id),
                    document_id=str(meta.get("document_id") or ""),
                    document_title=str(meta.get("document_title") or ""),
                    source=str(meta.get("source") or ""),
                    document_type=str(meta.get("document_type") or ""),
                    section_heading=str(meta.get("section_heading") or ""),
                    content=str(document or ""),
                    distance=dist,
                    similarity=similarity,
                    rank=rank,
                )
            )
        return hits


def _chunk_metadata(chunk: DocumentChunk, model: str, dimensions: int) -> dict[str, Any]:
    # Chroma metadata values must be str | int | float | bool.
    return {
        "document_id": chunk.document_id,
        "document_title": chunk.document_title,
        "source": chunk.source,
        "document_type": chunk.document_type,
        "section_heading": chunk.section_heading,
        "content_hash": chunk.content_hash,
        "chunk_index": chunk.chunk_index,
        "embedding_model": model,
        "embedding_dimensions": dimensions,
    }
