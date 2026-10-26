"""Shared RAG data types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class LoadedDocument:
    """A source document ready for chunking."""

    document_id: str
    title: str
    source: str
    document_type: str
    content: str
    content_hash: str


@dataclass(frozen=True)
class DocumentChunk:
    """One embeddable text unit with provenance metadata."""

    chunk_id: str
    document_id: str
    document_title: str
    source: str
    document_type: str
    section_heading: str
    content: str
    chunk_index: int
    content_hash: str
    token_estimate: int = 0


@dataclass(frozen=True)
class EmbeddingResult:
    """Vectors plus optional provider usage for one batch."""

    vectors: list[list[float]]
    model: str
    dimensions: int
    input_tokens: int | None = None
    request_id: str | None = None


@dataclass(frozen=True)
class SearchHit:
    """One ranked chunk from semantic, BM25, or hybrid retrieval.

    Score fields are intentionally distinct:
    - distance / similarity: cosine distance and similarity from the vector store
      (semantic path). Absent semantic participation uses distance=1.0, similarity=0.0.
    - bm25_score: keyword relevance from BM25 (higher is better).
    - rrf_score: Reciprocal Rank Fusion score after combining ranked lists.
    """

    chunk_id: str
    document_id: str
    document_title: str
    source: str
    document_type: str
    section_heading: str
    content: str
    distance: float
    similarity: float
    rank: int
    bm25_score: float | None = None
    rrf_score: float | None = None
    semantic_rank: int | None = None
    bm25_rank: int | None = None
    # semantic | bm25 | hybrid — how this chunk entered the result set.
    retrieval_method: str = "semantic"


@dataclass
class IngestReport:
    """Summary of an ingestion run."""

    documents_seen: int = 0
    documents_indexed: int = 0
    documents_skipped: int = 0
    documents_removed: int = 0
    chunks_upserted: int = 0
    chunks_deleted: int = 0
    embedding_tokens: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "documents_seen": self.documents_seen,
            "documents_indexed": self.documents_indexed,
            "documents_skipped": self.documents_skipped,
            "documents_removed": self.documents_removed,
            "chunks_upserted": self.chunks_upserted,
            "chunks_deleted": self.chunks_deleted,
            "embedding_tokens": self.embedding_tokens,
            "errors": list(self.errors),
        }
