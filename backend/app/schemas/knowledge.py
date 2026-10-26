"""Knowledge search and grounded answer schemas (Phase 3.1–3.3).

GroundedAnswerPayload is the schema enforced by OpenAI Structured Outputs.
Citation validation against retrieved source IDs checks reference integrity;
it does not prove that every factual claim is grounded in those sources.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

RetrievalMode = Literal["semantic", "hybrid"]


class KnowledgeSearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(5, ge=1, le=50)
    retrieval_mode: RetrievalMode | None = Field(
        None,
        description="semantic (baseline) or hybrid (semantic + BM25 + RRF). Defaults to server config.",
    )


class KnowledgeSearchHit(BaseModel):
    chunk_id: str
    document_id: str
    document_title: str
    source: str
    document_type: str
    section_heading: str
    content: str
    # Cosine distance from Chroma (lower is closer). similarity = 1 - distance.
    # For BM25-only hits, distance=1.0 and similarity=0.0 (no semantic score).
    distance: float
    similarity: float
    rank: int
    bm25_score: float | None = None
    rrf_score: float | None = None
    semantic_rank: int | None = None
    bm25_rank: int | None = None
    retrieval_method: str = "semantic"


class KnowledgeSearchResponse(BaseModel):
    query: str
    top_k: int
    embedding_model: str
    hit_count: int
    hits: list[KnowledgeSearchHit]
    retrieval_mode: str = "semantic"
    reranker: str = "noop"
    semantic_candidates: int = 0
    bm25_candidates: int = 0
    retrieval_latency_ms: int | None = None


class KnowledgeAskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=4000)
    top_k: int = Field(5, ge=1, le=50)
    retrieval_mode: RetrievalMode | None = Field(
        None,
        description="semantic (baseline) or hybrid (semantic + BM25 + RRF). Defaults to server config.",
    )

    @field_validator("question")
    @classmethod
    def strip_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question cannot be blank")
        return cleaned


class GroundedAnswerPayload(BaseModel):
    """Schema-enforced LLM output for a grounded knowledge answer."""

    answer: str = Field(
        ...,
        min_length=1,
        max_length=12000,
        description=(
            "Markdown answer. Cite supporting sources inline as [S1], [S2], … "
            "using only the supplied source labels. If evidence is insufficient, "
            "say so clearly and set insufficient_context to true."
        ),
    )
    cited_sources: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Source labels actually used, e.g. S1, S2. Never invent labels.",
    )
    insufficient_context: bool = Field(
        ...,
        description=(
            "True when retrieved material does not adequately support a confident answer."
        ),
    )

    @field_validator("answer")
    @classmethod
    def strip_answer(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Answer cannot be blank")
        return cleaned

    @field_validator("cited_sources")
    @classmethod
    def normalize_cited_sources(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()
        for item in value:
            label = (item or "").strip().upper()
            if label.startswith("[") and label.endswith("]"):
                label = label[1:-1].strip().upper()
            if not label or label in seen:
                continue
            seen.add(label)
            cleaned.append(label)
        return cleaned


class KnowledgeCitation(BaseModel):
    """One validated source citation returned to the client."""

    ref: str = Field(..., description="Stable label such as S1")
    chunk_id: str
    document_id: str
    document_title: str
    source: str
    document_type: str
    section_heading: str
    content: str
    similarity: float
    rank: int
    bm25_score: float | None = None
    rrf_score: float | None = None
    retrieval_method: str = "semantic"


class KnowledgeAskResponse(BaseModel):
    question: str
    answer: str
    citations: list[KnowledgeCitation]
    insufficient_context: bool
    retrieved_count: int
    selected_count: int
    top_k: int
    model: str | None
    embedding_model: str
    request_id: str
    retrieval_latency_ms: int
    generation_latency_ms: int | None
    latency_ms: int
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None
    estimated_cost_usd: float | None = None
    # Embedding usage for this ask is recorded separately (purpose=embedding).
    embedding_tokens: int | None = None
    retrieval_mode: str = "semantic"
    reranker: str = "noop"
    citation_note: str = (
        "Citation validation checks that references match retrieved source IDs; "
        "it does not prove factual grounding of every claim."
    )
