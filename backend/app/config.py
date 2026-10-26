"""Runtime configuration. Secrets stay in the backend environment."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    # Baseline model: the Auto fallback. Must be a model in the registry.
    openai_model: str = "gpt-6-luna"
    openai_reasoning_effort: str = "low"
    openai_max_output_tokens: int = 2048
    # Comma-separated registry keys or model ids to turn off, e.g. "mini,gpt-5-nano".
    disabled_models: str = ""
    # At startup, ask the provider which registered models this key can use (no tokens billed).
    model_access_check: bool = True
    database_path: str = "data/operatesolve.db"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    # Local Swagger/ReDoc/OpenAPI. Keep true for local exploration; false for demos.
    expose_api_docs: bool = True
    log_level: str = "INFO"
    llm_provider: str = "openai"
    # Relative paths resolve from the backend directory.
    pricing_config_path: str = "pricing.json"

    # Phase 2.4 — context budgeting (conservative vs full model windows).
    # Soft input cap for chat prompts; still clamped per model max_input / context.
    context_max_input_tokens: int = 24_000
    # Multiply local token estimates when fitting history (heuristic is not exact).
    context_token_safety_margin: float = 1.15
    # Reserved inside the context window for assistant output + reasoning headroom.
    context_reasoning_reserve_tokens: int = 1_024

    # Phase 2.4 — conversation summarization (separate from the chat model).
    summarization_enabled: bool = True
    # Default to the current non-deprecated efficient model. Override if needed.
    summarization_model: str = "gpt-6-luna"
    summarization_max_output_tokens: int = 400
    summarization_reasoning_effort: str = "low"
    # Only call the summarizer when newly excluded history reaches this estimate.
    summarization_trigger_excluded_tokens: int = 800
    # Soft cap for the injected summary's estimated size when budgeting history.
    summarization_summary_budget_tokens: int = 500

    # Phase 2.5B — sequential vs parallel LLM experiment (learning only).
    parallel_experiment_timeout_seconds: float = 60.0
    parallel_experiment_max_concurrency: int = 3
    parallel_experiment_max_output_tokens: int = 400

    # Phase 3.1 — RAG foundation (embeddings + ChromaDB).
    knowledge_source_dir: str = "data/knowledge/source"
    chroma_persist_directory: str = "data/knowledge/chroma"
    chroma_collection_name: str = "operatesolve_knowledge"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 64
    rag_chunk_size_tokens: int = 500
    rag_chunk_overlap_tokens: int = 75

    # Phase 3.2 — grounded RAG answer generation (one LLM call per question).
    # Empty string uses OPENAI_MODEL. Must support structured_outputs.
    rag_answer_model: str = ""
    rag_answer_max_output_tokens: int = 1200
    rag_answer_reasoning_effort: str = "low"
    rag_answer_timeout_seconds: float = 60.0
    # Soft cap on selected chunk text in the prompt (local estimate, not billing).
    rag_context_token_budget: int = 3500
    # Drop retrieved chunks below this cosine similarity before generation.
    # 0.0 disables filtering (default); raise only after retrieval eval.
    rag_min_similarity: float = 0.0

    # Phase 3.3 — hybrid retrieval (semantic + BM25 + RRF) and optional rerank.
    # semantic = Phase 3.1/3.2 baseline; hybrid = semantic + BM25 fused with RRF.
    # Keep semantic as default until hybrid is measured better on the eval set.
    rag_retrieval_mode: str = "semantic"
    rag_semantic_candidates: int = 20
    rag_bm25_candidates: int = 20
    rag_rrf_k: int = 60
    rag_reranker_enabled: bool = False
    # lexical_overlap = lightweight query-dependent local reranker (no extra API).
    rag_reranker: str = "lexical_overlap"
    bm25_index_path: str = "data/knowledge/bm25/index.json"

    @property
    def pricing_file(self) -> Path:
        path = Path(self.pricing_config_path)
        return path if path.is_absolute() else BACKEND_DIR / path

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def disabled_model_list(self) -> list[str]:
        return [item.strip() for item in self.disabled_models.split(",") if item.strip()]

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_api_key.strip())

    @property
    def knowledge_source_path(self) -> Path:
        path = Path(self.knowledge_source_dir)
        return path if path.is_absolute() else BACKEND_DIR / path

    @property
    def chroma_path(self) -> Path:
        path = Path(self.chroma_persist_directory)
        return path if path.is_absolute() else BACKEND_DIR / path

    @property
    def bm25_path(self) -> Path:
        path = Path(self.bm25_index_path)
        return path if path.is_absolute() else BACKEND_DIR / path


@lru_cache
def get_settings() -> Settings:
    return Settings()
