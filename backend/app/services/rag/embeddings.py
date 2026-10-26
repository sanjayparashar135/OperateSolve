"""Embedding providers for RAG.

Usage is tracked separately from chat generation via purpose=embedding.
"""

from __future__ import annotations

import hashlib
import logging
import math
import time
import uuid
from dataclasses import dataclass
from typing import Protocol

from openai import AsyncOpenAI, OpenAI

from app.config import Settings
from app.db.database import utc_now
from app.db.repository import Repository
from app.services.rag.models import EmbeddingResult

logger = logging.getLogger(__name__)

# OpenAI allows up to 2048 inputs; keep batches smaller for token headroom.
DEFAULT_BATCH_SIZE = 64
MAX_BATCH_SIZE = 2048


class EmbeddingError(RuntimeError):
    """Provider or validation failure while embedding text."""


class EmbeddingProvider(Protocol):
    model: str
    dimensions: int

    async def embed_texts(self, texts: list[str]) -> EmbeddingResult: ...

    def embed_texts_sync(self, texts: list[str]) -> EmbeddingResult: ...


@dataclass
class _UsageRecorder:
    repository: Repository | None
    provider_name: str = "openai"

    def record(
        self,
        *,
        model: str,
        status: str,
        input_tokens: int | None,
        duration_ms: int,
        request_id: str,
        started_at: str,
        finished_at: str,
    ) -> None:
        if self.repository is None:
            return
        self.repository.record_usage(
            request_id=request_id,
            conversation_id=None,
            message_id=None,
            provider=self.provider_name,
            requested_model=model,
            model=model,
            status=status,
            input_tokens=input_tokens,
            output_tokens=0 if input_tokens is not None else None,
            cached_input_tokens=None,
            cache_write_tokens=None,
            reasoning_tokens=None,
            duration_ms=duration_ms,
            started_at=started_at,
            finished_at=finished_at,
            provider_response_id=None,
            purpose="embedding",
        )


class OpenAIEmbeddingProvider:
    """OpenAI Embeddings API with batching."""

    def __init__(
        self,
        settings: Settings,
        *,
        repository: Repository | None = None,
        batch_size: int | None = None,
    ) -> None:
        self.settings = settings
        self.model = settings.embedding_model
        self.dimensions = settings.embedding_dimensions
        self.batch_size = min(
            max(batch_size or settings.embedding_batch_size, 1),
            MAX_BATCH_SIZE,
        )
        self._async_client: AsyncOpenAI | None = None
        self._sync_client: OpenAI | None = None
        self._usage = _UsageRecorder(repository)

    def _get_async(self) -> AsyncOpenAI:
        if self._async_client is None:
            if not self.settings.openai_api_key.strip():
                raise EmbeddingError("OPENAI_API_KEY is not configured.")
            self._async_client = AsyncOpenAI(api_key=self.settings.openai_api_key)
        return self._async_client

    def _get_sync(self) -> OpenAI:
        if self._sync_client is None:
            if not self.settings.openai_api_key.strip():
                raise EmbeddingError("OPENAI_API_KEY is not configured.")
            self._sync_client = OpenAI(api_key=self.settings.openai_api_key)
        return self._sync_client

    async def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        return await self._embed(texts, sync=False)

    def embed_texts_sync(self, texts: list[str]) -> EmbeddingResult:
        return self._run_sync(texts)

    def _run_sync(self, texts: list[str]) -> EmbeddingResult:
        cleaned = _validate_inputs(texts)
        if not cleaned:
            return EmbeddingResult(vectors=[], model=self.model, dimensions=self.dimensions, input_tokens=0)

        all_vectors: list[list[float]] = []
        total_tokens = 0
        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start : start + self.batch_size]
            request_id = str(uuid.uuid4())
            started_at = utc_now()
            t0 = time.perf_counter()
            try:
                response = self._get_sync().embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                )
            except Exception as exc:
                finished_at = utc_now()
                self._usage.record(
                    model=self.model,
                    status="error",
                    input_tokens=None,
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                    request_id=request_id,
                    started_at=started_at,
                    finished_at=finished_at,
                )
                raise EmbeddingError(f"Embedding request failed: {exc}") from exc

            finished_at = utc_now()
            duration_ms = int((time.perf_counter() - t0) * 1000)
            vectors = _ordered_vectors(response.data, expected=len(batch), dimensions=self.dimensions)
            tokens = _prompt_tokens(response)
            total_tokens += tokens or 0
            self._usage.record(
                model=getattr(response, "model", None) or self.model,
                status="complete",
                input_tokens=tokens,
                duration_ms=duration_ms,
                request_id=request_id,
                started_at=started_at,
                finished_at=finished_at,
            )
            all_vectors.extend(vectors)

        return EmbeddingResult(
            vectors=all_vectors,
            model=self.model,
            dimensions=self.dimensions,
            input_tokens=total_tokens if total_tokens else None,
        )

    async def _embed(self, texts: list[str], *, sync: bool) -> EmbeddingResult:
        del sync
        cleaned = _validate_inputs(texts)
        if not cleaned:
            return EmbeddingResult(vectors=[], model=self.model, dimensions=self.dimensions, input_tokens=0)

        all_vectors: list[list[float]] = []
        total_tokens = 0
        for start in range(0, len(cleaned), self.batch_size):
            batch = cleaned[start : start + self.batch_size]
            request_id = str(uuid.uuid4())
            started_at = utc_now()
            t0 = time.perf_counter()
            try:
                response = await self._get_async().embeddings.create(
                    model=self.model,
                    input=batch,
                    dimensions=self.dimensions,
                )
            except Exception as exc:
                finished_at = utc_now()
                self._usage.record(
                    model=self.model,
                    status="error",
                    input_tokens=None,
                    duration_ms=int((time.perf_counter() - t0) * 1000),
                    request_id=request_id,
                    started_at=started_at,
                    finished_at=finished_at,
                )
                raise EmbeddingError(f"Embedding request failed: {exc}") from exc

            finished_at = utc_now()
            duration_ms = int((time.perf_counter() - t0) * 1000)
            vectors = _ordered_vectors(response.data, expected=len(batch), dimensions=self.dimensions)
            tokens = _prompt_tokens(response)
            total_tokens += tokens or 0
            self._usage.record(
                model=getattr(response, "model", None) or self.model,
                status="complete",
                input_tokens=tokens,
                duration_ms=duration_ms,
                request_id=request_id,
                started_at=started_at,
                finished_at=finished_at,
            )
            all_vectors.extend(vectors)

        return EmbeddingResult(
            vectors=all_vectors,
            model=self.model,
            dimensions=self.dimensions,
            input_tokens=total_tokens if total_tokens else None,
        )


class FakeEmbeddingProvider:
    """Deterministic embeddings for tests. No network calls."""

    def __init__(self, *, dimensions: int = 32, model: str = "fake-embed") -> None:
        self.dimensions = dimensions
        self.model = model
        self.fail = False
        self.calls: list[list[str]] = []

    async def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        return self.embed_texts_sync(texts)

    def embed_texts_sync(self, texts: list[str]) -> EmbeddingResult:
        cleaned = _validate_inputs(texts)
        self.calls.append(list(cleaned))
        if self.fail:
            raise EmbeddingError("Simulated embedding failure.")
        vectors = [_hash_embedding(text, self.dimensions) for text in cleaned]
        # Rough token estimate for usage-shaped tests.
        tokens = sum(max(1, len(text) // 4) for text in cleaned)
        return EmbeddingResult(
            vectors=vectors,
            model=self.model,
            dimensions=self.dimensions,
            input_tokens=tokens,
            request_id=str(uuid.uuid4()),
        )


def _validate_inputs(texts: list[str]) -> list[str]:
    cleaned: list[str] = []
    for text in texts:
        value = (text or "").strip()
        if not value:
            raise EmbeddingError("Cannot embed empty text.")
        cleaned.append(value)
    return cleaned


def _ordered_vectors(data: list, *, expected: int, dimensions: int) -> list[list[float]]:
    if len(data) != expected:
        raise EmbeddingError(f"Expected {expected} embeddings, got {len(data)}.")
    ordered = sorted(data, key=lambda item: item.index)
    vectors: list[list[float]] = []
    for item in ordered:
        vector = list(item.embedding)
        if len(vector) != dimensions:
            raise EmbeddingError(
                f"Embedding dimension mismatch: expected {dimensions}, got {len(vector)}."
            )
        vectors.append(vector)
    return vectors


def _prompt_tokens(response: object) -> int | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    value = getattr(usage, "prompt_tokens", None)
    return int(value) if value is not None else None


def _hash_embedding(text: str, dimensions: int) -> list[float]:
    """Bag a unit vector from token hashes so shared words increase similarity."""

    vector = [0.0] * dimensions
    tokens = [t for t in text.lower().split() if t]
    if not tokens:
        tokens = ["empty"]
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        for i in range(dimensions):
            # Signed byte contribution
            raw = digest[i % len(digest)]
            vector[i] += (raw / 255.0) * 2.0 - 1.0
    norm = math.sqrt(sum(v * v for v in vector)) or 1.0
    return [v / norm for v in vector]
