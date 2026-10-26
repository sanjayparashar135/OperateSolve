"""Local BM25 keyword index over the same chunks stored in Chroma.

The index is a JSON sidecar (no extra database). It must be rebuilt or refreshed
whenever ingestion upserts or deletes chunks so keyword search stays in sync.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.services.rag.models import DocumentChunk, SearchHit
from app.services.rag.store import ChromaVectorStore

logger = logging.getLogger(__name__)

TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_./+-]*", re.IGNORECASE)


class Bm25IndexError(RuntimeError):
    pass


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenizer; keeps ids like INC-2407 and oms_overloaded."""

    return [m.group(0).lower() for m in TOKEN_RE.finditer(text or "")]


@dataclass(frozen=True)
class Bm25ChunkRecord:
    chunk_id: str
    document_id: str
    document_title: str
    source: str
    document_type: str
    section_heading: str
    content: str
    content_hash: str
    chunk_index: int = 0


class Bm25Index:
    """In-memory BM25 over persisted chunk records."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._records: list[Bm25ChunkRecord] = []
        self._tokenized: list[list[str]] = []
        self._by_id: dict[str, int] = {}
        self._bm25 = None
        self._stale = True

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def ready(self) -> bool:
        return self._bm25 is not None and len(self._records) > 0 and not self._stale

    def clear(self) -> None:
        self._records = []
        self._tokenized = []
        self._by_id = {}
        self._bm25 = None
        self._stale = True

    def rebuild(self, chunks: list[DocumentChunk] | list[Bm25ChunkRecord]) -> None:
        records: list[Bm25ChunkRecord] = []
        for chunk in chunks:
            if isinstance(chunk, Bm25ChunkRecord):
                records.append(chunk)
            else:
                records.append(
                    Bm25ChunkRecord(
                        chunk_id=chunk.chunk_id,
                        document_id=chunk.document_id,
                        document_title=chunk.document_title,
                        source=chunk.source,
                        document_type=chunk.document_type,
                        section_heading=chunk.section_heading,
                        content=chunk.content,
                        content_hash=chunk.content_hash,
                        chunk_index=chunk.chunk_index,
                    )
                )
        self._records = records
        self._tokenized = [tokenize(record.content) for record in records]
        self._by_id = {record.chunk_id: i for i, record in enumerate(records)}
        self._bm25 = self._build_model(self._tokenized)
        self._stale = False
        logger.info("bm25_rebuilt chunks=%s path=%s", len(records), self.path)

    def rebuild_from_store(self, store: ChromaVectorStore) -> None:
        self.rebuild(store.list_chunks())
        self.save()

    def upsert_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        # Incremental: drop existing ids then append; rebuild BM25 model once.
        keep = [r for r in self._records if r.chunk_id not in {c.chunk_id for c in chunks}]
        for chunk in chunks:
            keep.append(
                Bm25ChunkRecord(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    document_title=chunk.document_title,
                    source=chunk.source,
                    document_type=chunk.document_type,
                    section_heading=chunk.section_heading,
                    content=chunk.content,
                    content_hash=chunk.content_hash,
                    chunk_index=chunk.chunk_index,
                )
            )
        self.rebuild(keep)

    def delete_chunk_ids(self, chunk_ids: list[str]) -> int:
        if not chunk_ids:
            return 0
        remove = set(chunk_ids)
        before = len(self._records)
        kept = [r for r in self._records if r.chunk_id not in remove]
        deleted = before - len(kept)
        if deleted:
            self.rebuild(kept)
        return deleted

    def delete_document(self, document_id: str) -> int:
        before = len(self._records)
        kept = [r for r in self._records if r.document_id != document_id]
        deleted = before - len(kept)
        if deleted:
            self.rebuild(kept)
        return deleted

    def search(self, query: str, *, top_k: int = 5) -> list[SearchHit]:
        cleaned = (query or "").strip()
        if not cleaned:
            return []
        if top_k < 1:
            raise Bm25IndexError("top_k must be at least 1.")
        if self._bm25 is None or not self._records:
            logger.warning("bm25_search_skipped reason=empty_or_missing_index")
            return []
        if self._stale:
            logger.warning("bm25_search_skipped reason=stale_index")
            return []

        tokens = tokenize(cleaned)
        if not tokens:
            return []

        scores = [float(s) for s in self._bm25.get_scores(tokens)]
        # rank_bm25 IDF is log((N-n+0.5)/(n+0.5)). On tiny corpora this can be
        # exactly 0 for terms that appear in every (or half of) docs, collapsing
        # all scores to 0. Fall back to query-term hit counts so exact keywords
        # still retrieve (important for incident IDs / error codes).
        if not scores or max(scores) <= 0.0:
            scores = [
                float(sum(1 for token in tokens if token in doc_tokens))
                for doc_tokens in self._tokenized
            ]
        ranked = sorted(
            ((score, idx) for idx, score in enumerate(scores) if score > 0.0),
            key=lambda item: item[0],
            reverse=True,
        )[:top_k]

        hits: list[SearchHit] = []
        for rank, (score, idx) in enumerate(ranked, start=1):
            record = self._records[idx]
            hits.append(
                SearchHit(
                    chunk_id=record.chunk_id,
                    document_id=record.document_id,
                    document_title=record.document_title,
                    source=record.source,
                    document_type=record.document_type,
                    section_heading=record.section_heading,
                    content=record.content,
                    distance=1.0,
                    similarity=0.0,
                    rank=rank,
                    bm25_score=score,
                    bm25_rank=rank,
                    retrieval_method="bm25",
                )
            )
        return hits

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "chunks": [
                {
                    "chunk_id": r.chunk_id,
                    "document_id": r.document_id,
                    "document_title": r.document_title,
                    "source": r.source,
                    "document_type": r.document_type,
                    "section_heading": r.section_heading,
                    "content": r.content,
                    "content_hash": r.content_hash,
                    "chunk_index": r.chunk_index,
                }
                for r in self._records
            ],
        }
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload), encoding="utf-8")
        tmp.replace(self.path)
        logger.info("bm25_saved chunks=%s path=%s", len(self._records), self.path)

    def load(self) -> bool:
        if not self.path.is_file():
            self.clear()
            return False
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("bm25_load_failed path=%s error=%s", self.path, exc)
            self.clear()
            return False
        chunks = payload.get("chunks") or []
        records = [
            Bm25ChunkRecord(
                chunk_id=str(item.get("chunk_id") or ""),
                document_id=str(item.get("document_id") or ""),
                document_title=str(item.get("document_title") or ""),
                source=str(item.get("source") or ""),
                document_type=str(item.get("document_type") or ""),
                section_heading=str(item.get("section_heading") or ""),
                content=str(item.get("content") or ""),
                content_hash=str(item.get("content_hash") or ""),
                chunk_index=int(item.get("chunk_index") or 0),
            )
            for item in chunks
            if item.get("chunk_id")
        ]
        self.rebuild(records)
        return True

    @staticmethod
    def _build_model(tokenized: list[list[str]]):
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise Bm25IndexError(
                "rank-bm25 is not installed. Run: pip install -r requirements.txt"
            ) from exc
        if not tokenized:
            return None
        # BM25Okapi requires a non-empty corpus; empty token docs are allowed.
        corpus = [tokens if tokens else [""] for tokens in tokenized]
        return BM25Okapi(corpus)


def sync_bm25_with_store(index: Bm25Index, store: ChromaVectorStore) -> None:
    """Rebuild BM25 from Chroma so keyword search matches the vector store."""

    index.rebuild_from_store(store)
