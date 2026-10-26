"""Idempotent document ingestion pipeline.

Documents → load → chunk → embed → upsert. Unchanged documents are skipped
using content hashes. Removed source files delete their stale chunks.
"""

from __future__ import annotations

import logging
from pathlib import Path

from app.services.rag.bm25 import Bm25Index, sync_bm25_with_store
from app.services.rag.chunking import chunk_document
from app.services.rag.embeddings import EmbeddingError, EmbeddingProvider
from app.services.rag.loader import DocumentLoadError, load_markdown_directory
from app.services.rag.models import DocumentChunk, IngestReport, LoadedDocument
from app.services.rag.store import ChromaVectorStore, VectorStoreError

logger = logging.getLogger(__name__)


class IngestionError(RuntimeError):
    pass


def ingest_directory(
    source_dir: Path,
    store: ChromaVectorStore,
    embedder: EmbeddingProvider,
    *,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 75,
    prune_missing: bool = True,
    bm25_index: Bm25Index | None = None,
) -> IngestReport:
    """Ingest all Markdown documents under source_dir into the vector store.

    When bm25_index is provided, it is rebuilt from Chroma after mutations so
    keyword search stays synchronized with upserts and deletes.
    """

    report = IngestReport()
    try:
        documents = load_markdown_directory(source_dir)
    except DocumentLoadError as exc:
        raise IngestionError(str(exc)) from exc

    report.documents_seen = len(documents)
    existing_hashes = store.document_hashes()
    seen_ids = {doc.document_id for doc in documents}
    mutated = False

    for document in documents:
        try:
            changed = _ingest_one(
                document,
                store,
                embedder,
                report,
                existing_hashes=existing_hashes,
                chunk_size_tokens=chunk_size_tokens,
                overlap_tokens=overlap_tokens,
            )
            mutated = mutated or changed
        except (EmbeddingError, VectorStoreError, ValueError) as exc:
            message = f"{document.document_id}: {exc}"
            logger.exception("ingest_failed document_id=%s", document.document_id)
            report.errors.append(message)

    if prune_missing:
        stale_ids = [doc_id for doc_id in existing_hashes if doc_id not in seen_ids]
        for doc_id in stale_ids:
            deleted = store.delete_document(doc_id)
            report.chunks_deleted += deleted
            report.documents_removed += 1
            mutated = True
            logger.info("ingest_pruned document_id=%s chunks=%s", doc_id, deleted)

    if bm25_index is not None and (mutated or bm25_index.size == 0):
        sync_bm25_with_store(bm25_index, store)
    elif bm25_index is not None and bm25_index.size != store.count():
        # Heal drift (manual chroma edits, crashed prior ingest, etc.).
        logger.warning(
            "bm25_store_drift bm25=%s chroma=%s rebuilding",
            bm25_index.size,
            store.count(),
        )
        sync_bm25_with_store(bm25_index, store)

    if report.errors and report.documents_indexed == 0 and report.documents_skipped == 0:
        raise IngestionError("; ".join(report.errors))
    return report


def _ingest_one(
    document: LoadedDocument,
    store: ChromaVectorStore,
    embedder: EmbeddingProvider,
    report: IngestReport,
    *,
    existing_hashes: dict[str, str],
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> bool:
    previous = existing_hashes.get(document.document_id)
    if previous == document.content_hash:
        report.documents_skipped += 1
        logger.info(
            "ingest_skip_unchanged document_id=%s hash=%s",
            document.document_id,
            document.content_hash[:12],
        )
        return False

    chunks = chunk_document(
        document,
        chunk_size_tokens=chunk_size_tokens,
        overlap_tokens=overlap_tokens,
    )
    deleted = store.delete_document(document.document_id)
    report.chunks_deleted += deleted

    vectors = _embed_chunks(chunks, embedder, report)
    store.upsert_chunks(chunks, vectors)
    report.documents_indexed += 1
    report.chunks_upserted += len(chunks)
    logger.info(
        "ingest_indexed document_id=%s chunks=%s deleted_stale=%s",
        document.document_id,
        len(chunks),
        deleted,
    )
    return True


def _embed_chunks(
    chunks: list[DocumentChunk],
    embedder: EmbeddingProvider,
    report: IngestReport,
) -> list[list[float]]:
    texts = [chunk.content for chunk in chunks]
    result = embedder.embed_texts_sync(texts)
    if result.input_tokens:
        report.embedding_tokens += result.input_tokens
    if len(result.vectors) != len(chunks):
        raise EmbeddingError(
            f"Embedding count mismatch: chunks={len(chunks)} vectors={len(result.vectors)}"
        )
    return result.vectors
