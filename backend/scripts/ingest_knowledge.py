"""Ingest Markdown knowledge sources into the local ChromaDB collection.

Run from the backend directory (billed embedding calls for changed docs only):

    .\\.venv\\Scripts\\python.exe -m scripts.ingest_knowledge
    .\\.venv\\Scripts\\python.exe -m scripts.ingest_knowledge --force
    .\\.venv\\Scripts\\python.exe -m scripts.ingest_knowledge --dry-run

Unchanged documents (same content hash) are skipped. Removed source files
prune their chunks. Costs stay low: text-embedding-3-small is $0.02 / 1M tokens.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from app.config import get_settings
from app.db.database import init_db
from app.db.repository import Repository
from app.services.rag.chunking import chunk_document
from app.services.rag.embeddings import OpenAIEmbeddingProvider
from app.services.rag.factory import build_bm25_index, build_vector_store, resolve_path
from app.services.rag.ingest import IngestionError, ingest_directory
from app.services.rag.loader import DocumentLoadError, load_markdown_directory

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Ingest OperateSolve knowledge documents.")
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Override knowledge source directory (default: KNOWLEDGE_SOURCE_DIR).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-embed every document even when content hashes match.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Load and chunk only; do not call embeddings or write to Chroma.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the ingest report as JSON.",
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    logging.basicConfig(level=settings.log_level)
    source = resolve_path(args.source) if args.source else settings.knowledge_source_path

    if args.dry_run:
        return _dry_run(source, settings, as_json=args.json)

    init_db(settings.database_path)
    repo = Repository(settings.database_path)
    store = build_vector_store(settings)
    embedder = OpenAIEmbeddingProvider(settings, repository=repo)
    bm25_index = build_bm25_index(settings, load=True)

    if args.force:
        for doc_id in list(store.document_hashes()):
            store.delete_document(doc_id)
        bm25_index.clear()

    try:
        report = ingest_directory(
            source,
            store,
            embedder,
            chunk_size_tokens=settings.rag_chunk_size_tokens,
            overlap_tokens=settings.rag_chunk_overlap_tokens,
            bm25_index=bm25_index,
        )
    except IngestionError as exc:
        print(f"Ingestion failed: {exc}", file=sys.stderr)
        return 1

    payload = {
        **report.as_dict(),
        "source": str(source),
        "chroma": str(settings.chroma_path),
        "bm25_index": str(settings.bm25_path),
        "bm25_chunks": bm25_index.size,
        "collection": settings.chroma_collection_name,
        "embedding_model": settings.embedding_model,
        "total_chunks": store.count(),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print("Knowledge ingestion complete")
        print(f"  source:            {payload['source']}")
        print(f"  documents seen:    {payload['documents_seen']}")
        print(f"  indexed:           {payload['documents_indexed']}")
        print(f"  skipped unchanged: {payload['documents_skipped']}")
        print(f"  removed stale:     {payload['documents_removed']}")
        print(f"  chunks upserted:   {payload['chunks_upserted']}")
        print(f"  chunks deleted:    {payload['chunks_deleted']}")
        print(f"  embedding tokens:  {payload['embedding_tokens']}")
        print(f"  total in store:    {payload['total_chunks']}")
        print(f"  bm25 chunks:       {payload['bm25_chunks']}")
        print(f"  model:             {payload['embedding_model']}")
        if payload["errors"]:
            print("  errors:")
            for err in payload["errors"]:
                print(f"    - {err}")
    return 1 if payload["errors"] else 0


def _dry_run(source: Path, settings, *, as_json: bool) -> int:
    try:
        documents = load_markdown_directory(source)
    except DocumentLoadError as exc:
        print(f"Dry run failed: {exc}", file=sys.stderr)
        return 1
    chunk_total = 0
    rows = []
    for doc in documents:
        chunks = chunk_document(
            doc,
            chunk_size_tokens=settings.rag_chunk_size_tokens,
            overlap_tokens=settings.rag_chunk_overlap_tokens,
        )
        chunk_total += len(chunks)
        rows.append(
            {
                "document_id": doc.document_id,
                "title": doc.title,
                "type": doc.document_type,
                "chunks": len(chunks),
                "content_hash": doc.content_hash[:12],
            }
        )
    payload = {
        "dry_run": True,
        "source": str(source),
        "documents": len(documents),
        "chunks": chunk_total,
        "items": rows,
    }
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"Dry run: {len(documents)} documents → ~{chunk_total} chunks")
        for row in rows:
            print(f"  - {row['document_id']}: {row['chunks']} chunks ({row['title']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
