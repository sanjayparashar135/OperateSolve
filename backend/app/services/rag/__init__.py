"""Phase 3.1–3.3 — RAG foundation, grounded answers, hybrid retrieval.

Ingest → chunk → embed → Chroma (+ BM25) → retrieve (semantic|hybrid RRF)
→ optional local rerank → select → generate → cite.
"""

from app.services.rag.models import DocumentChunk, LoadedDocument, SearchHit

__all__ = ["DocumentChunk", "LoadedDocument", "SearchHit"]
