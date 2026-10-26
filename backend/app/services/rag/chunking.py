"""Heading-aware Markdown chunking with overlap.

Uses the same character heuristic as chat context budgeting so chunk sizes
stay consistent with the rest of OperateSolve. Avoids heavy NLP frameworks.
"""

from __future__ import annotations

import re

from app.services.context.tokens import estimate_tokens
from app.services.rag.models import DocumentChunk, LoadedDocument

_HEADING_LINE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def chunk_document(
    document: LoadedDocument,
    *,
    chunk_size_tokens: int = 500,
    overlap_tokens: int = 75,
) -> list[DocumentChunk]:
    """Split a document into overlapping chunks, preferring Markdown headings."""

    if chunk_size_tokens < 50:
        raise ValueError("chunk_size_tokens must be at least 50")
    if overlap_tokens < 0 or overlap_tokens >= chunk_size_tokens:
        raise ValueError("overlap_tokens must be >= 0 and < chunk_size_tokens")

    sections = _split_sections(document.content)
    raw_chunks: list[tuple[str, str]] = []
    for heading, body in sections:
        pieces = _split_with_overlap(body, chunk_size_tokens, overlap_tokens)
        for piece in pieces:
            text = piece.strip()
            if text:
                raw_chunks.append((heading, text))

    if not raw_chunks:
        raise ValueError(f"Document produced no chunks: {document.document_id}")

    chunks: list[DocumentChunk] = []
    for index, (heading, text) in enumerate(raw_chunks):
        section = heading or document.title
        # Prefix keeps retrieval context without needing a separate title field in the vector.
        embed_text = f"{document.title}\n\n## {section}\n\n{text}" if section else text
        chunks.append(
            DocumentChunk(
                chunk_id=f"{document.document_id}#{index:04d}",
                document_id=document.document_id,
                document_title=document.title,
                source=document.source,
                document_type=document.document_type,
                section_heading=section,
                content=embed_text,
                chunk_index=index,
                content_hash=document.content_hash,
                token_estimate=estimate_tokens(embed_text),
            )
        )
    return chunks


def _split_sections(content: str) -> list[tuple[str, str]]:
    """Return (heading, body) pairs. Leading text before the first heading uses title-level body."""

    lines = content.replace("\r\n", "\n").split("\n")
    sections: list[tuple[str, list[str]]] = []
    current_heading = ""
    current_lines: list[str] = []

    for line in lines:
        match = _HEADING_LINE.match(line)
        if match:
            if current_lines or current_heading:
                sections.append((current_heading, current_lines))
            current_heading = match.group(2).strip()
            current_lines = []
            continue
        current_lines.append(line)

    if current_lines or current_heading:
        sections.append((current_heading, current_lines))

    result: list[tuple[str, str]] = []
    for heading, body_lines in sections:
        body = "\n".join(body_lines).strip()
        if not body and not heading:
            continue
        # Keep heading-only sections out; they have no retrieval value alone.
        if not body:
            continue
        result.append((heading, body))
    return result


def _split_with_overlap(
    text: str,
    chunk_size_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Greedy token-estimate windows over paragraphs, with character-based overlap."""

    if estimate_tokens(text) <= chunk_size_tokens:
        return [text]

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not paragraphs:
        return _window_by_chars(text, chunk_size_tokens, overlap_tokens)

    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for paragraph in paragraphs:
        para_tokens = estimate_tokens(paragraph)
        if para_tokens > chunk_size_tokens:
            if current:
                chunks.append("\n\n".join(current))
                current, current_tokens = [], 0
            chunks.extend(_window_by_chars(paragraph, chunk_size_tokens, overlap_tokens))
            continue

        if current and current_tokens + para_tokens > chunk_size_tokens:
            chunks.append("\n\n".join(current))
            overlap_text = _tail_for_overlap("\n\n".join(current), overlap_tokens)
            current = [overlap_text, paragraph] if overlap_text else [paragraph]
            current_tokens = estimate_tokens("\n\n".join(current))
        else:
            current.append(paragraph)
            current_tokens += para_tokens

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def _window_by_chars(text: str, chunk_size_tokens: int, overlap_tokens: int) -> list[str]:
    """Fallback sliding window using ~4 chars/token."""

    size = max(chunk_size_tokens * 4, 200)
    overlap = max(overlap_tokens * 4, 0)
    step = max(size - overlap, 1)
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + size, length)
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= length:
            break
        start += step
    return chunks


def _tail_for_overlap(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0 or not text:
        return ""
    # Approximate overlap by characters from the end.
    chars = max(overlap_tokens * 4, 1)
    tail = text[-chars:].strip()
    # Prefer starting at a word boundary when possible.
    space = tail.find(" ")
    if space > 0 and space < len(tail) - 1:
        return tail[space + 1 :]
    return tail
