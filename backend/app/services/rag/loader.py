"""Markdown document loading and metadata extraction.

Loaders stay source-specific so PDF / SharePoint / Jira connectors can be
added later without changing chunking or embedding.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from app.services.rag.models import LoadedDocument

logger = logging.getLogger(__name__)

_TYPE_LINE = re.compile(
    r"^\*\*Document type:\*\*\s*(?P<value>[^*\n]+)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_HEADING = re.compile(r"^#\s+(.+?)\s*$", re.MULTILINE)
_SLUG_SAFE = re.compile(r"[^a-z0-9]+")


class DocumentLoadError(ValueError):
    """Raised when a source file cannot become a LoadedDocument."""


def document_id_for_path(path: Path, root: Path) -> str:
    """Deterministic id from the path relative to the knowledge root."""

    relative = path.resolve().relative_to(root.resolve()).as_posix()
    stem = Path(relative).with_suffix("").as_posix().lower()
    slug = _SLUG_SAFE.sub("-", stem).strip("-")
    if not slug:
        raise DocumentLoadError(f"Could not derive document id from path: {path}")
    return slug


def content_hash(text: str) -> str:
    """Stable hash of normalized document body."""

    normalized = text.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def extract_title(content: str, fallback: str) -> str:
    match = _HEADING.search(content)
    if match:
        return match.group(1).strip()
    return fallback


def extract_document_type(content: str) -> str:
    match = _TYPE_LINE.search(content)
    if match:
        return match.group("value").strip().lower().replace(" ", "_")
    return "general"


def load_markdown_file(path: Path, root: Path) -> LoadedDocument:
    """Parse one Markdown file into a LoadedDocument."""

    if not path.is_file():
        raise DocumentLoadError(f"Not a file: {path}")
    if path.suffix.lower() not in {".md", ".markdown"}:
        raise DocumentLoadError(f"Unsupported document type: {path.suffix}")

    try:
        raw = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DocumentLoadError(f"Could not read {path}: {exc}") from exc

    content = raw.replace("\r\n", "\n").replace("\r", "\n").strip()
    if not content:
        raise DocumentLoadError(f"Document is empty: {path}")

    doc_id = document_id_for_path(path, root)
    source = path.resolve().relative_to(root.resolve()).as_posix()
    title = extract_title(content, fallback=path.stem.replace("-", " ").title())
    return LoadedDocument(
        document_id=doc_id,
        title=title,
        source=source,
        document_type=extract_document_type(content),
        content=content,
        content_hash=content_hash(content),
    )


def load_markdown_directory(root: Path) -> list[LoadedDocument]:
    """Load all Markdown files under root (recursive). Skips malformed files with errors logged."""

    if not root.exists():
        raise DocumentLoadError(f"Knowledge source directory does not exist: {root}")
    if not root.is_dir():
        raise DocumentLoadError(f"Knowledge source path is not a directory: {root}")

    documents: list[LoadedDocument] = []
    errors: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in {".md", ".markdown"}:
            continue
        if path.name.startswith("."):
            continue
        try:
            documents.append(load_markdown_file(path, root))
        except DocumentLoadError as exc:
            logger.warning("skip_document path=%s error=%s", path, exc)
            errors.append(str(exc))

    ids = [doc.document_id for doc in documents]
    if len(ids) != len(set(ids)):
        raise DocumentLoadError("Duplicate document ids detected after loading.")

    if not documents and errors:
        raise DocumentLoadError(
            "No valid documents loaded; first error: " + errors[0]
        )
    return documents
