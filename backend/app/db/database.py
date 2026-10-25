"""SQLite connection and schema."""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    title_customized INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    position INTEGER NOT NULL,
    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL CHECK (status IN ('complete', 'streaming', 'error', 'stopped')),
    model TEXT,
    latency_ms INTEGER,
    input_tokens INTEGER,
    output_tokens INTEGER,
    request_id TEXT,
    provider_response_id TEXT,
    client_request_id TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_messages_client_request
    ON messages(client_request_id)
    WHERE client_request_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_messages_conversation
    ON messages(conversation_id, position);

CREATE INDEX IF NOT EXISTS idx_conversations_updated
    ON conversations(updated_at DESC);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(database_path: str) -> None:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(database_path) as connection:
        connection.executescript(SCHEMA)
    recover_interrupted(database_path)


def recover_interrupted(database_path: str) -> None:
    """Mark orphaned streams so a refresh can offer retry instead of spinning forever."""

    now = utc_now()
    with connect(database_path) as connection:
        connection.execute(
            """
            UPDATE messages
            SET status = 'error',
                error = CASE
                    WHEN error IS NULL OR TRIM(error) = ''
                    THEN 'Generation was interrupted before it finished.'
                    ELSE error
                END,
                updated_at = ?
            WHERE status = 'streaming'
            """,
            (now,),
        )


@contextmanager
def connect(database_path: str) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(database_path, timeout=5.0)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA busy_timeout = 5000")
    connection.execute("PRAGMA journal_mode = WAL")
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()
