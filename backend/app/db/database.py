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


USAGE_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_usage (
    id TEXT PRIMARY KEY,
    request_id TEXT NOT NULL UNIQUE,
    conversation_id TEXT,
    message_id TEXT,
    provider TEXT NOT NULL,
    requested_model TEXT,
    model TEXT,
    status TEXT NOT NULL,
    input_tokens INTEGER,
    output_tokens INTEGER,
    cached_input_tokens INTEGER,
    cache_write_tokens INTEGER,
    reasoning_tokens INTEGER,
    usage_reported INTEGER NOT NULL DEFAULT 0,
    duration_ms INTEGER,
    started_at TEXT NOT NULL,
    finished_at TEXT NOT NULL,
    provider_response_id TEXT,
    source TEXT NOT NULL DEFAULT 'live' CHECK (source IN ('live', 'backfill'))
);

CREATE INDEX IF NOT EXISTS idx_llm_usage_started ON llm_usage(started_at);
CREATE INDEX IF NOT EXISTS idx_llm_usage_conversation ON llm_usage(conversation_id);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def init_db(database_path: str) -> None:
    path = Path(database_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with connect(database_path) as connection:
        connection.executescript(SCHEMA)
    with connect(database_path) as connection:
        migrate(connection)
    recover_interrupted(database_path)


def migrate(connection: sqlite3.Connection) -> None:
    """Apply pending migrations, tracked with PRAGMA user_version.

    Phase 1 databases start at version 0. Each migration must be safe to
    re-run so a partially upgraded file still converges.
    """

    version = connection.execute("PRAGMA user_version").fetchone()[0]
    for target, migration in enumerate(MIGRATIONS, start=1):
        if version < target:
            migration(connection)
            connection.execute(f"PRAGMA user_version = {target}")


def _add_usage_tracking(connection: sqlite3.Connection) -> None:
    columns = {row["name"] for row in connection.execute("PRAGMA table_info(messages)")}
    if "cached_input_tokens" not in columns:
        connection.execute("ALTER TABLE messages ADD COLUMN cached_input_tokens INTEGER")
    for statement in USAGE_SCHEMA.split(";"):
        if statement.strip():
            connection.execute(statement)
    # Import Phase 1 assistant metrics so earlier spend is not invisible.
    # Phase 1 kept only the latest attempt per message and never stored cached tokens.
    connection.execute(
        """
        INSERT OR IGNORE INTO llm_usage (
            id, request_id, conversation_id, message_id, provider, requested_model, model,
            status, input_tokens, output_tokens, usage_reported, duration_ms,
            started_at, finished_at, provider_response_id, source
        )
        SELECT
            lower(hex(randomblob(16))), request_id, conversation_id, id, 'openai', model, model,
            status, input_tokens, output_tokens,
            CASE WHEN input_tokens IS NOT NULL AND output_tokens IS NOT NULL THEN 1 ELSE 0 END,
            latency_ms, updated_at, updated_at, provider_response_id, 'backfill'
        FROM messages
        WHERE role = 'assistant'
          AND request_id IS NOT NULL
          AND status IN ('complete', 'stopped', 'error')
          AND (
              input_tokens IS NOT NULL
              OR output_tokens IS NOT NULL
              OR provider_response_id IS NOT NULL
              OR status IN ('complete', 'stopped')
          )
        """
    )


def _add_model_routing(connection: sqlite3.Connection) -> None:
    """Per-conversation model preference, and routing details per message and per request.

    Existing conversations become Auto + General, which routes to the same
    baseline model they used before.
    """

    additions = {
        "conversations": [
            ("model_selection", "TEXT NOT NULL DEFAULT 'auto'"),
            ("task_mode", "TEXT NOT NULL DEFAULT 'general'"),
        ],
        "messages": [
            ("requested_model", "TEXT"),
            ("routing_mode", "TEXT"),
            ("routing_reason", "TEXT"),
        ],
        "llm_usage": [
            ("selection", "TEXT"),
            ("task_mode", "TEXT"),
            ("routing_mode", "TEXT"),
            ("routing_reason", "TEXT"),
            ("first_token_ms", "INTEGER"),
        ],
    }
    for table, columns in additions.items():
        existing = {row["name"] for row in connection.execute(f"PRAGMA table_info({table})")}
        for name, definition in columns:
            if name not in existing:
                connection.execute(f"ALTER TABLE {table} ADD COLUMN {name} {definition}")


def _add_context_summaries(connection: sqlite3.Connection) -> None:
    """Conversation summary checkpoints and chat vs summarization usage purpose."""

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_summaries (
            conversation_id TEXT PRIMARY KEY,
            summary_text TEXT NOT NULL,
            covered_through_position INTEGER NOT NULL,
            covered_through_message_id TEXT NOT NULL,
            summary_tokens INTEGER,
            model TEXT,
            updated_at TEXT NOT NULL,
            FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
        )
        """
    )
    usage_columns = {row["name"] for row in connection.execute("PRAGMA table_info(llm_usage)")}
    if "purpose" not in usage_columns:
        connection.execute(
            "ALTER TABLE llm_usage ADD COLUMN purpose TEXT NOT NULL DEFAULT 'chat'"
        )


MIGRATIONS = [_add_usage_tracking, _add_model_routing, _add_context_summaries]


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
