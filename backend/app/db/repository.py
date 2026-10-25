"""SQLite data access for conversations and messages."""

from __future__ import annotations

import sqlite3
import uuid
from typing import Any

from app.db.database import connect, utc_now


class DuplicateClientRequest(Exception):
    def __init__(self, client_request_id: str) -> None:
        super().__init__(client_request_id)
        self.client_request_id = client_request_id


class Repository:
    def __init__(self, database_path: str) -> None:
        self.database_path = database_path

    def ping(self) -> None:
        with connect(self.database_path) as connection:
            connection.execute("SELECT 1")

    def create_conversation(
        self,
        title: str | None = None,
        model_selection: str = "auto",
        task_mode: str = "general",
    ) -> dict[str, Any]:
        conversation_id = str(uuid.uuid4())
        now = utc_now()
        customized = 1 if title else 0
        stored_title = title or "New conversation"
        with connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    id, title, title_customized, model_selection, task_mode, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (conversation_id, stored_title, customized, model_selection, task_mode, now, now),
            )
        conversation = self.get_conversation(conversation_id)
        if conversation is None:
            raise RuntimeError("Conversation was not saved")
        return conversation

    def list_conversations(self, limit: int = 100) -> list[dict[str, Any]]:
        with connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM conversations
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [_conversation(row) for row in rows]

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE id = ?",
                (conversation_id,),
            ).fetchone()
        return _conversation(row) if row else None

    def rename_conversation(self, conversation_id: str, title: str) -> dict[str, Any] | None:
        now = utc_now()
        with connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                UPDATE conversations
                SET title = ?, title_customized = 1, updated_at = ?
                WHERE id = ?
                """,
                (title, now, conversation_id),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_conversation(conversation_id)

    def set_model_preference(
        self,
        conversation_id: str,
        model_selection: str,
        task_mode: str,
    ) -> dict[str, Any] | None:
        """Does not touch updated_at, so changing the model does not reorder the sidebar."""

        with connect(self.database_path) as connection:
            cursor = connection.execute(
                "UPDATE conversations SET model_selection = ?, task_mode = ? WHERE id = ?",
                (model_selection, task_mode, conversation_id),
            )
            if cursor.rowcount == 0:
                return None
        return self.get_conversation(conversation_id)

    def delete_conversation(self, conversation_id: str) -> bool:
        with connect(self.database_path) as connection:
            cursor = connection.execute(
                "DELETE FROM conversations WHERE id = ?",
                (conversation_id,),
            )
            return cursor.rowcount > 0

    def list_messages(self, conversation_id: str) -> list[dict[str, Any]]:
        with connect(self.database_path) as connection:
            rows = connection.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY position ASC
                """,
                (conversation_id,),
            ).fetchall()
        return [_message(row) for row in rows]

    def get_message(self, message_id: str) -> dict[str, Any] | None:
        with connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT * FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
        return _message(row) if row else None

    def get_by_client_request(self, client_request_id: str) -> dict[str, Any] | None:
        with connect(self.database_path) as connection:
            row = connection.execute(
                "SELECT * FROM messages WHERE client_request_id = ?",
                (client_request_id,),
            ).fetchone()
        return _message(row) if row else None

    def assistant_after(self, user_message_id: str) -> dict[str, Any] | None:
        user = self.get_message(user_message_id)
        if user is None:
            return None
        with connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                  AND role = 'assistant'
                  AND position > ?
                ORDER BY position ASC
                LIMIT 1
                """,
                (user["conversation_id"], user["position"]),
            ).fetchone()
        return _message(row) if row else None

    def last_message(self, conversation_id: str) -> dict[str, Any] | None:
        with connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT * FROM messages
                WHERE conversation_id = ?
                ORDER BY position DESC
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
        return _message(row) if row else None

    def has_streaming(self, conversation_id: str) -> bool:
        with connect(self.database_path) as connection:
            row = connection.execute(
                """
                SELECT 1 FROM messages
                WHERE conversation_id = ? AND status = 'streaming'
                LIMIT 1
                """,
                (conversation_id,),
            ).fetchone()
        return row is not None

    def insert_exchange(
        self,
        conversation_id: str,
        content: str,
        client_request_id: str,
        title: str | None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        user_id = str(uuid.uuid4())
        assistant_id = str(uuid.uuid4())
        now = utc_now()
        with connect(self.database_path) as connection:
            base = connection.execute(
                "SELECT COALESCE(MAX(position), 0) FROM messages WHERE conversation_id = ?",
                (conversation_id,),
            ).fetchone()[0]
            try:
                connection.execute(
                    """
                    INSERT INTO messages (
                        id, conversation_id, position, role, content, status,
                        client_request_id, created_at, updated_at
                    ) VALUES (?, ?, ?, 'user', ?, 'complete', ?, ?, ?)
                    """,
                    (user_id, conversation_id, base + 1, content, client_request_id, now, now),
                )
            except sqlite3.IntegrityError as exc:
                raise DuplicateClientRequest(client_request_id) from exc
            connection.execute(
                """
                INSERT INTO messages (
                    id, conversation_id, position, role, content, status, created_at, updated_at
                ) VALUES (?, ?, ?, 'assistant', '', 'streaming', ?, ?)
                """,
                (assistant_id, conversation_id, base + 2, now, now),
            )
            if title:
                connection.execute(
                    """
                    UPDATE conversations
                    SET title = ?, updated_at = ?
                    WHERE id = ? AND title_customized = 0
                    """,
                    (title, now, conversation_id),
                )
            else:
                connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, conversation_id),
                )
        user = self.get_message(user_id)
        assistant = self.get_message(assistant_id)
        if user is None or assistant is None:
            raise RuntimeError("Messages were not saved")
        return user, assistant

    def mark_streaming(
        self,
        message_id: str,
        request_id: str,
        *,
        requested_model: str | None = None,
        routing_mode: str | None = None,
        routing_reason: str | None = None,
    ) -> None:
        now = utc_now()
        with connect(self.database_path) as connection:
            connection.execute(
                """
                UPDATE messages
                SET status = 'streaming',
                    content = '',
                    error = NULL,
                    request_id = ?,
                    model = NULL,
                    requested_model = ?,
                    routing_mode = ?,
                    routing_reason = ?,
                    latency_ms = NULL,
                    input_tokens = NULL,
                    output_tokens = NULL,
                    cached_input_tokens = NULL,
                    provider_response_id = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (request_id, requested_model, routing_mode, routing_reason, now, message_id),
            )

    def finish_message(
        self,
        message_id: str,
        *,
        content: str,
        status: str,
        model: str | None,
        latency_ms: int | None,
        input_tokens: int | None,
        output_tokens: int | None,
        request_id: str | None,
        provider_response_id: str | None,
        error: str | None,
        cached_input_tokens: int | None = None,
    ) -> dict[str, Any] | None:
        now = utc_now()
        with connect(self.database_path) as connection:
            connection.execute(
                """
                UPDATE messages
                SET content = ?,
                    status = ?,
                    model = ?,
                    latency_ms = ?,
                    input_tokens = ?,
                    output_tokens = ?,
                    cached_input_tokens = ?,
                    request_id = ?,
                    provider_response_id = ?,
                    error = ?,
                    updated_at = ?
                WHERE id = ?
                """,
                (
                    content,
                    status,
                    model,
                    latency_ms,
                    input_tokens,
                    output_tokens,
                    cached_input_tokens,
                    request_id,
                    provider_response_id,
                    error,
                    now,
                    message_id,
                ),
            )
            row = connection.execute(
                "SELECT * FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
            if row is not None:
                connection.execute(
                    "UPDATE conversations SET updated_at = ? WHERE id = ?",
                    (now, row["conversation_id"]),
                )
        return _message(row) if row is not None else None

    def record_usage(
        self,
        *,
        request_id: str,
        conversation_id: str,
        message_id: str,
        provider: str,
        requested_model: str | None,
        model: str | None,
        status: str,
        input_tokens: int | None,
        output_tokens: int | None,
        cached_input_tokens: int | None,
        cache_write_tokens: int | None,
        reasoning_tokens: int | None,
        duration_ms: int | None,
        started_at: str,
        finished_at: str,
        provider_response_id: str | None,
        selection: str | None = None,
        task_mode: str | None = None,
        routing_mode: str | None = None,
        routing_reason: str | None = None,
        first_token_ms: int | None = None,
    ) -> bool:
        """Store one provider request. A repeated request_id is ignored, so usage counts once."""

        reported = 1 if input_tokens is not None and output_tokens is not None else 0
        with connect(self.database_path) as connection:
            cursor = connection.execute(
                """
                INSERT OR IGNORE INTO llm_usage (
                    id, request_id, conversation_id, message_id, provider, requested_model, model,
                    status, input_tokens, output_tokens, cached_input_tokens, cache_write_tokens,
                    reasoning_tokens, usage_reported, duration_ms, started_at, finished_at,
                    provider_response_id, source, selection, task_mode, routing_mode,
                    routing_reason, first_token_ms
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'live', ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid.uuid4()),
                    request_id,
                    conversation_id,
                    message_id,
                    provider,
                    requested_model,
                    model,
                    status,
                    input_tokens,
                    output_tokens,
                    cached_input_tokens,
                    cache_write_tokens,
                    reasoning_tokens,
                    reported,
                    duration_ms,
                    started_at,
                    finished_at,
                    provider_response_id,
                    selection,
                    task_mode,
                    routing_mode,
                    routing_reason,
                    first_token_ms,
                ),
            )
            return cursor.rowcount > 0

    def list_usage(self, since: str | None = None) -> list[dict[str, Any]]:
        """Usage rows, newest first, with the current conversation title when it still exists."""

        query = """
            SELECT u.*, c.title AS conversation_title
            FROM llm_usage u
            LEFT JOIN conversations c ON c.id = u.conversation_id
        """
        params: tuple[Any, ...] = ()
        if since is not None:
            query += " WHERE u.started_at >= ?"
            params = (since,)
        query += " ORDER BY u.started_at DESC"
        with connect(self.database_path) as connection:
            rows = connection.execute(query, params).fetchall()
        return [_usage(row) for row in rows]


def _usage(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "request_id": row["request_id"],
        "conversation_id": row["conversation_id"],
        "conversation_title": row["conversation_title"],
        "message_id": row["message_id"],
        "provider": row["provider"],
        "requested_model": row["requested_model"],
        "model": row["model"],
        "status": row["status"],
        "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"],
        "cached_input_tokens": row["cached_input_tokens"],
        "cache_write_tokens": row["cache_write_tokens"],
        "reasoning_tokens": row["reasoning_tokens"],
        "usage_reported": bool(row["usage_reported"]),
        "duration_ms": row["duration_ms"],
        "started_at": row["started_at"],
        "finished_at": row["finished_at"],
        "source": row["source"],
        "selection": row["selection"],
        "task_mode": row["task_mode"],
        "routing_mode": row["routing_mode"],
        "routing_reason": row["routing_reason"],
        "first_token_ms": row["first_token_ms"],
    }


def _conversation(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "title_customized": bool(row["title_customized"]),
        "model_selection": row["model_selection"],
        "task_mode": row["task_mode"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def _message(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "conversation_id": row["conversation_id"],
        "position": row["position"],
        "role": row["role"],
        "content": row["content"],
        "status": row["status"],
        "model": row["model"],
        "requested_model": row["requested_model"],
        "routing_mode": row["routing_mode"],
        "routing_reason": row["routing_reason"],
        "latency_ms": row["latency_ms"],
        "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"],
        "cached_input_tokens": row["cached_input_tokens"],
        "request_id": row["request_id"],
        "provider_response_id": row["provider_response_id"],
        "client_request_id": row["client_request_id"],
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
