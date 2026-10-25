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

    def create_conversation(self, title: str | None = None) -> dict[str, Any]:
        conversation_id = str(uuid.uuid4())
        now = utc_now()
        customized = 1 if title else 0
        stored_title = title or "New conversation"
        with connect(self.database_path) as connection:
            connection.execute(
                """
                INSERT INTO conversations (id, title, title_customized, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, stored_title, customized, now, now),
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

    def mark_streaming(self, message_id: str, request_id: str) -> None:
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
                    latency_ms = NULL,
                    input_tokens = NULL,
                    output_tokens = NULL,
                    provider_response_id = NULL,
                    updated_at = ?
                WHERE id = ?
                """,
                (request_id, now, message_id),
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


def _conversation(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "id": row["id"],
        "title": row["title"],
        "title_customized": bool(row["title_customized"]),
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
        "latency_ms": row["latency_ms"],
        "input_tokens": row["input_tokens"],
        "output_tokens": row["output_tokens"],
        "request_id": row["request_id"],
        "provider_response_id": row["provider_response_id"],
        "client_request_id": row["client_request_id"],
        "error": row["error"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }
