"""Conversation orchestration: persistence, context, and streamed generation."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from app.config import Settings
from app.db.repository import DuplicateClientRequest, Repository
from app.extensions import augment_instructions
from app.schemas.chat import MessageOut
from app.services.llm.base import ChatTurn, LLMDelta, LLMDone, LLMError, LLMProvider

logger = logging.getLogger(__name__)

INSTRUCTIONS = (
    "You are OperateSolve, an enterprise operations copilot. "
    "Help operators triage incidents, draft status updates, review runbooks, "
    "and reason about operational risk. Be concise, structured, and explicit "
    "about assumptions. Use markdown when it improves clarity. "
    "Do not invent live telemetry, ticket data, or customer impact. "
    "If that context was not provided, say what you still need. "
    "Prefer practical next steps an on-call operator can execute."
)

_prepare_lock = threading.Lock()


class ChatNotFound(Exception):
    pass


class ChatConflict(Exception):
    pass


@dataclass
class PreparedGeneration:
    conversation_id: str
    user_message: dict[str, Any]
    assistant_message: dict[str, Any]
    replay: bool


class ChatService:
    def __init__(self, repository: Repository, provider: LLMProvider, settings: Settings) -> None:
        self.repo = repository
        self.provider = provider
        self.settings = settings

    def create_conversation(self, title: str | None) -> dict[str, Any]:
        return self.repo.create_conversation(title)

    def list_conversations(self) -> list[dict[str, Any]]:
        return self.repo.list_conversations()

    def get_conversation(self, conversation_id: str) -> dict[str, Any]:
        conversation = self.repo.get_conversation(conversation_id)
        if conversation is None:
            raise ChatNotFound("Conversation not found")
        return conversation

    def get_detail(self, conversation_id: str) -> dict[str, Any]:
        conversation = self.get_conversation(conversation_id)
        return {
            "conversation": conversation,
            "messages": [public_message(message) for message in self.repo.list_messages(conversation_id)],
        }

    def rename_conversation(self, conversation_id: str, title: str) -> dict[str, Any]:
        renamed = self.repo.rename_conversation(conversation_id, title)
        if renamed is None:
            raise ChatNotFound("Conversation not found")
        return renamed

    def delete_conversation(self, conversation_id: str) -> None:
        if not self.repo.delete_conversation(conversation_id):
            raise ChatNotFound("Conversation not found")

    def prepare_new_message(
        self,
        conversation_id: str,
        content: str,
        client_request_id: str,
    ) -> PreparedGeneration:
        with _prepare_lock:
            conversation = self.repo.get_conversation(conversation_id)
            if conversation is None:
                raise ChatNotFound("Conversation not found")
            existing = self.repo.get_by_client_request(client_request_id)
            if existing is not None:
                return self._replay_existing(conversation_id, existing)
            if self.repo.has_streaming(conversation_id):
                raise ChatConflict("A response is already being generated in this conversation.")
            title = None if conversation["title_customized"] or conversation["title"] != "New conversation" else derive_title(content)
            try:
                user, assistant = self.repo.insert_exchange(
                    conversation_id,
                    content,
                    client_request_id,
                    title,
                )
            except DuplicateClientRequest:
                existing = self.repo.get_by_client_request(client_request_id)
                if existing is None:
                    raise ChatConflict("This message is still being saved. Try again.")
                return self._replay_existing(conversation_id, existing)
            return PreparedGeneration(conversation_id, user, assistant, replay=False)

    def prepare_retry(self, conversation_id: str, message_id: str) -> PreparedGeneration:
        with _prepare_lock:
            if self.repo.get_conversation(conversation_id) is None:
                raise ChatNotFound("Conversation not found")
            message = self.repo.get_message(message_id)
            if message is None or message["conversation_id"] != conversation_id:
                raise ChatNotFound("Message not found")
            if message["role"] != "assistant":
                raise ChatConflict("Only an assistant response can be retried.")
            last = self.repo.last_message(conversation_id)
            if last is None or last["id"] != message["id"]:
                raise ChatConflict("Only the latest response can be retried.")
            if message["status"] == "streaming" or self.repo.has_streaming(conversation_id):
                raise ChatConflict("A response is already being generated in this conversation.")
            user = self._preceding_user(conversation_id, message["position"])
            if user is None:
                raise ChatConflict("There is no prompt to retry.")
            request_id = str(uuid.uuid4())
            self.repo.mark_streaming(message["id"], request_id)
            assistant = self.repo.get_message(message["id"])
            if assistant is None:
                raise ChatNotFound("Message not found")
            return PreparedGeneration(conversation_id, user, assistant, replay=False)

    async def replay(self, prepared: PreparedGeneration) -> AsyncIterator[str]:
        assistant = self.repo.get_message(prepared.assistant_message["id"]) or prepared.assistant_message
        yield sse(self._meta(prepared, assistant.get("request_id"), replay=True))
        payload = public_message(assistant)
        if assistant["status"] == "error":
            yield sse({"type": "error", "error": assistant.get("error") or "The response failed.", "message": payload})
            return
        if assistant.get("content"):
            yield sse({"type": "delta", "text": assistant["content"]})
        yield sse({"type": "done", "message": payload})

    async def generate(
        self,
        prepared: PreparedGeneration,
        is_disconnected: Callable[[], Awaitable[bool]] | None,
        operator: str,
    ) -> AsyncIterator[str]:
        turns = self._turns(prepared.conversation_id, exclude_message_id=prepared.assistant_message["id"])
        if not any(turn.role == "user" for turn in turns):
            saved = self.repo.finish_message(
                prepared.assistant_message["id"],
                content="",
                status="error",
                model=None,
                latency_ms=0,
                input_tokens=None,
                output_tokens=None,
                request_id=prepared.assistant_message.get("request_id"),
                provider_response_id=None,
                error="There is no prompt to answer.",
            )
            if saved is not None:
                yield sse(self._meta(prepared, saved.get("request_id"), replay=False))
                yield sse({"type": "error", "error": saved["error"], "message": public_message(saved)})
            return

        instructions = augment_instructions(INSTRUCTIONS, conversation_id=prepared.conversation_id)
        async for chunk in self._run(
            prepared,
            turns,
            instructions,
            is_disconnected,
            operator,
        ):
            yield chunk

    def _replay_existing(self, conversation_id: str, existing: dict[str, Any]) -> PreparedGeneration:
        if existing["conversation_id"] != conversation_id:
            raise ChatConflict("This request id was already used in another conversation.")
        if existing["role"] != "user":
            raise ChatConflict("This request id was already used.")
        assistant = self.repo.assistant_after(existing["id"])
        if assistant is None:
            raise ChatConflict("This message is still being saved. Try again.")
        if assistant["status"] == "streaming":
            raise ChatConflict("A response is already being generated for this message.")
        return PreparedGeneration(conversation_id, existing, assistant, replay=True)

    def _preceding_user(self, conversation_id: str, position: int) -> dict[str, Any] | None:
        users = [
            message
            for message in self.repo.list_messages(conversation_id)
            if message["role"] == "user" and message["position"] < position
        ]
        return users[-1] if users else None

    def _turns(self, conversation_id: str, *, exclude_message_id: str) -> list[ChatTurn]:
        turns: list[ChatTurn] = []
        for message in self.repo.list_messages(conversation_id):
            if message["id"] == exclude_message_id or message["status"] == "streaming":
                continue
            if message["role"] == "assistant" and not message["content"]:
                continue
            if message["role"] not in {"user", "assistant"}:
                continue
            turns.append(ChatTurn(role=message["role"], content=message["content"]))
        return turns

    def _meta(self, prepared: PreparedGeneration, request_id: str | None, *, replay: bool) -> dict[str, Any]:
        conversation = self.repo.get_conversation(prepared.conversation_id)
        return {
            "type": "meta",
            "request_id": request_id or "",
            "conversation_id": prepared.conversation_id,
            "user_message_id": prepared.user_message["id"],
            "assistant_message_id": prepared.assistant_message["id"],
            "title": conversation["title"] if conversation else "New conversation",
            "replay": replay,
        }

    async def _run(
        self,
        prepared: PreparedGeneration,
        turns: list[ChatTurn],
        instructions: str,
        is_disconnected: Callable[[], Awaitable[bool]] | None,
        operator: str,
    ) -> AsyncIterator[str]:
        request_id = str(uuid.uuid4())
        assistant_id = prepared.assistant_message["id"]
        self.repo.mark_streaming(assistant_id, request_id)
        logger.info(
            "generation_started request_id=%s conversation_id=%s message_id=%s model=%s operator=%s",
            request_id,
            prepared.conversation_id,
            assistant_id,
            self.settings.openai_model,
            operator,
        )
        started = time.perf_counter()
        parts: list[str] = []
        model: str | None = self.settings.openai_model
        input_tokens: int | None = None
        output_tokens: int | None = None
        provider_response_id: str | None = None
        status = "complete"
        error: str | None = None
        saved = None

        try:
            yield sse(self._meta(prepared, request_id, replay=False))
            async for event in self.provider.stream(
                turns,
                model=self.settings.openai_model,
                instructions=instructions,
            ):
                if await _disconnected(is_disconnected):
                    status = "stopped"
                    break
                if isinstance(event, LLMDelta):
                    if event.text:
                        parts.append(event.text)
                        yield sse({"type": "delta", "text": event.text})
                elif isinstance(event, LLMError):
                    status = "error"
                    error = event.message or "The model request failed. You can retry."
                    provider_response_id = event.provider_response_id
                    break
                elif isinstance(event, LLMDone):
                    model = event.model or model
                    input_tokens = event.input_tokens
                    output_tokens = event.output_tokens
                    provider_response_id = event.provider_response_id
            if status == "complete" and not "".join(parts).strip():
                status = "error"
                error = "The model returned an empty response. You can retry."
        except (asyncio.CancelledError, GeneratorExit):
            status = "stopped"
            raise
        except Exception:
            logger.exception(
                "generation_failed request_id=%s conversation_id=%s",
                request_id,
                prepared.conversation_id,
            )
            status = "error"
            error = "The model request failed. You can retry."
        finally:
            latency_ms = int((time.perf_counter() - started) * 1000)
            content = "".join(parts)
            try:
                saved = self.repo.finish_message(
                    assistant_id,
                    content=content,
                    status=status,
                    model=model,
                    latency_ms=latency_ms,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    request_id=request_id,
                    provider_response_id=provider_response_id,
                    error=_clip(error) if error else None,
                )
            except Exception:
                saved = None
                logger.exception("persist_failed request_id=%s", request_id)
            logger.info(
                "generation_finished request_id=%s conversation_id=%s message_id=%s model=%s latency_ms=%s input_tokens=%s output_tokens=%s status=%s provider_response_id=%s",
                request_id,
                prepared.conversation_id,
                assistant_id,
                model,
                latency_ms,
                "unavailable" if input_tokens is None else input_tokens,
                "unavailable" if output_tokens is None else output_tokens,
                status,
                provider_response_id or "",
            )

        if status == "stopped" or saved is None:
            return
        payload = public_message(saved)
        if status == "error":
            yield sse({"type": "error", "error": saved.get("error") or error, "message": payload})
            return
        yield sse({"type": "done", "message": payload})


def public_message(message: dict[str, Any]) -> dict[str, Any]:
    return MessageOut.model_validate(
        {
            "id": message["id"],
            "conversation_id": message["conversation_id"],
            "role": message["role"],
            "content": message["content"],
            "status": message["status"],
            "model": message.get("model"),
            "latency_ms": message.get("latency_ms"),
            "input_tokens": message.get("input_tokens"),
            "output_tokens": message.get("output_tokens"),
            "request_id": message.get("request_id"),
            "error": message.get("error"),
            "created_at": message["created_at"],
            "updated_at": message["updated_at"],
        }
    ).model_dump()


def derive_title(content: str) -> str:
    collapsed = " ".join(content.split())
    if len(collapsed) <= 72:
        return collapsed or "New conversation"
    return collapsed[:71].rstrip() + "…"


def sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _clip(text: str, limit: int = 500) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


async def _disconnected(check: Callable[[], Awaitable[bool]] | None) -> bool:
    if check is None:
        return False
    try:
        return await check()
    except Exception:
        return False
