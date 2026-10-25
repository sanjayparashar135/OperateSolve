"""Conversation and streaming message routes."""

from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends, HTTPException, Request
from starlette.responses import Response, StreamingResponse

from app.api.deps import get_operator, get_service
from app.schemas.chat import (
    ChatCreate,
    ChatRename,
    ConversationDetail,
    ConversationOut,
    MessageCreate,
)
from app.services.chat_service import ChatConflict, ChatNotFound, ChatService

router = APIRouter(prefix="/api", tags=["chats"])

SSE_HEADERS = {
    "Cache-Control": "no-cache, no-transform",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


@router.get("/chats", response_model=list[ConversationOut])
def list_chats(service: ChatService = Depends(get_service)) -> list[dict]:
    return service.list_conversations()


@router.post("/chats", response_model=ConversationOut, status_code=201)
def create_chat(payload: ChatCreate, service: ChatService = Depends(get_service)) -> dict:
    return service.create_conversation(payload.title)


@router.get("/chats/{conversation_id}", response_model=ConversationDetail)
def get_chat(conversation_id: str, service: ChatService = Depends(get_service)) -> dict:
    try:
        return service.get_detail(conversation_id)
    except ChatNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/chats/{conversation_id}", response_model=ConversationOut)
def rename_chat(
    conversation_id: str,
    payload: ChatRename,
    service: ChatService = Depends(get_service),
) -> dict:
    try:
        return service.rename_conversation(conversation_id, payload.title)
    except ChatNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.delete("/chats/{conversation_id}", status_code=204)
def delete_chat(conversation_id: str, service: ChatService = Depends(get_service)) -> Response:
    try:
        service.delete_conversation(conversation_id)
    except ChatNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(status_code=204)


@router.post("/chats/{conversation_id}/messages")
async def create_message(
    conversation_id: str,
    payload: MessageCreate,
    request: Request,
    service: ChatService = Depends(get_service),
    operator: str = Depends(get_operator),
) -> StreamingResponse:
    try:
        prepared = service.prepare_new_message(conversation_id, payload.content, payload.client_request_id)
    except ChatNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ChatConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    generator: AsyncIterator[str]
    if prepared.replay:
        generator = service.replay(prepared)
    else:
        generator = service.generate(prepared, request.is_disconnected, operator)
    return StreamingResponse(generator, media_type="text/event-stream", headers=SSE_HEADERS)


@router.post("/chats/{conversation_id}/messages/{message_id}/retry")
async def retry_message(
    conversation_id: str,
    message_id: str,
    request: Request,
    service: ChatService = Depends(get_service),
    operator: str = Depends(get_operator),
) -> StreamingResponse:
    try:
        prepared = service.prepare_retry(conversation_id, message_id)
    except ChatNotFound as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ChatConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return StreamingResponse(
        service.generate(prepared, request.is_disconnected, operator),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
