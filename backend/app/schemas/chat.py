"""Request and response models."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ChatCreate(BaseModel):
    title: str | None = None

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        if len(cleaned) > 120:
            raise ValueError("Title must be 120 characters or fewer")
        return cleaned


class ChatRename(BaseModel):
    title: str = Field(min_length=1, max_length=120)

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Title cannot be blank")
        return cleaned


class MessageCreate(BaseModel):
    content: str = Field(min_length=1, max_length=20000)
    client_request_id: str = Field(min_length=8, max_length=80, pattern=r"^[A-Za-z0-9._:-]+$")

    @field_validator("content")
    @classmethod
    def clean_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Message content cannot be blank")
        return cleaned


class ConversationOut(BaseModel):
    id: str
    title: str
    title_customized: bool
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: str
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str
    status: Literal["complete", "streaming", "error", "stopped"]
    model: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    request_id: str | None = None
    error: str | None = None
    created_at: str
    updated_at: str


class ConversationDetail(BaseModel):
    conversation: ConversationOut
    messages: list[MessageOut]


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    database: Literal["ok", "error"]
    llm_configured: bool
    model: str
    provider: str
