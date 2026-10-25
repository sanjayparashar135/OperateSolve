"""Request and response models."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


KEY_PATTERN = r"^[A-Za-z0-9._-]+$"


def _optional_key():
    return Field(default=None, min_length=1, max_length=40, pattern=KEY_PATTERN)


class ModelPreference(BaseModel):
    """model_selection is "auto" or a registry key; the chat service validates both values."""

    model_selection: str = Field(min_length=1, max_length=40, pattern=KEY_PATTERN)
    task_mode: str | None = _optional_key()


class ChatCreate(BaseModel):
    title: str | None = None
    model_selection: str | None = _optional_key()
    task_mode: str | None = _optional_key()

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
    # When present, saved as the conversation's preference before this turn is routed.
    model_selection: str | None = _optional_key()
    task_mode: str | None = _optional_key()

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
    model_selection: str = "auto"
    task_mode: str = "general"
    created_at: str
    updated_at: str


class MessageOut(BaseModel):
    id: str
    conversation_id: str
    role: Literal["user", "assistant"]
    content: str
    status: Literal["complete", "streaming", "error", "stopped"]
    model: str | None = None
    requested_model: str | None = None
    routing_mode: str | None = None
    routing_reason: str | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
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


class ModelRatesOut(BaseModel):
    input: float
    cached_input: float | None
    cache_write: float | None
    output: float


class ModelOut(BaseModel):
    key: str
    model_id: str
    provider: str
    display_name: str
    description: str
    available: bool
    unavailable_reason: str | None
    baseline: bool
    deprecated: bool
    shutdown_date: str | None
    replacement: str | None
    capabilities: list[str]
    reasoning_efforts: list[str]
    reasoning_effort: str | None
    context_window: int
    max_output_tokens: int
    priced: bool
    rates: ModelRatesOut | None
    docs_url: str | None


class AutoRule(BaseModel):
    task_mode: str
    label: str
    target: str
    resolved_model: str | None
    routing_mode: str | None
    reason: str


class ModelCatalog(BaseModel):
    default_selection: str
    default_task_mode: str
    baseline_model: str
    access_checked_at: str | None
    models: list[ModelOut]
    auto_rules: list[AutoRule]
