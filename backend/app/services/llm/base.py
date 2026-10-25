"""Provider-neutral streaming types.

Add a new model vendor by implementing LLMProvider and registering it in factory.py.
"""

from dataclasses import dataclass
from typing import AsyncIterator, Literal, Protocol


@dataclass(frozen=True)
class ChatTurn:
    role: Literal["user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMDelta:
    text: str


@dataclass(frozen=True)
class LLMDone:
    model: str | None
    input_tokens: int | None
    output_tokens: int | None
    provider_response_id: str | None = None


@dataclass(frozen=True)
class LLMError:
    message: str
    provider_response_id: str | None = None


LLMEvent = LLMDelta | LLMDone | LLMError


class LLMProvider(Protocol):
    def stream(
        self,
        turns: list[ChatTurn],
        *,
        model: str | None = None,
        instructions: str | None = None,
    ) -> AsyncIterator[LLMEvent]:
        """Yield text deltas, then a terminal done or error event."""
