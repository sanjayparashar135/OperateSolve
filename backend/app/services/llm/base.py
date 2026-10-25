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
class LLMUsage:
    """Token counts reported by the provider. Unreported values stay None.

    cached_input_tokens and cache_write_tokens are subsets of input_tokens.
    reasoning_tokens is a subset of output_tokens.
    """

    input_tokens: int | None
    output_tokens: int | None
    cached_input_tokens: int | None = None
    cache_write_tokens: int | None = None
    reasoning_tokens: int | None = None

    @property
    def reported(self) -> bool:
        return self.input_tokens is not None and self.output_tokens is not None


@dataclass(frozen=True)
class LLMDone:
    model: str | None
    usage: LLMUsage | None
    provider_response_id: str | None = None


@dataclass(frozen=True)
class LLMError:
    message: str
    provider_response_id: str | None = None
    usage: LLMUsage | None = None
    # False when the provider refused locally (e.g. no API key), so nothing could be billed.
    request_sent: bool = True


LLMEvent = LLMDelta | LLMDone | LLMError


class LLMProvider(Protocol):
    """Providers may also define `async check_model_access(model_id) -> bool | None`.

    It should cost nothing and return None when access cannot be determined.
    """

    def stream(
        self,
        turns: list[ChatTurn],
        *,
        model: str | None = None,
        instructions: str | None = None,
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[LLMEvent]:
        """Yield text deltas, then a terminal done or error event.

        reasoning_effort was already validated for the model by the registry;
        None means use the provider's configured default.
        """
