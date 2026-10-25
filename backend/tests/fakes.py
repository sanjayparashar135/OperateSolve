"""In-memory model provider for API tests."""

from app.services.llm.base import ChatTurn, LLMDelta, LLMDone, LLMError, LLMEvent, LLMUsage


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.chunks = ["Hello ", "operator"]
        self.fail = False
        self.request_sent = True
        self.input_tokens: int | None = 12
        self.output_tokens: int | None = 5
        self.cached_input_tokens: int | None = None
        self.cache_write_tokens: int | None = None
        self.reasoning_tokens: int | None = None
        # None echoes the requested model, as the real API does.
        self.model: str | None = None
        self.fail_models: set[str] = set()

    def usage(self) -> LLMUsage | None:
        if self.input_tokens is None and self.output_tokens is None:
            return None
        return LLMUsage(
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            cached_input_tokens=self.cached_input_tokens,
            cache_write_tokens=self.cache_write_tokens,
            reasoning_tokens=self.reasoning_tokens,
        )

    async def stream(
        self,
        turns: list[ChatTurn],
        *,
        model: str | None = None,
        instructions: str | None = None,
        reasoning_effort: str | None = None,
    ):
        self.calls.append(
            {
                "turns": list(turns),
                "model": model,
                "instructions": instructions,
                "reasoning_effort": reasoning_effort,
            }
        )
        if self.fail or model in self.fail_models:
            yield LLMError(message="The model request failed. You can retry.", request_sent=self.request_sent)
            return
        for chunk in self.chunks:
            yield LLMDelta(text=chunk)
        yield LLMDone(
            model=self.model or model,
            usage=self.usage(),
            provider_response_id="resp_test",
        )
