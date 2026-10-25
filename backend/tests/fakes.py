"""In-memory model provider for API tests."""

from app.services.llm.base import ChatTurn, LLMDelta, LLMDone, LLMError, LLMEvent


class FakeProvider:
    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.chunks = ["Hello ", "operator"]
        self.fail = False
        self.input_tokens: int | None = 12
        self.output_tokens: int | None = 5
        self.model = "gpt-6-luna"

    async def stream(self, turns: list[ChatTurn], *, model: str | None = None, instructions: str | None = None):
        self.calls.append(
            {
                "turns": list(turns),
                "model": model,
                "instructions": instructions,
            }
        )
        if self.fail:
            yield LLMError(message="The model request failed. You can retry.")
            return
        for chunk in self.chunks:
            yield LLMDelta(text=chunk)
        yield LLMDone(
            model=self.model,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            provider_response_id="resp_test",
        )
