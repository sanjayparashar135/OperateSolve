"""Select the configured model provider."""

from app.config import Settings
from app.services.llm.base import LLMProvider
from app.services.llm.openai_provider import OpenAIResponsesProvider


def build_provider(settings: Settings) -> LLMProvider:
    if settings.llm_provider == "openai":
        return OpenAIResponsesProvider(settings)
    raise ValueError(
        f"Unsupported LLM provider '{settings.llm_provider}'. "
        "Register additional providers in app.services.llm.factory."
    )
