"""Runtime configuration. Secrets stay in the backend environment."""

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_api_key: str = ""
    # Baseline model: the Auto fallback. Must be a model in the registry.
    openai_model: str = "gpt-6-luna"
    openai_reasoning_effort: str = "low"
    openai_max_output_tokens: int = 2048
    # Comma-separated registry keys or model ids to turn off, e.g. "mini,gpt-5-nano".
    disabled_models: str = ""
    # At startup, ask the provider which registered models this key can use (no tokens billed).
    model_access_check: bool = True
    database_path: str = "data/operatesolve.db"
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    log_level: str = "INFO"
    llm_provider: str = "openai"
    # Relative paths resolve from the backend directory.
    pricing_config_path: str = "pricing.json"

    @property
    def pricing_file(self) -> Path:
        path = Path(self.pricing_config_path)
        return path if path.is_absolute() else BACKEND_DIR / path

    @property
    def cors_origin_list(self) -> list[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    @property
    def disabled_model_list(self) -> list[str]:
        return [item.strip() for item in self.disabled_models.split(",") if item.strip()]

    @property
    def llm_configured(self) -> bool:
        return bool(self.openai_api_key.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()
