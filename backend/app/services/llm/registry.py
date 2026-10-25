"""Models OperateSolve may call, and whether each one can be used right now.

Entries are provider-neutral. A future Azure OpenAI or Bedrock deployment adds
a ModelSpec with its own provider name; the router only reads this registry.
Facts below were checked against the OpenAI model, pricing, deprecation, and
reasoning docs on 2026-10-25. Prices live in pricing.json, not here.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")
GPT6_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")
GPT5_EFFORTS = ("minimal", "low", "medium", "high")


@dataclass(frozen=True)
class ModelSpec:
    key: str
    model_id: str
    provider: str
    display_name: str
    description: str
    reasoning_efforts: tuple[str, ...]
    # Used when the configured effort is not accepted by this model.
    fallback_reasoning_effort: str
    context_window: int
    max_input_tokens: int
    max_output_tokens: int
    capabilities: tuple[str, ...] = ("text", "responses_api", "streaming", "reasoning")
    deprecated: bool = False
    shutdown_date: date | None = None
    replacement: str | None = None
    docs_url: str | None = None


MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(
        key="luna",
        model_id="gpt-6-luna",
        provider="openai",
        display_name="Luna",
        description="Balanced default. Current efficient GPT-6 model.",
        reasoning_efforts=GPT6_EFFORTS,
        fallback_reasoning_effort="low",
        context_window=1_050_000,
        max_input_tokens=922_000,
        max_output_tokens=128_000,
        capabilities=("text", "responses_api", "streaming", "reasoning", "prompt_caching"),
        docs_url="https://developers.openai.com/api/docs/models/gpt-6-luna",
    ),
    ModelSpec(
        key="mini",
        model_id="gpt-5-mini",
        provider="openai",
        display_name="Mini",
        description="Older GPT-5 small model for experiments with longer answers.",
        reasoning_efforts=GPT5_EFFORTS,
        fallback_reasoning_effort="minimal",
        context_window=400_000,
        max_input_tokens=272_000,
        max_output_tokens=128_000,
        deprecated=True,
        shutdown_date=date(2026, 12, 11),
        replacement="gpt-5.6-terra",
        docs_url="https://developers.openai.com/api/docs/models/gpt-5-mini",
    ),
    ModelSpec(
        key="nano",
        model_id="gpt-5-nano",
        provider="openai",
        display_name="Nano",
        description="Cheapest GPT-5 model for short, simple tasks.",
        reasoning_efforts=GPT5_EFFORTS,
        fallback_reasoning_effort="minimal",
        context_window=400_000,
        max_input_tokens=272_000,
        max_output_tokens=128_000,
        capabilities=("text", "responses_api", "streaming", "reasoning", "prompt_caching"),
        deprecated=True,
        shutdown_date=date(2026, 12, 11),
        replacement="gpt-5.6-luna",
        docs_url="https://developers.openai.com/api/docs/models/gpt-5-nano",
    ),
)


@dataclass(frozen=True)
class ModelStatus:
    spec: ModelSpec
    available: bool
    reason: str | None = None


class ModelRegistry:
    def __init__(
        self,
        specs: Iterable[ModelSpec],
        *,
        provider: str,
        baseline_model: str,
        disabled: Iterable[str] = (),
        today: Callable[[], date] | None = None,
    ) -> None:
        self._specs = {spec.key: spec for spec in specs}
        self._by_model_id = {spec.model_id: spec for spec in self._specs.values()}
        self.provider = provider
        self._disabled = {item.strip().lower() for item in disabled if item.strip()}
        self._today = today or (lambda: datetime.now(timezone.utc).date())
        self._access: dict[str, bool] = {}
        self.access_checked_at: str | None = None
        baseline = self.get(baseline_model)
        if baseline is None:
            known = ", ".join(spec.model_id for spec in self._specs.values())
            raise ValueError(f"OPENAI_MODEL '{baseline_model}' is not a registered model. Use one of: {known}.")
        self.baseline = baseline

    def get(self, key_or_model_id: str | None) -> ModelSpec | None:
        if not key_or_model_id:
            return None
        value = key_or_model_id.strip().lower()
        return self._specs.get(value) or self._by_model_id.get(value)

    def keys(self) -> list[str]:
        return list(self._specs)

    def display_name(self, model_id: str | None) -> str | None:
        """Display name for a model id the provider reported, including dated snapshots."""

        if not model_id:
            return None
        spec = self.get(model_id) or self.get(_SNAPSHOT_SUFFIX.sub("", model_id.strip().lower()))
        return spec.display_name if spec else None

    def status(self, spec: ModelSpec) -> ModelStatus:
        if spec.provider != self.provider:
            return ModelStatus(spec, False, f"Needs the {spec.provider} provider; LLM_PROVIDER is {self.provider}.")
        if spec.key in self._disabled or spec.model_id in self._disabled:
            return ModelStatus(spec, False, "Disabled by DISABLED_MODELS in backend/.env.")
        if spec.shutdown_date is not None and self._today() >= spec.shutdown_date:
            return ModelStatus(spec, False, f"Retired by the provider on {spec.shutdown_date.isoformat()}.")
        if self._access.get(spec.key) is False:
            return ModelStatus(spec, False, "This API key cannot access the model (checked at startup).")
        return ModelStatus(spec, True)

    def statuses(self) -> list[ModelStatus]:
        return [self.status(spec) for spec in self._specs.values()]

    def record_access(self, key: str, accessible: bool | None) -> None:
        """None means the check was inconclusive, so availability is left unchanged."""

        if accessible is None:
            self._access.pop(key, None)
        else:
            self._access[key] = accessible

    def reasoning_effort(self, spec: ModelSpec, configured: str) -> str | None:
        effort = configured.strip().lower()
        if not effort:
            return None
        if effort in spec.reasoning_efforts:
            return effort
        return spec.fallback_reasoning_effort


async def check_access(registry: ModelRegistry, provider: Any, *, timeout: float = 15.0) -> None:
    """Ask the provider which registered models this key can use. Costs no tokens.

    Providers without a check_model_access method are skipped.
    """

    checker = getattr(provider, "check_model_access", None)
    if checker is None:
        return
    targets = [status.spec for status in registry.statuses() if status.available]

    async def one(spec: ModelSpec) -> None:
        try:
            accessible = await checker(spec.model_id)
        except Exception:
            accessible = None
        registry.record_access(spec.key, accessible)
        logger.info(
            "model_access_checked model=%s accessible=%s",
            spec.model_id,
            "unknown" if accessible is None else accessible,
        )

    try:
        await asyncio.wait_for(asyncio.gather(*(one(spec) for spec in targets)), timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning("model_access_check_timed_out")
    registry.access_checked_at = datetime.now(timezone.utc).isoformat()
