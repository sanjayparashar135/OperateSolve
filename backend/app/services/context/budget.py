"""Model-aware conservative input-token budgets.

Official windows (checked 2026-10-25 against OpenAI model pages):
  gpt-6-luna: 1,050,000 context / 922,000 max input / 128,000 max output
  gpt-5-mini / gpt-5-nano: 400,000 context / 272,000 max input / 128,000 max output

OperateSolve uses a much smaller configurable input budget so interactive chats
stay cheap. The hard ceiling is still clamped to each model's documented limits
minus reserved output and reasoning headroom.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.services.llm.registry import ModelSpec


class ContextBudgetExceeded(Exception):
    """Mandatory prompt content (instructions + current user message) does not fit."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


@dataclass(frozen=True)
class ContextBudget:
    model: str
    context_window: int
    max_input_tokens: int
    max_output_tokens: int
    # Usable input tokens after clamping and output/reasoning reserves.
    input_budget: int
    reserved_output_tokens: int
    reasoning_reserve_tokens: int
    safety_margin: float
    summary_budget_tokens: int


def resolve_budget(spec: ModelSpec, settings: Settings) -> ContextBudget:
    reserved_output = min(max(settings.openai_max_output_tokens, 1), spec.max_output_tokens)
    reasoning_reserve = max(settings.context_reasoning_reserve_tokens, 0)
    headroom = reserved_output + reasoning_reserve
    model_input_cap = min(spec.max_input_tokens, max(spec.context_window - headroom, 1))
    configured = max(settings.context_max_input_tokens, 1)
    input_budget = min(configured, model_input_cap)
    return ContextBudget(
        model=spec.model_id,
        context_window=spec.context_window,
        max_input_tokens=spec.max_input_tokens,
        max_output_tokens=spec.max_output_tokens,
        input_budget=input_budget,
        reserved_output_tokens=reserved_output,
        reasoning_reserve_tokens=reasoning_reserve,
        safety_margin=max(settings.context_token_safety_margin, 1.0),
        summary_budget_tokens=max(settings.summarization_summary_budget_tokens, 64),
    )
