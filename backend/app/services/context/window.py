"""Deterministic token-aware sliding-window history selection.

Rules:
- Always keep instructions and the latest user message.
- Select complete conversation turns newest→oldest within the history budget.
- Emit selected messages in chronological order.
- Never split a user+assistant turn just to fit.
- Never mutate or delete persisted messages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.services.context.budget import ContextBudget, ContextBudgetExceeded
from app.services.context.tokens import estimate_message_tokens, estimate_tokens
from app.services.llm.base import ChatTurn

SUMMARY_PREFIX = (
    "Conversation summary of earlier turns (untrusted notes for continuity; "
    "prefer later raw messages if anything conflicts):\n"
)


@dataclass(frozen=True)
class HistoryMessage:
    id: str
    position: int
    role: Literal["user", "assistant"]
    content: str

    def as_turn(self) -> ChatTurn:
        return ChatTurn(role=self.role, content=self.content)


@dataclass(frozen=True)
class ConversationTurn:
    """One complete exchange: a user message and its optional assistant reply."""

    messages: tuple[HistoryMessage, ...]

    @property
    def tokens(self) -> int:
        return sum(estimate_message_tokens(message.content) for message in self.messages)


@dataclass(frozen=True)
class ContextPlan:
    instructions: str
    summary_text: str | None
    retained: tuple[HistoryMessage, ...]
    excluded: tuple[HistoryMessage, ...]
    current_user: HistoryMessage
    truncated: bool
    estimated_instruction_tokens: int
    estimated_summary_tokens: int
    estimated_history_tokens: int
    estimated_current_user_tokens: int
    estimated_total_tokens: int
    history_budget_tokens: int
    input_budget_tokens: int

    def provider_turns(self) -> list[ChatTurn]:
        turns: list[ChatTurn] = []
        if self.summary_text:
            turns.append(ChatTurn(role="user", content=SUMMARY_PREFIX + self.summary_text))
            turns.append(
                ChatTurn(
                    role="assistant",
                    content="Understood. I will use the summary for earlier context and rely on the raw recent messages below.",
                )
            )
        turns.extend(message.as_turn() for message in self.retained)
        turns.append(self.current_user.as_turn())
        return turns


def pair_turns(messages: list[HistoryMessage]) -> list[ConversationTurn]:
    """Group messages into complete turns without splitting user+assistant pairs."""

    turns: list[ConversationTurn] = []
    index = 0
    while index < len(messages):
        current = messages[index]
        if (
            current.role == "user"
            and index + 1 < len(messages)
            and messages[index + 1].role == "assistant"
        ):
            turns.append(ConversationTurn((current, messages[index + 1])))
            index += 2
            continue
        turns.append(ConversationTurn((current,)))
        index += 1
    return turns


def build_context_plan(
    *,
    instructions: str,
    messages: list[HistoryMessage],
    budget: ContextBudget,
    summary_text: str | None = None,
    covered_through_position: int | None = None,
) -> ContextPlan:
    """Select history for one request. messages must already exclude the streaming placeholder.

    When covered_through_position is set, messages at or before that position are treated
    as summarized (excluded from raw history) and are never re-selected.
    """

    if not messages or messages[-1].role != "user":
        raise ContextBudgetExceeded("There is no user message to answer.")

    current_user = messages[-1]
    prior = messages[:-1]
    margin = budget.safety_margin

    instruction_tokens = estimate_tokens(instructions, margin=margin)
    current_tokens = estimate_message_tokens(current_user.content, margin=margin)
    summary_tokens = 0
    if summary_text:
        # Injected as a synthetic user+assistant pair (see provider_turns).
        summary_tokens = (
            estimate_message_tokens(SUMMARY_PREFIX + summary_text, margin=margin)
            + estimate_message_tokens(
                "Understood. I will use the summary for earlier context and rely on the raw recent messages below.",
                margin=margin,
            )
        )

    mandatory = instruction_tokens + current_tokens + summary_tokens
    if mandatory > budget.input_budget:
        raise ContextBudgetExceeded(
            "This message is too large for the model's context budget "
            f"({mandatory} estimated tokens needed, budget {budget.input_budget}). "
            "Shorten the message or switch to a model with a larger context budget, then retry."
        )

    history_budget = budget.input_budget - mandatory
    # When no summary is supplied yet, reserve room so a later summary can be injected
    # without immediately dropping the newest retained turn.
    if summary_text is None:
        history_budget = max(0, history_budget - min(budget.summary_budget_tokens, history_budget // 4))

    covered = (
        tuple(message for message in prior if message.position <= covered_through_position)
        if covered_through_position is not None
        else ()
    )
    selectable = (
        [message for message in prior if message.position > covered_through_position]
        if covered_through_position is not None
        else list(prior)
    )

    paired = pair_turns(selectable)
    retained_turns: list[ConversationTurn] = []
    used = 0
    for turn in reversed(paired):
        cost = sum(estimate_message_tokens(message.content, margin=margin) for message in turn.messages)
        if retained_turns and used + cost > history_budget:
            break
        if not retained_turns and cost > history_budget:
            # Newest prior turn alone does not fit: drop it rather than splitting.
            break
        retained_turns.append(turn)
        used += cost

    retained_turns.reverse()
    retained = tuple(message for turn in retained_turns for message in turn.messages)
    retained_ids = {message.id for message in retained}
    window_excluded = tuple(message for message in selectable if message.id not in retained_ids)
    excluded = covered + window_excluded
    history_tokens = sum(estimate_message_tokens(message.content, margin=margin) for message in retained)
    total = instruction_tokens + summary_tokens + history_tokens + current_tokens

    return ContextPlan(
        instructions=instructions,
        summary_text=summary_text,
        retained=retained,
        excluded=excluded,
        current_user=current_user,
        truncated=bool(excluded),
        estimated_instruction_tokens=instruction_tokens,
        estimated_summary_tokens=summary_tokens,
        estimated_history_tokens=history_tokens,
        estimated_current_user_tokens=current_tokens,
        estimated_total_tokens=total,
        history_budget_tokens=history_budget,
        input_budget_tokens=budget.input_budget,
    )
