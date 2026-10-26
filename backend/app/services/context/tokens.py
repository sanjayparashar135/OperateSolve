"""Local token estimation for budgeting. Not a substitute for provider billing counts.

Uses a fixed characters-per-token heuristic with an optional safety margin.
OpenAI's reported usage remains the source of truth for cost accounting.
"""

from __future__ import annotations

from typing import Protocol

# Conservative English/code average. Real BPE counts vary by language and model.
CHARS_PER_TOKEN = 4.0
# Per-message framing overhead (role labels, separators) that the heuristic misses.
MESSAGE_OVERHEAD_TOKENS = 4


class _HasContent(Protocol):
    content: str


def estimate_tokens(text: str, *, margin: float = 1.0) -> int:
    """Estimate tokens for a string. margin > 1 pads the estimate for safety."""

    if not text:
        return 0
    raw = max(1, int((len(text) / CHARS_PER_TOKEN) + 0.999))
    if margin <= 1.0:
        return raw
    return max(raw, int(raw * margin + 0.999))


def estimate_message_tokens(content: str, *, margin: float = 1.0) -> int:
    """Estimate tokens for one chat message, including framing overhead."""

    return estimate_tokens(content, margin=margin) + MESSAGE_OVERHEAD_TOKENS


def estimate_messages(messages: list[_HasContent], *, margin: float = 1.0) -> int:
    return sum(estimate_message_tokens(message.content, margin=margin) for message in messages)
