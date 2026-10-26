"""Prompt-cache request settings for every registered model, in one place.

OpenAI caches prompt prefixes automatically; nothing is cached by this app.
These settings only pin the documented defaults and add the routing hint that
older models benefit from. Checked against the OpenAI prompt caching guide and
model pages on 2026-10-25:

- gpt-6-luna (GPT-5.6 and later): implicit mode puts a breakpoint at the end of
  the latest user message, and earlier user-message endings are lookup
  boundaries, so each turn can reuse the previous turn's prefix. The default
  lifetime is "30m" (writes 1.25x input, reads 0.1x input). The minimum
  cacheable prefix is 1,024 visible tokens. prompt_cache_key only separates
  cache accounting on these models, so it is not sent.
- gpt-5-nano (earlier model): implicit breakpoints at fixed intervals, no
  cache-write surcharge. It is not on the "24h" retention list, so
  prompt_cache_retention is left at its in_memory default. A stable
  prompt_cache_key helps route related requests to the same cache.
- gpt-5-mini: its model page does not list prompt_caching, so nothing is sent.
  Any cached tokens it reports are still recorded and priced.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

# Stable across releases on purpose: changing it moves traffic to a cold cache.
PROMPT_CACHE_KEY = "operatesolve-chat-v1"

_SNAPSHOT_SUFFIX = re.compile(r"-\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class PromptCachePolicy:
    # prompt_cache_options.ttl; only GPT-5.6 and later accept prompt_cache_options.
    ttl: str | None = None
    send_cache_key: bool = False
    # Documented minimum for a cacheable prefix; None means it varies by request settings.
    min_prefix_tokens: int | None = None


NO_POLICY = PromptCachePolicy()

POLICIES: dict[str, PromptCachePolicy] = {
    "gpt-6-luna": PromptCachePolicy(ttl="30m", min_prefix_tokens=1024),
    "gpt-5-nano": PromptCachePolicy(send_cache_key=True),
    "gpt-5-mini": NO_POLICY,
}


def policy_for(model: str | None) -> PromptCachePolicy:
    """Exact model id first, then the base id of a dated snapshot. Unknown models get no settings."""

    if not model:
        return NO_POLICY
    key = model.strip().lower()
    return POLICIES.get(key) or POLICIES.get(_SNAPSHOT_SUFFIX.sub("", key)) or NO_POLICY


def request_params(model: str | None) -> dict[str, Any]:
    """Extra Responses API arguments for this model's prompt caching."""

    policy = policy_for(model)
    params: dict[str, Any] = {}
    if policy.ttl:
        params["prompt_cache_options"] = {"mode": "implicit", "ttl": policy.ttl}
    if policy.send_cache_key:
        params["prompt_cache_key"] = PROMPT_CACHE_KEY
    return params
