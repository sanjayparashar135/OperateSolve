"""Deterministic model selection.

Rules, in order:
1. A manual selection (a registry key such as "mini") always wins. If that model
   is unavailable the request is rejected; there is no substitute.
2. "auto" routes by the task mode the user picked:
   simple -> nano, general -> luna, detailed -> mini.
3. If the Auto target is unavailable, the baseline model (OPENAI_MODEL,
   gpt-6-luna by default) is used and the reason says so.
4. Routing happens once, before the provider is called. An API error is
   reported as an error; it never triggers a second attempt on another model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from app.services.llm.registry import ModelRegistry, ModelSpec

AUTO = "auto"
TASK_MODES = ("simple", "general", "detailed")
DEFAULT_TASK_MODE = "general"
AUTO_RULES: dict[str, str] = {"simple": "nano", "general": "luna", "detailed": "mini"}

RoutingMode = Literal["manual", "auto", "fallback"]


class InvalidModelSelection(ValueError):
    pass


class ModelUnavailable(Exception):
    pass


@dataclass(frozen=True)
class RoutingDecision:
    selection: str
    task_mode: str | None
    routing_mode: RoutingMode
    requested_model: str
    model: str
    model_key: str
    reason: str
    reasoning_effort: str | None

    def public(self) -> dict[str, str | None]:
        return {
            "selection": self.selection,
            "task_mode": self.task_mode,
            "routing_mode": self.routing_mode,
            "requested_model": self.requested_model,
            "model": self.model,
            "reason": self.reason,
        }


class ModelRouter:
    def __init__(self, registry: ModelRegistry, *, reasoning_effort: str) -> None:
        self.registry = registry
        self.reasoning_effort = reasoning_effort
        missing = {key for key in AUTO_RULES.values() if registry.get(key) is None}
        if missing:
            raise ValueError(f"Auto routing targets are not registered: {', '.join(sorted(missing))}")

    def normalize(self, selection: str | None, task_mode: str | None) -> tuple[str, str]:
        """Validate a stored or requested preference. Raises InvalidModelSelection."""

        chosen = (selection or AUTO).strip().lower()
        mode = (task_mode or DEFAULT_TASK_MODE).strip().lower()
        if mode not in TASK_MODES:
            raise InvalidModelSelection(f"Unknown task mode '{task_mode}'. Use one of: {', '.join(TASK_MODES)}.")
        if chosen != AUTO:
            spec = self.registry.get(chosen)
            if spec is None or spec.key != chosen:
                options = ", ".join([AUTO, *self.registry.keys()])
                raise InvalidModelSelection(f"Unknown model selection '{selection}'. Use one of: {options}.")
        return chosen, mode

    def route(self, selection: str | None, task_mode: str | None) -> RoutingDecision:
        chosen, mode = self.normalize(selection, task_mode)
        if chosen != AUTO:
            spec = self.registry.get(chosen)
            assert spec is not None
            status = self.registry.status(spec)
            if not status.available:
                raise ModelUnavailable(
                    f"{_label(spec)} is unavailable: {status.reason} Choose another model."
                )
            return self._decision(chosen, None, "manual", spec, spec, f"Manual selection: {_label(spec)}.")

        target = self.registry.get(AUTO_RULES[mode])
        assert target is not None
        rule = f"Auto · {mode.capitalize()} task → {_label(target)}"
        status = self.registry.status(target)
        if status.available:
            return self._decision(chosen, mode, "auto", target, target, f"{rule}.")

        baseline = self.registry.baseline
        baseline_status = self.registry.status(baseline)
        if not baseline_status.available:
            raise ModelUnavailable(
                f"{rule} is unavailable ({_sentence(status.reason)}) and the fallback "
                f"{_label(baseline)} is unavailable too ({_sentence(baseline_status.reason)})."
            )
        return self._decision(
            chosen,
            mode,
            "fallback",
            target,
            baseline,
            f"{rule} is unavailable ({_sentence(status.reason)}); fell back to {_label(baseline)}.",
        )

    def _decision(
        self,
        selection: str,
        task_mode: str | None,
        routing_mode: RoutingMode,
        requested: ModelSpec,
        actual: ModelSpec,
        reason: str,
    ) -> RoutingDecision:
        return RoutingDecision(
            selection=selection,
            task_mode=task_mode,
            routing_mode=routing_mode,
            requested_model=requested.model_id,
            model=actual.model_id,
            model_key=actual.key,
            reason=reason,
            reasoning_effort=self.registry.reasoning_effort(actual, self.reasoning_effort),
        )


def _label(spec: ModelSpec) -> str:
    return f"{spec.display_name} ({spec.model_id})"


def _sentence(text: str | None) -> str:
    return (text or "unavailable").rstrip(".")
