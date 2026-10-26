"""Phase 2.5B — sequential vs parallel LLM execution experiment.

Three independent analysis tasks share one fictional incident. Sequential mode
runs them one after another; parallel mode runs them concurrently with a
bounded semaphore. Token usage and cost are recorded once per call under
purpose=parallel_experiment. Parallelism can reduce wall-clock latency; it does
not reduce total tokens or cost.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

from app.config import Settings
from app.db.database import utc_now
from app.db.repository import Repository
from app.services.llm.base import ChatTurn, LLMComplete, LLMError, LLMUsage
from app.services.llm.registry import ModelRegistry
from app.services.pricing import PricingTable, estimate_cost

logger = logging.getLogger(__name__)

PURPOSE = "parallel_experiment"
TaskStatus = Literal["complete", "error", "timeout"]


@dataclass(frozen=True)
class ExperimentTask:
    task_id: str
    task_name: str
    instructions: str
    prompt: str


SHARED_INSTRUCTIONS = (
    "You are an enterprise operations analyst for OperateSolve. "
    "Answer only the assigned task for the fictional incident. "
    "Be concise and operational. Do not invent telemetry you were not given."
)

EXPERIMENT_TASKS: tuple[ExperimentTask, ...] = (
    ExperimentTask(
        task_id="root_causes",
        task_name="Identify probable root causes",
        instructions=SHARED_INSTRUCTIONS,
        prompt=(
            "Task: Identify the most probable root causes for this fictional incident.\n"
            "List 2–4 concise causes ranked by likelihood. One short sentence each.\n\n"
            "Incident:\n{description}"
        ),
    ),
    ExperimentTask(
        task_id="business_impact",
        task_name="Assess business impact",
        instructions=SHARED_INSTRUCTIONS,
        prompt=(
            "Task: Assess the business impact of this fictional incident.\n"
            "Cover customer experience, revenue/operations risk, and urgency in 3–5 short sentences.\n\n"
            "Incident:\n{description}"
        ),
    ),
    ExperimentTask(
        task_id="remediation",
        task_name="Recommend remediation steps",
        instructions=SHARED_INSTRUCTIONS,
        prompt=(
            "Task: Recommend remediation steps for this fictional incident.\n"
            "Give 3–5 ordered, actionable steps. Note any step that needs human approval.\n\n"
            "Incident:\n{description}"
        ),
    ),
)


class ParallelExperimentError(Exception):
    """Raised when the experiment cannot start (configuration / model issues)."""

    def __init__(self, message: str, *, code: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class ParallelExperimentService:
    def __init__(
        self,
        repository: Repository,
        provider: Any,
        settings: Settings,
        registry: ModelRegistry,
        pricing: PricingTable | None = None,
    ) -> None:
        self.repo = repository
        self.provider = provider
        self.settings = settings
        self.registry = registry
        self.pricing = pricing

    async def run(self, description: str, mode: Literal["sequential", "parallel"]) -> dict[str, Any]:
        complete = getattr(self.provider, "complete", None)
        if complete is None:
            raise ParallelExperimentError(
                "The configured LLM provider does not support non-streaming completions.",
                code="unsupported_provider",
                status_code=409,
            )

        model_id = self.settings.openai_model
        spec = self.registry.get(model_id)
        if spec is None:
            raise ParallelExperimentError(
                f"Configured model '{model_id}' is not in the model registry.",
                code="unknown_model",
                status_code=409,
            )
        status = self.registry.status(spec)
        if not status.available:
            raise ParallelExperimentError(
                status.reason or f"Model '{spec.model_id}' is unavailable.",
                code="model_unavailable",
                status_code=409,
            )

        effort = self.registry.reasoning_effort(spec, self.settings.openai_reasoning_effort)
        max_output = self.settings.parallel_experiment_max_output_tokens
        timeout_s = self.settings.parallel_experiment_timeout_seconds
        concurrency = max(1, self.settings.parallel_experiment_max_concurrency)

        logger.info(
            "parallel_experiment_started mode=%s model=%s tasks=%s timeout_s=%s concurrency=%s",
            mode,
            spec.model_id,
            len(EXPERIMENT_TASKS),
            timeout_s,
            concurrency,
        )

        wall_started = time.perf_counter()
        if mode == "sequential":
            task_results = []
            for task in EXPERIMENT_TASKS:
                task_results.append(
                    await self._run_task(
                        task,
                        description=description,
                        model=spec.model_id,
                        reasoning_effort=effort,
                        max_output_tokens=max_output,
                        timeout_s=timeout_s,
                    )
                )
        else:
            semaphore = asyncio.Semaphore(concurrency)

            async def bounded(task: ExperimentTask) -> dict[str, Any]:
                async with semaphore:
                    return await self._run_task(
                        task,
                        description=description,
                        model=spec.model_id,
                        reasoning_effort=effort,
                        max_output_tokens=max_output,
                        timeout_s=timeout_s,
                    )

            task_results = list(await asyncio.gather(*(bounded(task) for task in EXPERIMENT_TASKS)))

        total_latency_ms = int((time.perf_counter() - wall_started) * 1000)
        totals = self._aggregate(task_results, total_latency_ms)

        logger.info(
            "parallel_experiment_finished mode=%s model=%s total_latency_ms=%s "
            "successful=%s failed=%s input_tokens=%s output_tokens=%s",
            mode,
            spec.model_id,
            totals["latency_ms"],
            totals["successful_tasks"],
            totals["failed_tasks"],
            totals["input_tokens"],
            totals["output_tokens"],
        )

        return {
            "mode": mode,
            "model": spec.model_id,
            "tasks": task_results,
            "totals": totals,
        }

    async def _run_task(
        self,
        task: ExperimentTask,
        *,
        description: str,
        model: str,
        reasoning_effort: str | None,
        max_output_tokens: int,
        timeout_s: float,
    ) -> dict[str, Any]:
        request_id = str(uuid.uuid4())
        started_at = utc_now()
        started = time.perf_counter()
        prompt = task.prompt.format(description=description)
        turns = [ChatTurn(role="user", content=prompt)]

        try:
            result = await asyncio.wait_for(
                self.provider.complete(
                    turns,
                    model=model,
                    instructions=task.instructions,
                    reasoning_effort=reasoning_effort,
                    max_output_tokens=max_output_tokens,
                ),
                timeout=timeout_s,
            )
        except TimeoutError:
            latency_ms = int((time.perf_counter() - started) * 1000)
            message = f"Task timed out after {timeout_s:g}s."
            self._record_usage(
                request_id=request_id,
                model=model,
                status="error",
                usage=None,
                duration_ms=latency_ms,
                started_at=started_at,
                provider_response_id=None,
                routing_reason=f"{task.task_id}: timeout",
            )
            return self._task_payload(
                task,
                request_id=request_id,
                status="timeout",
                result=None,
                error=message,
                latency_ms=latency_ms,
                usage=None,
                cost=None,
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - started) * 1000)
            message = f"Task failed unexpectedly: {exc}"
            self._record_usage(
                request_id=request_id,
                model=model,
                status="error",
                usage=None,
                duration_ms=latency_ms,
                started_at=started_at,
                provider_response_id=None,
                routing_reason=f"{task.task_id}: {exc}",
            )
            return self._task_payload(
                task,
                request_id=request_id,
                status="error",
                result=None,
                error=message,
                latency_ms=latency_ms,
                usage=None,
                cost=None,
            )

        latency_ms = int((time.perf_counter() - started) * 1000)
        usage = getattr(result, "usage", None)
        model_name = getattr(result, "model", None) or model
        provider_response_id = getattr(result, "provider_response_id", None)

        if isinstance(result, LLMError):
            # Still record once when the provider was contacted (or refused locally).
            self._record_usage(
                request_id=request_id,
                model=model_name,
                status="error",
                usage=usage,
                duration_ms=latency_ms,
                started_at=started_at,
                provider_response_id=provider_response_id,
                routing_reason=f"{task.task_id}: {result.message}",
            )
            return self._task_payload(
                task,
                request_id=request_id,
                status="error",
                result=None,
                error=result.message,
                latency_ms=latency_ms,
                usage=usage,
                cost=self._estimate_cost(model_name, usage),
            )

        if not isinstance(result, LLMComplete):
            self._record_usage(
                request_id=request_id,
                model=model_name,
                status="error",
                usage=usage,
                duration_ms=latency_ms,
                started_at=started_at,
                provider_response_id=provider_response_id,
                routing_reason=f"{task.task_id}: unexpected provider result",
            )
            return self._task_payload(
                task,
                request_id=request_id,
                status="error",
                result=None,
                error="Unexpected provider result for experiment task.",
                latency_ms=latency_ms,
                usage=usage,
                cost=None,
            )

        text = (result.text or "").strip()
        if not text:
            self._record_usage(
                request_id=request_id,
                model=model_name,
                status="error",
                usage=usage,
                duration_ms=latency_ms,
                started_at=started_at,
                provider_response_id=provider_response_id,
                routing_reason=f"{task.task_id}: empty response",
            )
            return self._task_payload(
                task,
                request_id=request_id,
                status="error",
                result=None,
                error="The model returned an empty response.",
                latency_ms=latency_ms,
                usage=usage,
                cost=self._estimate_cost(model_name, usage),
            )

        cost = self._record_usage(
            request_id=request_id,
            model=model_name,
            status="complete",
            usage=usage,
            duration_ms=latency_ms,
            started_at=started_at,
            provider_response_id=provider_response_id,
            routing_reason=f"parallel experiment · {task.task_id}",
        )
        return self._task_payload(
            task,
            request_id=request_id,
            status="complete",
            result=text,
            error=None,
            latency_ms=latency_ms,
            usage=usage,
            cost=cost,
        )

    def _task_payload(
        self,
        task: ExperimentTask,
        *,
        request_id: str,
        status: TaskStatus,
        result: str | None,
        error: str | None,
        latency_ms: int,
        usage: LLMUsage | None,
        cost: float | None,
    ) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "task_name": task.task_name,
            "status": status,
            "result": result,
            "error": error,
            "request_id": request_id,
            "latency_ms": latency_ms,
            "input_tokens": usage.input_tokens if usage else None,
            "output_tokens": usage.output_tokens if usage else None,
            "estimated_cost_usd": cost,
        }

    def _aggregate(self, tasks: list[dict[str, Any]], total_latency_ms: int) -> dict[str, Any]:
        input_tokens = 0
        output_tokens = 0
        cost_total = 0.0
        cost_any = False
        successful = 0
        failed = 0
        for task in tasks:
            if task["status"] == "complete":
                successful += 1
            else:
                failed += 1
            if task["input_tokens"] is not None:
                input_tokens += task["input_tokens"]
            if task["output_tokens"] is not None:
                output_tokens += task["output_tokens"]
            if task["estimated_cost_usd"] is not None:
                cost_total += task["estimated_cost_usd"]
                cost_any = True
        return {
            "latency_ms": total_latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost_usd": cost_total if cost_any else None,
            "successful_tasks": successful,
            "failed_tasks": failed,
        }

    def _record_usage(
        self,
        *,
        request_id: str,
        model: str | None,
        status: str,
        usage: LLMUsage | None,
        duration_ms: int,
        started_at: str,
        provider_response_id: str | None,
        routing_reason: str | None,
    ) -> float | None:
        try:
            self.repo.record_usage(
                request_id=request_id,
                conversation_id=None,
                message_id=None,
                provider=self.settings.llm_provider,
                requested_model=self.settings.openai_model,
                model=model,
                status=status,
                input_tokens=usage.input_tokens if usage else None,
                output_tokens=usage.output_tokens if usage else None,
                cached_input_tokens=usage.cached_input_tokens if usage else None,
                cache_write_tokens=usage.cache_write_tokens if usage else None,
                reasoning_tokens=usage.reasoning_tokens if usage else None,
                duration_ms=duration_ms,
                started_at=started_at,
                finished_at=utc_now(),
                provider_response_id=provider_response_id,
                selection=None,
                task_mode=None,
                routing_mode=None,
                routing_reason=routing_reason,
                first_token_ms=None,
                purpose=PURPOSE,
            )
        except Exception:
            logger.exception("parallel_experiment_usage_persist_failed request_id=%s", request_id)

        return self._estimate_cost(model, usage)

    def _estimate_cost(self, model: str | None, usage: LLMUsage | None) -> float | None:
        if self.pricing is None or usage is None or not usage.reported:
            return None
        model_pricing = self.pricing.find(model)
        if model_pricing is None:
            return None
        estimate = estimate_cost(
            model_pricing,
            input_tokens=usage.input_tokens or 0,
            output_tokens=usage.output_tokens or 0,
            cached_input_tokens=usage.cached_input_tokens,
            cache_write_tokens=usage.cache_write_tokens,
        )
        return float(estimate.total)
