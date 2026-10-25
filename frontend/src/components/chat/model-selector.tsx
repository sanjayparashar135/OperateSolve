"use client";

import { ChevronDownIcon, CpuIcon, SparklesIcon } from "lucide-react";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { formatRate } from "@/lib/format";
import type { ModelCatalog, ModelInfo, ModelPreference, TaskMode } from "@/lib/types";
import { cn } from "@/lib/utils";

const TASK_MODES: Array<{ value: TaskMode; label: string; hint: string }> = [
  { value: "simple", label: "Simple", hint: "Short lookups, rewrites, classification" },
  { value: "general", label: "General", hint: "Everyday operations questions" },
  { value: "detailed", label: "Detailed", hint: "Longer analysis and drafts" },
];

export function ModelSelector({
  catalog,
  preference,
  disabled,
  onChange,
}: {
  catalog: ModelCatalog | null;
  preference: ModelPreference;
  disabled: boolean;
  onChange: (next: ModelPreference) => void;
}) {
  const models = catalog?.models ?? [];
  const isAuto = preference.model_selection === "auto";
  const manual = models.find((model) => model.key === preference.model_selection) ?? null;
  const rule = catalog?.auto_rules.find((item) => item.task_mode === preference.task_mode) ?? null;
  const routed = isAuto ? models.find((model) => model.model_id === rule?.resolved_model) ?? null : manual;
  const manualUnavailable = !isAuto && manual !== null && !manual.available;

  return (
    <div className="model-bar">
      <DropdownMenu>
        <DropdownMenuTrigger
          disabled={disabled || !catalog}
          render={
            <button
              type="button"
              className={cn("model-trigger", manualUnavailable && "is-unavailable")}
              aria-label="Choose model"
            />
          }
        >
          {isAuto ? <SparklesIcon /> : <CpuIcon />}
          <span className="model-trigger-name">{isAuto ? "Auto" : (manual?.display_name ?? preference.model_selection)}</span>
          {isAuto && routed ? <span className="model-trigger-sub">→ {routed.display_name}</span> : null}
          {isAuto && rule?.routing_mode === "fallback" ? <span className="model-flag warn">fallback</span> : null}
          {manualUnavailable ? <span className="model-flag danger">unavailable</span> : null}
          <ChevronDownIcon className="model-trigger-caret" />
        </DropdownMenuTrigger>
        <DropdownMenuContent side="top" align="start" sideOffset={8} className="model-menu w-[340px]">
          <DropdownMenuRadioGroup
            value={preference.model_selection}
            onValueChange={(value: string) => onChange({ ...preference, model_selection: value })}
          >
            <p className="model-menu-label">Routing</p>
            <DropdownMenuRadioItem value="auto" className="model-option">
              <div className="model-option-body">
                <span className="model-option-head">
                  <SparklesIcon className="model-option-icon" />
                  Auto
                </span>
                <span className="model-option-desc">
                  {catalog?.auto_rules
                    .map((item) => `${item.label} → ${displayName(models, item.target)}`)
                    .join(" · ") ?? "Routes by task mode"}
                </span>
              </div>
            </DropdownMenuRadioItem>
            <DropdownMenuSeparator />
            <p className="model-menu-label">Always use</p>
            {models.map((model) => (
              <DropdownMenuRadioItem key={model.key} value={model.key} disabled={!model.available} className="model-option">
                <ModelOption model={model} />
              </DropdownMenuRadioItem>
            ))}
          </DropdownMenuRadioGroup>
          {catalog?.access_checked_at ? (
            <p className="model-menu-foot">Access checked against your API key at API startup.</p>
          ) : null}
        </DropdownMenuContent>
      </DropdownMenu>

      {isAuto ? (
        <div className="segmented small task-modes" role="group" aria-label="Task mode">
          {TASK_MODES.map((mode) => {
            const modeRule = catalog?.auto_rules.find((item) => item.task_mode === mode.value);
            return (
              <Tooltip key={mode.value}>
                <TooltipTrigger
                  render={
                    <button
                      type="button"
                      aria-pressed={preference.task_mode === mode.value}
                      className={cn(preference.task_mode === mode.value && "active")}
                      disabled={disabled}
                      onClick={() => onChange({ ...preference, task_mode: mode.value })}
                    />
                  }
                >
                  {mode.label}
                </TooltipTrigger>
                <TooltipContent>{modeRule ? `${mode.hint}. ${modeRule.reason}` : mode.hint}</TooltipContent>
              </Tooltip>
            );
          })}
        </div>
      ) : null}
    </div>
  );
}

function ModelOption({ model }: { model: ModelInfo }) {
  return (
    <div className="model-option-body">
      <span className="model-option-head">
        {model.display_name}
        <span className="model-option-id">{model.model_id}</span>
        {model.baseline ? <span className="model-flag">default</span> : null}
        {model.deprecated && model.available ? (
          <span className="model-flag warn" title={model.replacement ? `OpenAI suggests ${model.replacement}` : undefined}>
            deprecated{model.shutdown_date ? ` · ends ${formatShutdown(model.shutdown_date)}` : ""}
          </span>
        ) : null}
        {!model.available ? <span className="model-flag danger">unavailable</span> : null}
      </span>
      <span className="model-option-desc">
        {model.available ? model.description : model.unavailable_reason}
      </span>
      <span className="model-option-price">
        {model.rates
          ? `${formatRate(model.rates.input)} in · ${formatRate(model.rates.cached_input)} cached · ${formatRate(
              model.rates.output,
            )} out / 1M`
          : "Unpriced: tokens are counted, cost is not estimated"}
      </span>
    </div>
  );
}

function displayName(models: ModelInfo[], modelId: string): string {
  return models.find((model) => model.model_id === modelId)?.display_name ?? modelId;
}

function formatShutdown(isoDate: string): string {
  const date = new Date(`${isoDate}T00:00:00Z`);
  if (Number.isNaN(date.getTime())) return isoDate;
  return new Intl.DateTimeFormat(undefined, { month: "short", day: "numeric", year: "numeric", timeZone: "UTC" }).format(date);
}
