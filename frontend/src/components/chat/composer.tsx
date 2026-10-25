"use client";

import { useEffect, useRef, type ReactNode } from "react";
import { ArrowUpIcon, SquareIcon } from "lucide-react";
import { Button } from "@/components/ui/button";

export function Composer({
  value,
  streaming,
  disabled,
  toolbar,
  onChange,
  onSubmit,
  onStop,
}: {
  value: string;
  streaming: boolean;
  disabled: boolean;
  toolbar?: ReactNode;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
}) {
  const fieldRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const field = fieldRef.current;
    if (!field) return;
    field.style.height = "0px";
    field.style.height = `${Math.min(field.scrollHeight, 220)}px`;
  }, [value]);

  return (
    <form
      className="composer"
      onSubmit={(event) => {
        event.preventDefault();
        if (!streaming) onSubmit();
      }}
    >
      <label className="sr-only" htmlFor="composer">
        Message OperateSolve
      </label>
      <textarea
        id="composer"
        ref={fieldRef}
        rows={1}
        value={value}
        placeholder="Ask about an incident, a change, or a runbook"
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        onKeyDown={(event) => {
          if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            if (!streaming) onSubmit();
          }
        }}
      />
      <div className="composer-bar">
        <div className="composer-tools">{toolbar}</div>
        {streaming ? (
          <Button type="button" size="icon-lg" variant="secondary" aria-label="Stop generating" onClick={onStop}>
            <SquareIcon />
          </Button>
        ) : (
          <Button type="submit" size="icon-lg" aria-label="Send message" disabled={disabled || value.trim().length === 0}>
            <ArrowUpIcon />
          </Button>
        )}
      </div>
    </form>
  );
}
