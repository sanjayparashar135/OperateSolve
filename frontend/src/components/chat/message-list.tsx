"use client";

import { useState } from "react";
import { CheckIcon, CopyIcon, RefreshCwIcon } from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Tooltip, TooltipContent, TooltipTrigger } from "@/components/ui/tooltip";
import { MarkdownMessage } from "@/components/chat/markdown-message";
import { Metrics } from "@/components/chat/metrics";
import type { Message } from "@/lib/types";

export function MessageList({
  messages,
  streaming,
  onRetry,
}: {
  messages: Message[];
  streaming: boolean;
  onRetry: (message: Message) => void;
}) {
  const latestAssistantId = [...messages].reverse().find((message) => message.role === "assistant")?.id;

  return (
    <div className="transcript-inner">
      {messages.map((message) =>
        message.role === "user" ? (
          <article key={message.id} className="message user message-in">
            <p>{message.content}</p>
          </article>
        ) : (
          <article key={message.id} className="message assistant message-in">
            <header>
              <span className="assistant-kicker">OperateSolve</span>
              {message.status === "stopped" ? <span className="pill">Stopped</span> : null}
              {message.status === "error" ? <span className="pill danger">Failed</span> : null}
            </header>
            {message.status === "streaming" && message.content.length === 0 ? (
              <div className="thinking" aria-label="Generating response">
                <span />
                <span />
                <span />
              </div>
            ) : (
              <div className={message.status === "streaming" ? "with-caret" : undefined}>
                <MarkdownMessage content={message.content} />
              </div>
            )}
            {message.error && message.status === "error" ? <p className="message-error">{message.error}</p> : null}
            <footer>
              <Metrics message={message} />
              <div className="message-actions">
                {message.content ? (
                  <CopyButton content={message.content} />
                ) : null}
                {message.id === latestAssistantId && message.status !== "streaming" ? (
                  <Tooltip>
                    <TooltipTrigger
                      render={<Button type="button" variant="ghost" size="icon-sm" disabled={streaming} />}
                      aria-label={message.status === "complete" ? "Regenerate response" : "Retry response"}
                      onClick={() => onRetry(message)}
                    >
                      <RefreshCwIcon />
                    </TooltipTrigger>
                    <TooltipContent>{message.status === "complete" ? "Regenerate" : "Retry"}</TooltipContent>
                  </Tooltip>
                ) : null}
              </div>
            </footer>
          </article>
        ),
      )}
    </div>
  );
}

function CopyButton({ content }: { content: string }) {
  const [copied, setCopied] = useState(false);

  async function copy() {
    try {
      await navigator.clipboard.writeText(content);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      toast.error("Could not copy the response.");
    }
  }

  return (
    <Tooltip>
      <TooltipTrigger
        render={<Button type="button" variant="ghost" size="icon-sm" />}
        aria-label="Copy response"
        onClick={() => void copy()}
      >
        {copied ? <CheckIcon /> : <CopyIcon />}
      </TooltipTrigger>
      <TooltipContent>{copied ? "Copied" : "Copy"}</TooltipContent>
    </Tooltip>
  );
}
