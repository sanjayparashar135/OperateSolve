"use client";

import { useState, type ReactNode } from "react";
import { CheckIcon, CopyIcon } from "lucide-react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";

const remarkPlugins = [remarkGfm];

const components: Components = {
  a({ href, children }) {
    const safe = href && /^https?:\/\//i.test(href) ? href : undefined;
    if (!safe) return <span>{children}</span>;
    return (
      <a href={safe} target="_blank" rel="noreferrer">
        {children}
      </a>
    );
  },
  pre({ children }) {
    return <CodeBlock>{children}</CodeBlock>;
  },
};

export function MarkdownMessage({ content }: { content: string }) {
  return (
    <div className="markdown">
      <ReactMarkdown remarkPlugins={remarkPlugins} components={components}>
        {content}
      </ReactMarkdown>
    </div>
  );
}

function CodeBlock({ children }: { children?: ReactNode }) {
  const [copied, setCopied] = useState(false);
  const text = nodeText(children).replace(/\n$/, "");

  async function copy() {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1400);
    } catch {
      setCopied(false);
    }
  }

  return (
    <div className="code-frame">
      <button type="button" className="code-copy" onClick={copy}>
        {copied ? <CheckIcon /> : <CopyIcon />}
        {copied ? "Copied" : "Copy"}
      </button>
      <pre>{children}</pre>
    </div>
  );
}

function nodeText(node: ReactNode): string {
  if (node == null || typeof node === "boolean") return "";
  if (typeof node === "string" || typeof node === "number") return String(node);
  if (Array.isArray(node)) return node.map((item) => nodeText(item)).join("");
  if (typeof node === "object" && "props" in node) {
    const element = node as { props?: { children?: ReactNode } };
    return nodeText(element.props?.children);
  }
  return "";
}
