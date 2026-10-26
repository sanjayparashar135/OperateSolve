"use client";

import { useState, type FormEvent } from "react";
import Link from "next/link";
import {
  ArrowLeftIcon,
  ChevronDownIcon,
  LibraryIcon,
  LoaderCircleIcon,
  MessageSquareTextIcon,
  SearchIcon,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { MarkdownMessage } from "@/components/chat/markdown-message";
import { askKnowledge, isAbortError, searchKnowledge } from "@/lib/api";
import { formatCount, formatLatency, formatUsd } from "@/lib/format";
import type {
  KnowledgeAskResponse,
  KnowledgeCitation,
  KnowledgeSearchHit,
  KnowledgeSearchResponse,
  RetrievalMode,
} from "@/lib/types";
import { cn } from "@/lib/utils";

type Mode = "search" | "ask";

const SAMPLE_QUERY =
  "How do we recover the Order Management API after database connection exhaustion?";

export function KnowledgeSearchPanel() {
  const [mode, setMode] = useState<Mode>("ask");
  const [retrievalMode, setRetrievalMode] = useState<RetrievalMode>("semantic");
  const [query, setQuery] = useState(SAMPLE_QUERY);
  const [topK, setTopK] = useState(5);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [searchResult, setSearchResult] = useState<KnowledgeSearchResponse | null>(null);
  const [askResult, setAskResult] = useState<KnowledgeAskResponse | null>(null);

  async function onSubmit(event: FormEvent) {
    event.preventDefault();
    const text = query.trim();
    if (!text) {
      setError(mode === "ask" ? "Enter a question." : "Enter a search query.");
      return;
    }
    setLoading(true);
    setError(null);
    try {
      if (mode === "search") {
        const next = await searchKnowledge(text, topK, undefined, retrievalMode);
        setSearchResult(next);
        setAskResult(null);
      } else {
        const next = await askKnowledge(text, topK, undefined, retrievalMode);
        setAskResult(next);
        setSearchResult(null);
      }
    } catch (reason: unknown) {
      if (isAbortError(reason)) return;
      if (mode === "search") setSearchResult(null);
      else setAskResult(null);
      setError(
        reason instanceof Error
          ? reason.message
          : mode === "ask"
            ? "Could not generate a grounded answer."
            : "Could not search the knowledge base.",
      );
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="knowledge-page">
      <div className="knowledge-inner">
        <Link href="/" className="back-link">
          <ArrowLeftIcon />
          Back to chats
        </Link>

        <header className="knowledge-head">
          <div>
            <p className="eyebrow">Phase 3.3</p>
            <h1>Knowledge</h1>
            <p className="knowledge-lede">
              Search Northstar Commerce documents, or ask for a grounded answer with source
              citations. Choose semantic or hybrid (BM25 + RRF) retrieval.
            </p>
          </div>
        </header>

        <div className="knowledge-mode-tabs" role="tablist" aria-label="Knowledge mode">
          <button
            type="button"
            role="tab"
            aria-selected={mode === "search"}
            className={cn("knowledge-mode-tab", mode === "search" && "active")}
            onClick={() => setMode("search")}
            disabled={loading}
          >
            <SearchIcon />
            Search Documents
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={mode === "ask"}
            className={cn("knowledge-mode-tab", mode === "ask" && "active")}
            onClick={() => setMode("ask")}
            disabled={loading}
          >
            <MessageSquareTextIcon />
            Ask Knowledge
          </button>
        </div>

        <form className="knowledge-form" onSubmit={(event) => void onSubmit(event)}>
          <label className="field">
            <span>{mode === "ask" ? "Question" : "Query"}</span>
            <textarea
              className="knowledge-textarea"
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              maxLength={4000}
              rows={4}
              placeholder={
                mode === "ask"
                  ? "Ask about runbooks, APIs, incidents…"
                  : "Search runbooks, APIs, incidents…"
              }
              disabled={loading}
            />
          </label>
          <div className="knowledge-form-row">
            <label className="field field-inline">
              <span>Top K</span>
              <Input
                type="number"
                min={1}
                max={50}
                value={topK}
                onChange={(event) => setTopK(Number(event.target.value) || 5)}
                disabled={loading}
                className="knowledge-topk"
              />
            </label>
            <fieldset className="knowledge-retrieval" disabled={loading}>
              <legend>Retrieval</legend>
              <label className="knowledge-retrieval-option">
                <input
                  type="radio"
                  name="retrieval-mode"
                  value="semantic"
                  checked={retrievalMode === "semantic"}
                  onChange={() => setRetrievalMode("semantic")}
                />
                <span>Semantic</span>
              </label>
              <label className="knowledge-retrieval-option">
                <input
                  type="radio"
                  name="retrieval-mode"
                  value="hybrid"
                  checked={retrievalMode === "hybrid"}
                  onChange={() => setRetrievalMode("hybrid")}
                />
                <span>Hybrid</span>
              </label>
            </fieldset>
          </div>
          <div className="knowledge-actions">
            <Button type="submit" disabled={loading}>
              {loading ? (
                <>
                  <LoaderCircleIcon className="spin" />
                  {mode === "ask" ? "Generating…" : "Searching…"}
                </>
              ) : mode === "ask" ? (
                <>
                  <MessageSquareTextIcon />
                  Ask
                </>
              ) : (
                <>
                  <LibraryIcon />
                  Search
                </>
              )}
            </Button>
          </div>
        </form>

        {error ? <p className="knowledge-error">{error}</p> : null}

        {mode === "search" && searchResult ? <SearchResults result={searchResult} /> : null}
        {mode === "ask" && askResult ? <AskResults result={askResult} /> : null}
      </div>
    </main>
  );
}

function SearchResults({ result }: { result: KnowledgeSearchResponse }) {
  const mode = result.retrieval_mode ?? "semantic";
  return (
    <section className="knowledge-results" aria-live="polite">
      <div className="knowledge-results-head">
        <h2>
          {result.hit_count === 0
            ? "No matching chunks"
            : `${result.hit_count} matching chunk${result.hit_count === 1 ? "" : "s"}`}
        </h2>
        <p>
          Retrieval <code>{mode}</code>
          {result.reranker && result.reranker !== "noop" ? (
            <>
              {" · "}reranker <code>{result.reranker}</code>
            </>
          ) : null}
          {" · "}model <code>{result.embedding_model}</code>
          {result.retrieval_latency_ms != null ? (
            <>
              {" · "}
              {formatLatency(result.retrieval_latency_ms)}
            </>
          ) : null}
          {result.hit_count === 0
            ? " · ingest documents if the store is empty"
            : mode === "hybrid"
              ? " · scores: cosine similarity, BM25, RRF"
              : " · cosine similarity (higher is closer)"}
        </p>
      </div>
      {result.hits.length === 0 ? (
        <p className="knowledge-empty">
          The vector store returned no hits. Run knowledge ingestion, then search again.
        </p>
      ) : (
        <ul className="knowledge-hit-list">
          {result.hits.map((hit) => (
            <li key={hit.chunk_id}>
              <KnowledgeHitCard hit={hit} />
            </li>
          ))}
        </ul>
      )}
    </section>
  );
}

function AskResults({ result }: { result: KnowledgeAskResponse }) {
  return (
    <section className="knowledge-results" aria-live="polite">
      <div className="knowledge-results-head">
        <h2>{result.insufficient_context ? "Insufficient evidence" : "Grounded answer"}</h2>
        <p className="knowledge-metrics">
          Retrieval <code>{result.retrieval_mode ?? "semantic"}</code>
          {result.reranker && result.reranker !== "noop" ? (
            <>
              {" · "}reranker <code>{result.reranker}</code>
            </>
          ) : null}
          {result.model ? (
            <>
              {" · "}model <code>{result.model}</code>
            </>
          ) : null}
          {" · "}
          {formatLatency(result.latency_ms)}
          {result.input_tokens != null || result.output_tokens != null ? (
            <>
              {" · "}
              {formatCount(result.input_tokens)} in / {formatCount(result.output_tokens)} out
            </>
          ) : null}
          {result.estimated_cost_usd != null ? (
            <>
              {" · "}
              {formatUsd(result.estimated_cost_usd)}
            </>
          ) : null}
          {" · "}
          retrieved {result.retrieved_count}, selected {result.selected_count}
        </p>
      </div>

      {result.insufficient_context ? (
        <p className="knowledge-insufficient" role="status">
          Supporting information in the knowledge base appears insufficient for a confident answer.
          Treat the text below as an abstention, not a verified fact.
        </p>
      ) : null}

      <div className="knowledge-answer">
        <MarkdownMessage content={result.answer} />
      </div>

      {result.citations.length > 0 ? (
        <div className="knowledge-citations">
          <h3>Sources</h3>
          <p className="knowledge-citation-note">{result.citation_note}</p>
          <ol className="knowledge-citation-list">
            {result.citations.map((citation) => (
              <li key={`${citation.ref}-${citation.chunk_id}`}>
                <CitationCard citation={citation} />
              </li>
            ))}
          </ol>
        </div>
      ) : (
        <p className="knowledge-empty">No validated source citations for this answer.</p>
      )}
    </section>
  );
}

function CitationCard({ citation }: { citation: KnowledgeCitation }) {
  const [open, setOpen] = useState(false);
  return (
    <article className="knowledge-citation">
      <button
        type="button"
        className="knowledge-citation-toggle"
        aria-expanded={open}
        onClick={() => setOpen((value) => !value)}
      >
        <span className="knowledge-rank">[{citation.ref}]</span>
        <div className="knowledge-citation-summary">
          <h4>{citation.document_title || citation.document_id}</h4>
          <p className="knowledge-meta">
            {citation.section_heading ? <span>{citation.section_heading}</span> : null}
            {citation.section_heading && citation.source ? (
              <span aria-hidden="true"> · </span>
            ) : null}
            {citation.source ? <span>{citation.source}</span> : null}
          </p>
        </div>
        <ChevronDownIcon className={cn("knowledge-chevron", open && "open")} />
      </button>
      {open ? (
        <div className="knowledge-citation-body">
          <pre className="knowledge-chunk">{citation.content}</pre>
          <p className="knowledge-ids">
            <span>doc: {citation.document_id}</span>
            <span>chunk: {citation.chunk_id}</span>
            <span>similarity: {citation.similarity.toFixed(4)}</span>
            {citation.bm25_score != null ? (
              <span>bm25: {citation.bm25_score.toFixed(4)}</span>
            ) : null}
            {citation.rrf_score != null ? (
              <span>rrf: {citation.rrf_score.toFixed(4)}</span>
            ) : null}
            {citation.retrieval_method ? <span>{citation.retrieval_method}</span> : null}
          </p>
        </div>
      ) : null}
    </article>
  );
}

function KnowledgeHitCard({ hit }: { hit: KnowledgeSearchHit }) {
  const method = hit.retrieval_method ?? "semantic";
  return (
    <article className="knowledge-hit">
      <div className="knowledge-hit-head">
        <span className="knowledge-rank">#{hit.rank}</span>
        <div>
          <h3>{hit.document_title || hit.document_id}</h3>
          <p className="knowledge-meta">
            {hit.section_heading ? <span>{hit.section_heading}</span> : null}
            {hit.section_heading && hit.source ? <span aria-hidden="true"> · </span> : null}
            {hit.source ? <span>{hit.source}</span> : null}
            {method ? (
              <>
                <span aria-hidden="true"> · </span>
                <span>{method}</span>
              </>
            ) : null}
          </p>
        </div>
        <div
          className="knowledge-scores"
          title="Cosine similarity from Chroma; BM25 keyword score; RRF fusion score"
        >
          {hit.semantic_rank != null || method === "semantic" || method === "hybrid" ? (
            <span>
              similarity <strong>{hit.similarity.toFixed(4)}</strong>
            </span>
          ) : null}
          {hit.bm25_score != null ? (
            <span>
              bm25 <strong>{hit.bm25_score.toFixed(4)}</strong>
            </span>
          ) : null}
          {hit.rrf_score != null ? (
            <span>
              rrf <strong>{hit.rrf_score.toFixed(4)}</strong>
            </span>
          ) : null}
          {hit.bm25_score == null && hit.rrf_score == null ? (
            <span>
              distance <strong>{hit.distance.toFixed(4)}</strong>
            </span>
          ) : null}
        </div>
      </div>
      <pre className="knowledge-chunk">{hit.content}</pre>
      <p className="knowledge-ids">
        <span>doc: {hit.document_id}</span>
        <span>chunk: {hit.chunk_id}</span>
        {hit.semantic_rank != null ? <span>sem-rank: {hit.semantic_rank}</span> : null}
        {hit.bm25_rank != null ? <span>bm25-rank: {hit.bm25_rank}</span> : null}
      </p>
    </article>
  );
}
