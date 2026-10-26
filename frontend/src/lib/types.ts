export type Role = "user" | "assistant";

export type MessageStatus = "complete" | "streaming" | "error" | "stopped";

export type TaskMode = "simple" | "general" | "detailed";

export type RoutingMode = "manual" | "auto" | "fallback";

/** "auto" or a registry key such as "luna". */
export type ModelSelection = string;

export type ModelPreference = {
  model_selection: ModelSelection;
  task_mode: TaskMode;
};

export type Conversation = {
  id: string;
  title: string;
  title_customized: boolean;
  model_selection: ModelSelection;
  task_mode: TaskMode;
  created_at: string;
  updated_at: string;
};

export type Message = {
  id: string;
  conversation_id: string;
  role: Role;
  content: string;
  status: MessageStatus;
  model: string | null;
  requested_model?: string | null;
  routing_mode?: RoutingMode | null;
  routing_reason?: string | null;
  latency_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
  cached_input_tokens?: number | null;
  request_id: string | null;
  error: string | null;
  created_at: string;
  updated_at: string;
};

export type ConversationDetail = {
  conversation: Conversation;
  messages: Message[];
};

export type Health = {
  status: "ok" | "degraded";
  database: "ok" | "error";
  llm_configured: boolean;
  model: string;
  provider: string;
};

export type MetaEvent = {
  type: "meta";
  request_id: string;
  conversation_id: string;
  user_message_id: string;
  assistant_message_id: string;
  title: string;
  replay: boolean;
  routing: RoutingInfo | null;
};

export type RoutingInfo = {
  selection: ModelSelection;
  task_mode: TaskMode | null;
  routing_mode: RoutingMode;
  requested_model: string;
  model: string;
  reason: string;
};

export type ModelInfo = {
  key: string;
  model_id: string;
  provider: string;
  display_name: string;
  description: string;
  available: boolean;
  unavailable_reason: string | null;
  baseline: boolean;
  deprecated: boolean;
  shutdown_date: string | null;
  replacement: string | null;
  capabilities: string[];
  reasoning_efforts: string[];
  reasoning_effort: string | null;
  context_window: number;
  max_output_tokens: number;
  priced: boolean;
  rates: { input: number; cached_input: number | null; cache_write: number | null; output: number } | null;
  docs_url: string | null;
};

export type AutoRule = {
  task_mode: TaskMode;
  label: string;
  target: string;
  resolved_model: string | null;
  routing_mode: RoutingMode | null;
  reason: string;
};

export type ModelCatalog = {
  default_selection: ModelSelection;
  default_task_mode: TaskMode;
  baseline_model: string;
  access_checked_at: string | null;
  models: ModelInfo[];
  auto_rules: AutoRule[];
};

export type UsageSummary = {
  requests: number;
  requests_with_usage: number;
  requests_missing_usage: number;
  requests_unpriced: number;
  requests_cached_unknown: number;
  requests_backfilled: number;
  input_tokens: number;
  cached_input_tokens: number;
  cache_write_tokens: number;
  output_tokens: number;
  reasoning_tokens: number;
  /** cached_input_tokens / input_tokens over requests that reported a cached count (0–1). */
  cache_hit_rate: number | null;
  estimated_cost_usd: number | null;
  average_cost_usd: number | null;
  estimated_input_cost_usd: number | null;
  estimated_uncached_input_cost_usd: number | null;
  estimated_cache_read_savings_usd: number | null;
  estimated_cache_write_overhead_usd: number | null;
  /** Read savings minus write overhead. Negative when writes cost more than reads saved. */
  estimated_cache_savings_usd: number | null;
  average_latency_ms: number | null;
  average_first_token_ms: number | null;
  first_request_at: string | null;
  last_request_at: string | null;
};

export type ModelRates = {
  input: number;
  cached_input: number | null;
  cache_write: number | null;
  output: number;
  long_context_above_input_tokens: number | null;
};

export type ModelUsage = UsageSummary & { model: string; display_name: string | null; rates: ModelRates | null };

export type RouteUsage = UsageSummary & {
  routing_mode: RoutingMode | "legacy";
  task_mode: TaskMode | null;
  requested_model: string;
  model: string;
  display_name: string | null;
};

export type ConversationUsage = UsageSummary & {
  conversation_id: string | null;
  title: string | null;
  deleted: boolean;
};

export type DailyUsage = {
  date: string;
  requests: number;
  input_tokens: number;
  output_tokens: number;
  cached_input_tokens: number;
  estimated_cost_usd: number | null;
};

export type RecentRequest = {
  request_id: string;
  started_at: string;
  conversation_id: string | null;
  conversation_title: string | null;
  model: string | null;
  requested_model: string | null;
  selection: string | null;
  task_mode: TaskMode | null;
  routing_mode: RoutingMode | null;
  routing_reason: string | null;
  first_token_ms: number | null;
  purpose?: "chat" | "summarization" | "incident_analysis" | "parallel_experiment" | string;
  status: string;
  source: "live" | "backfill" | string;
  usage_reported: boolean;
  priced: boolean;
  input_tokens: number | null;
  cached_input_tokens: number | null;
  cache_write_tokens: number | null;
  output_tokens: number | null;
  reasoning_tokens: number | null;
  duration_ms: number | null;
  estimated_cost_usd: number | null;
  estimated_cache_savings_usd: number | null;
};

export type UsageReport = {
  generated_at: string;
  days: number | null;
  window_start: string | null;
  pricing: {
    currency: string;
    tier: string;
    source_url: string | null;
    verified_at: string | null;
    notes: string | null;
    error: string | null;
    models: string[];
  };
  totals: UsageSummary;
  by_model: ModelUsage[];
  by_route: RouteUsage[];
  by_conversation: ConversationUsage[];
  daily: DailyUsage[];
  recent: RecentRequest[];
};

export type StreamHandlers = {
  onMeta: (event: MetaEvent) => void;
  onDelta: (text: string) => void;
  onDone: (message: Message) => void;
  onError: (event: { error: string; message: Message }) => void;
};

export type IncidentSeverity = "P1" | "P2" | "P3" | "P4";

export type IncidentAnalysis = {
  incident_id: string;
  severity: IncidentSeverity;
  summary: string;
  probable_cause: string;
  recommended_actions: string[];
  requires_human_approval: boolean;
};

export type IncidentAnalyzeResponse = {
  analysis: IncidentAnalysis;
  request_id: string;
  model: string | null;
  latency_ms: number;
  input_tokens: number | null;
  output_tokens: number | null;
  cached_input_tokens: number | null;
  cache_write_tokens: number | null;
  reasoning_tokens: number | null;
  estimated_cost_usd: number | null;
};

export type ExperimentMode = "sequential" | "parallel";

export type ExperimentTaskStatus = "complete" | "error" | "timeout";

export type ParallelTaskResult = {
  task_id: string;
  task_name: string;
  status: ExperimentTaskStatus;
  result: string | null;
  error: string | null;
  request_id: string;
  latency_ms: number;
  input_tokens: number | null;
  output_tokens: number | null;
  estimated_cost_usd: number | null;
};

export type ParallelExperimentTotals = {
  latency_ms: number;
  input_tokens: number;
  output_tokens: number;
  estimated_cost_usd: number | null;
  successful_tasks: number;
  failed_tasks: number;
};

export type ParallelExperimentResponse = {
  mode: ExperimentMode;
  model: string;
  tasks: ParallelTaskResult[];
  totals: ParallelExperimentTotals;
};

export type RetrievalMode = "semantic" | "hybrid";

export type KnowledgeSearchHit = {
  chunk_id: string;
  document_id: string;
  document_title: string;
  source: string;
  document_type: string;
  section_heading: string;
  content: string;
  distance: number;
  similarity: number;
  rank: number;
  bm25_score?: number | null;
  rrf_score?: number | null;
  semantic_rank?: number | null;
  bm25_rank?: number | null;
  retrieval_method?: string;
};

export type KnowledgeSearchResponse = {
  query: string;
  top_k: number;
  embedding_model: string;
  hit_count: number;
  hits: KnowledgeSearchHit[];
  retrieval_mode?: string;
  reranker?: string;
  semantic_candidates?: number;
  bm25_candidates?: number;
  retrieval_latency_ms?: number | null;
};

export type KnowledgeCitation = {
  ref: string;
  chunk_id: string;
  document_id: string;
  document_title: string;
  source: string;
  document_type: string;
  section_heading: string;
  content: string;
  similarity: number;
  rank: number;
  bm25_score?: number | null;
  rrf_score?: number | null;
  retrieval_method?: string;
};

export type KnowledgeAskResponse = {
  question: string;
  answer: string;
  citations: KnowledgeCitation[];
  insufficient_context: boolean;
  retrieved_count: number;
  selected_count: number;
  top_k: number;
  model: string | null;
  embedding_model: string;
  request_id: string;
  retrieval_latency_ms: number;
  generation_latency_ms: number | null;
  latency_ms: number;
  input_tokens: number | null;
  output_tokens: number | null;
  cached_input_tokens: number | null;
  cache_write_tokens: number | null;
  reasoning_tokens: number | null;
  estimated_cost_usd: number | null;
  embedding_tokens: number | null;
  retrieval_mode?: string;
  reranker?: string;
  citation_note: string;
};
