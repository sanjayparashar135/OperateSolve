export type Role = "user" | "assistant";

export type MessageStatus = "complete" | "streaming" | "error" | "stopped";

export type Conversation = {
  id: string;
  title: string;
  title_customized: boolean;
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
  latency_ms: number | null;
  input_tokens: number | null;
  output_tokens: number | null;
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
};

export type StreamHandlers = {
  onMeta: (event: MetaEvent) => void;
  onDelta: (text: string) => void;
  onDone: (message: Message) => void;
  onError: (event: { error: string; message: Message }) => void;
};
