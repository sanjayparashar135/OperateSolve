"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { PanelLeftIcon } from "lucide-react";
import { useRouter, useSearchParams } from "next/navigation";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import { Skeleton } from "@/components/ui/skeleton";
import { Composer } from "@/components/chat/composer";
import { MessageList } from "@/components/chat/message-list";
import { Sidebar } from "@/components/chat/sidebar";
import { Welcome } from "@/components/chat/welcome";
import {
  ApiError,
  API_BASE,
  createChat,
  getChat,
  getHealth,
  isAbortError,
  listChats,
  retryMessage,
  streamMessage,
} from "@/lib/api";
import { upsertConversation } from "@/lib/format";
import type { Conversation, Health, Message, MetaEvent } from "@/lib/types";

export function ChatApp() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const requestedId = searchParams.get("c");
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [messages, setMessages] = useState<Message[]>([]);
  const [draft, setDraft] = useState("");
  const [loadingList, setLoadingList] = useState(true);
  const [loadingMessages, setLoadingMessages] = useState(Boolean(requestedId));
  const [seenId, setSeenId] = useState(requestedId);
  const [streamingId, setStreamingId] = useState<string | null>(null);
  const [streaming, setStreaming] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [health, setHealth] = useState<Health | null>(null);
  const [apiOnline, setApiOnline] = useState(true);

  const sendingRef = useRef(false);
  const abortRef = useRef<AbortController | null>(null);
  const streamConversationRef = useRef<string | null>(null);
  const visibleRef = useRef<string | null>(null);
  const scrollerRef = useRef<HTMLDivElement>(null);
  const stickRef = useRef(true);

  if (seenId !== requestedId) {
    setSeenId(requestedId);
    if (streamingId !== requestedId) {
      setMessages([]);
      setLoadingMessages(requestedId !== null);
    }
  }

  const replaceUrl = useCallback(
    (id: string | null) => {
      router.replace(id ? `/?c=${id}` : "/", { scroll: false });
    },
    [router],
  );

  const stop = useCallback(() => {
    abortRef.current?.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    listChats(controller.signal)
      .then((items) => {
        setConversations(items);
        setApiOnline(true);
      })
      .catch((error: unknown) => {
        if (isAbortError(error)) return;
        setApiOnline(false);
        toast.error(error instanceof Error ? error.message : "Could not load conversations.");
      })
      .finally(() => setLoadingList(false));
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    const load = () => {
      getHealth(controller.signal)
        .then((next) => {
          setHealth(next);
          setApiOnline(true);
        })
        .catch((error: unknown) => {
          if (isAbortError(error)) return;
          setApiOnline(false);
        });
    };
    load();
    const timer = window.setInterval(load, 30000);
    return () => {
      controller.abort();
      window.clearInterval(timer);
    };
  }, []);

  useEffect(() => {
    visibleRef.current = requestedId;
  }, [requestedId]);

  useEffect(() => {
    if (!requestedId || streamingId === requestedId) return;
    const controller = new AbortController();
    const target = requestedId;
    getChat(target, controller.signal)
      .then((detail) => {
        if (visibleRef.current !== target) return;
        setMessages(detail.messages);
        setConversations((current) => upsertConversation(current, detail.conversation));
        setApiOnline(true);
      })
      .catch((error: unknown) => {
        if (isAbortError(error) || visibleRef.current !== target) return;
        setMessages([]);
        toast.error(error instanceof Error ? error.message : "Could not open that conversation.");
      })
      .finally(() => {
        if (visibleRef.current === target) setLoadingMessages(false);
      });
    return () => controller.abort();
  }, [requestedId, streamingId]);

  useEffect(() => {
    const scroller = scrollerRef.current;
    if (!scroller || !stickRef.current) return;
    scroller.scrollTop = scroller.scrollHeight;
  }, [messages]);

  useEffect(() => {
    function onKey(event: KeyboardEvent) {
      if (event.key === "Escape" && sendingRef.current) {
        stop();
      }
      if ((event.metaKey || event.ctrlKey) && event.shiftKey && event.key.toLowerCase() === "o") {
        event.preventDefault();
        stop();
        setSidebarOpen(false);
        replaceUrl(null);
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [replaceUrl, stop]);

  function selectConversation(id: string | null) {
    if (sendingRef.current && streamConversationRef.current !== id) stop();
    visibleRef.current = id;
    setSidebarOpen(false);
    stickRef.current = true;
    replaceUrl(id);
  }

  async function refreshList() {
    try {
      setConversations(await listChats());
    } catch {
      // Keep the list already on screen.
    }
  }

  async function syncConversation(conversationId: string, content?: string) {
    if (visibleRef.current !== conversationId) return;
    try {
      const fresh = await getChat(conversationId);
      if (visibleRef.current !== conversationId) return;
      setMessages(fresh.messages);
      setConversations((current) => upsertConversation(current, fresh.conversation));
      if (content && !fresh.messages.some((message) => message.role === "user" && message.content === content)) {
        setDraft(content);
      }
    } catch {
      // The optimistic transcript remains visible.
    }
  }

  async function runStream(
    conversationId: string,
    tempUserId: string,
    tempAssistantId: string,
    work: (handlers: Parameters<typeof streamMessage>[3], signal: AbortSignal) => Promise<boolean>,
    content?: string,
  ) {
    let assistantId = tempAssistantId;
    const controller = new AbortController();
    abortRef.current = controller;
    const handlers = {
      onMeta: (meta: MetaEvent) => {
        assistantId = meta.assistant_message_id;
        setMessages((current) =>
          current.map((message) => {
            if (message.id === tempUserId) return { ...message, id: meta.user_message_id, conversation_id: meta.conversation_id };
            if (message.id === tempAssistantId || message.id === meta.assistant_message_id) {
              return { ...message, id: meta.assistant_message_id, conversation_id: meta.conversation_id, request_id: meta.request_id };
            }
            return message;
          }),
        );
        if (meta.title) {
          setConversations((current) =>
            current.map((conversation) =>
              conversation.id === meta.conversation_id ? { ...conversation, title: meta.title } : conversation,
            ),
          );
        }
      },
      onDelta: (text: string) => {
        setMessages((current) =>
          current.map((message) => (message.id === assistantId ? { ...message, content: message.content + text } : message)),
        );
      },
      onDone: (message: Message) => {
        setMessages((current) => current.map((item) => (item.id === message.id || item.id === assistantId ? message : item)));
      },
      onError: (event: { error: string; message: Message }) => {
        setMessages((current) =>
          current.map((item) => (item.id === event.message.id || item.id === assistantId ? event.message : item)),
        );
      },
    };

    try {
      const terminal = await work(handlers, controller.signal);
      if (!terminal && !controller.signal.aborted) {
        await syncConversation(conversationId, content);
      }
    } catch (error) {
      if (!isAbortError(error)) {
        const message = error instanceof ApiError ? error.message : error instanceof Error ? error.message : "The request failed.";
        toast.error(message);
      }
      await syncConversation(conversationId, isAbortError(error) ? undefined : content);
    } finally {
      sendingRef.current = false;
      streamConversationRef.current = null;
      abortRef.current = null;
      setStreaming(false);
      setStreamingId(null);
      void refreshList();
    }
  }

  async function send(text: string) {
    const content = text.trim();
    if (!content || sendingRef.current) return;
    if (content.length > 20000) {
      toast.error("Messages are limited to 20,000 characters.");
      return;
    }
    sendingRef.current = true;
    setStreaming(true);
    setDraft("");
    stickRef.current = true;
    let conversationId = requestedId;
    const createdNew = !conversationId;
    let optimistic = false;
    try {
      if (!conversationId) {
        const created = await createChat();
        conversationId = created.id;
        setConversations((current) => upsertConversation(current, created));
      }
      visibleRef.current = conversationId;
      streamConversationRef.current = conversationId;
      setStreamingId(conversationId);
      const clientRequestId = crypto.randomUUID();
      const tempUserId = `temp-user-${clientRequestId}`;
      const tempAssistantId = `temp-assistant-${clientRequestId}`;
      const now = new Date().toISOString();
      const base = {
        conversation_id: conversationId,
        model: null,
        latency_ms: null,
        input_tokens: null,
        output_tokens: null,
        request_id: null,
        error: null,
        created_at: now,
        updated_at: now,
      };
      const optimisticUser: Message = { ...base, id: tempUserId, role: "user", content, status: "complete" };
      const optimisticAssistant: Message = { ...base, id: tempAssistantId, role: "assistant", content: "", status: "streaming" };
      setMessages((current) => (createdNew ? [optimisticUser, optimisticAssistant] : [...current, optimisticUser, optimisticAssistant]));
      if (createdNew) replaceUrl(conversationId);
      optimistic = true;
      await runStream(
        conversationId,
        tempUserId,
        tempAssistantId,
        (handlers, signal) => streamMessage(conversationId!, content, clientRequestId, handlers, signal),
        content,
      );
    } catch (error) {
      sendingRef.current = false;
      streamConversationRef.current = null;
      setStreaming(false);
      setStreamingId(null);
      if (!optimistic) setDraft(content);
      toast.error(error instanceof Error ? error.message : "Could not start the conversation.");
    }
  }

  async function retry(message: Message) {
    if (!requestedId || sendingRef.current || message.status === "streaming") return;
    sendingRef.current = true;
    streamConversationRef.current = requestedId;
    visibleRef.current = requestedId;
    setStreamingId(requestedId);
    setStreaming(true);
    stickRef.current = true;
    setMessages((current) =>
      current.map((item) =>
        item.id === message.id
          ? { ...item, content: "", status: "streaming", error: null, latency_ms: null, input_tokens: null, output_tokens: null }
          : item,
      ),
    );
    await runStream(requestedId, "", message.id, (handlers, signal) => retryMessage(requestedId, message.id, handlers, signal));
  }

  const active = conversations.find((conversation) => conversation.id === requestedId) ?? null;
  const showWelcome = !requestedId && !loadingMessages;

  return (
    <div className="shell">
      {sidebarOpen ? <button type="button" className="backdrop" aria-label="Close conversations" onClick={() => setSidebarOpen(false)} /> : null}
      <Sidebar
        open={sidebarOpen}
        conversations={conversations}
        activeId={requestedId}
        loading={loadingList}
        health={health}
        apiOnline={apiOnline}
        onClose={() => setSidebarOpen(false)}
        onNew={() => selectConversation(null)}
        onSelect={selectConversation}
        onRenamed={(conversation) => setConversations((current) => current.map((item) => (item.id === conversation.id ? conversation : item)))}
        onDeleted={(id) => {
          setConversations((current) => current.filter((item) => item.id !== id));
          if (requestedId === id) selectConversation(null);
        }}
      />
      <section className="stage" aria-live="polite">
        <header className="topbar">
          <Button type="button" variant="ghost" size="icon" className="menu-button" aria-label="Open conversations" onClick={() => setSidebarOpen(true)}>
            <PanelLeftIcon />
          </Button>
          <div className="topbar-title">
            <h2>{active?.title ?? "New conversation"}</h2>
            <p>{health?.model ?? "OperateSolve"}</p>
          </div>
        </header>
        {!apiOnline ? (
          <div className="banner warn" role="status">
            The API at {API_BASE} is unreachable. Start the backend, then refresh this page.
          </div>
        ) : health && !health.llm_configured ? (
          <div className="banner" role="status">
            Add <code>OPENAI_API_KEY</code> to <code>backend/.env</code> and restart the API to enable live responses.
          </div>
        ) : null}
        <div
          className="transcript"
          ref={scrollerRef}
          onScroll={(event) => {
            const element = event.currentTarget;
            stickRef.current = element.scrollHeight - element.scrollTop - element.clientHeight < 96;
          }}
        >
          {loadingMessages ? (
            <div className="transcript-inner">
              <Skeleton className="ml-auto h-16 w-2/3" />
              <Skeleton className="h-28 w-full" />
            </div>
          ) : showWelcome ? (
            <Welcome disabled={streaming || !apiOnline} onPrompt={(prompt) => void send(prompt)} />
          ) : (
            <MessageList messages={messages} streaming={streaming} onRetry={(message) => void retry(message)} />
          )}
        </div>
        <div className="composer-wrap">
          <Composer
            value={draft}
            streaming={streaming}
            disabled={!apiOnline}
            onChange={setDraft}
            onSubmit={() => void send(draft)}
            onStop={stop}
          />
          <p className="composer-hint">Enter to send · Shift+Enter for a new line · Esc to stop</p>
        </div>
      </section>
    </div>
  );
}
