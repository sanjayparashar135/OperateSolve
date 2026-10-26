"use client";

import { useRef, useState } from "react";
import Link from "next/link";
import {
  ChartColumnIcon,
  EllipsisIcon,
  FlaskConicalIcon,
  LibraryIcon,
  PlusIcon,
  ShieldAlertIcon,
} from "lucide-react";
import { toast } from "sonner";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Skeleton } from "@/components/ui/skeleton";
import { BrandMark } from "@/components/chat/brand-mark";
import { deleteChat, renameChat } from "@/lib/api";
import { formatWhen, groupConversations } from "@/lib/format";
import type { Conversation, Health } from "@/lib/types";
import { cn } from "@/lib/utils";

export function Sidebar({
  open,
  conversations,
  activeId,
  loading,
  health,
  apiOnline,
  onClose,
  onNew,
  onSelect,
  onRenamed,
  onDeleted,
}: {
  open: boolean;
  conversations: Conversation[];
  activeId: string | null;
  loading: boolean;
  health: Health | null;
  apiOnline: boolean;
  onClose: () => void;
  onNew: () => void;
  onSelect: (id: string) => void;
  onRenamed: (conversation: Conversation) => void;
  onDeleted: (id: string) => void;
}) {
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const [pendingDelete, setPendingDelete] = useState<Conversation | null>(null);
  const [busy, setBusy] = useState(false);
  const cancelRename = useRef(false);
  const groups = groupConversations(conversations);

  async function commitRename(id: string) {
    const title = renameValue.trim();
    if (!title) {
      setRenamingId(null);
      return;
    }
    setBusy(true);
    try {
      const updated = await renameChat(id, title);
      onRenamed(updated);
      setRenamingId(null);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not rename the conversation.");
    } finally {
      setBusy(false);
    }
  }

  async function confirmDelete() {
    if (!pendingDelete) return;
    setBusy(true);
    try {
      await deleteChat(pendingDelete.id);
      onDeleted(pendingDelete.id);
      setPendingDelete(null);
      toast.success("Conversation deleted");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Could not delete the conversation.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <aside className={cn("sidebar", open && "open")} aria-label="Conversations">
      <div className="sidebar-head">
        <div className="brand">
          <BrandMark className="brand-mark" />
          <div>
            <p className="brand-name">OperateSolve</p>
            <p className="brand-sub">Enterprise operations</p>
          </div>
        </div>
        <Button type="button" variant="outline" className="new-chat" onClick={onNew}>
          <PlusIcon />
          New chat
        </Button>
        <Link href="/incidents" className="nav-link">
          <ShieldAlertIcon />
          Incident analysis
        </Link>
        <Link href="/knowledge" className="nav-link">
          <LibraryIcon />
          Knowledge search
        </Link>
        <Link href="/experiments" className="nav-link">
          <FlaskConicalIcon />
          LLM Experiments
        </Link>
        <Link href="/usage" className="nav-link">
          <ChartColumnIcon />
          Usage &amp; cost
        </Link>
      </div>

      <ScrollArea className="sidebar-scroll">
        <div className="sidebar-list">
          {loading ? (
            <div className="stack">
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
              <Skeleton className="h-12 w-full" />
            </div>
          ) : groups.length === 0 ? (
            <p className="empty-history">Conversations you start will stay here after a refresh.</p>
          ) : (
            groups.map((group) => (
              <section key={group.label} className="conv-group">
                <h2>{group.label}</h2>
                {group.items.map((conversation) => (
                  <div
                    key={conversation.id}
                    className={cn("conv", activeId === conversation.id && "active")}
                  >
                    {renamingId === conversation.id ? (
                      <form
                        className="rename-form"
                        onSubmit={(event) => {
                          event.preventDefault();
                          void commitRename(conversation.id);
                        }}
                      >
                        <Input
                          value={renameValue}
                          autoFocus
                          maxLength={120}
                          aria-label="Conversation title"
                          disabled={busy}
                          onChange={(event) => setRenameValue(event.target.value)}
                          onBlur={() => {
                            if (cancelRename.current) {
                              cancelRename.current = false;
                              return;
                            }
                            void commitRename(conversation.id);
                          }}
                          onKeyDown={(event) => {
                            if (event.key === "Escape") {
                              event.preventDefault();
                              cancelRename.current = true;
                              setRenamingId(null);
                            }
                          }}
                        />
                      </form>
                    ) : (
                      <button type="button" className="conv-main" onClick={() => onSelect(conversation.id)}>
                        <span className="conv-title">{conversation.title}</span>
                        <span className="conv-time">{formatWhen(conversation.updated_at)}</span>
                      </button>
                    )}
                    <DropdownMenu>
                      <DropdownMenuTrigger
                        render={<Button type="button" variant="ghost" size="icon-sm" aria-label="Conversation actions" />}
                      >
                        <EllipsisIcon />
                      </DropdownMenuTrigger>
                      <DropdownMenuContent align="end">
                        <DropdownMenuItem
                          onClick={() => {
                            setRenamingId(conversation.id);
                            setRenameValue(conversation.title);
                          }}
                        >
                          Rename
                        </DropdownMenuItem>
                        <DropdownMenuItem variant="destructive" onClick={() => setPendingDelete(conversation)}>
                          Delete
                        </DropdownMenuItem>
                      </DropdownMenuContent>
                    </DropdownMenu>
                  </div>
                ))}
              </section>
            ))
          )}
        </div>
      </ScrollArea>

      <footer className="sidebar-foot">
        <span className={cn("status-dot", apiOnline ? "ok" : "down")} />
        <div>
          <p>{health?.model ?? "Connecting"}</p>
          <p>
            {!apiOnline
              ? "API unreachable"
              : health?.llm_configured
                ? "Ready for live responses"
                : "API key not configured"}
          </p>
        </div>
        <button type="button" className="mobile-close" onClick={onClose}>
          Close
        </button>
      </footer>

      <Dialog open={pendingDelete !== null} onOpenChange={(next) => !next && setPendingDelete(null)}>
        <DialogContent>
          <DialogHeader>
            <DialogTitle>Delete this conversation?</DialogTitle>
            <DialogDescription>
              {pendingDelete ? `"${pendingDelete.title}" and its messages will be removed from this device.` : ""}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => setPendingDelete(null)}>
              Cancel
            </Button>
            <Button type="button" variant="destructive" disabled={busy} onClick={() => void confirmDelete()}>
              Delete
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </aside>
  );
}
