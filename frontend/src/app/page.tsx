import { Suspense } from "react";
import { ChatApp } from "@/components/chat/chat-app";

export default function Home() {
  return (
    <Suspense fallback={<div className="boot">Loading OperateSolve…</div>}>
      <ChatApp />
    </Suspense>
  );
}
