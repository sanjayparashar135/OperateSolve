import type { Metadata } from "next";
import { KnowledgeSearchPanel } from "@/components/knowledge/knowledge-search";

export const metadata: Metadata = {
  title: "Knowledge · OperateSolve",
  description: "Search and grounded answers over the OperateSolve enterprise knowledge base.",
};

export default function KnowledgePage() {
  return <KnowledgeSearchPanel />;
}
