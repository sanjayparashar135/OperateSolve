import type { Metadata } from "next";
import { UsageDashboard } from "@/components/usage/usage-dashboard";

export const metadata: Metadata = {
  title: "Usage & cost · OperateSolve",
  description: "OpenAI API token usage and estimated cost for this OperateSolve backend.",
};

export default function UsagePage() {
  return <UsageDashboard />;
}
