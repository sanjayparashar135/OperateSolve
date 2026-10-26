import type { Metadata } from "next";
import { IncidentAnalysisPanel } from "@/components/incidents/incident-analysis";

export const metadata: Metadata = {
  title: "Incident analysis · OperateSolve",
  description: "Schema-enforced structured incident analysis for OperateSolve.",
};

export default function IncidentsPage() {
  return <IncidentAnalysisPanel />;
}
