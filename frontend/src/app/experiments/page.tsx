import type { Metadata } from "next";
import { ParallelExperimentPanel } from "@/components/experiments/parallel-experiment";

export const metadata: Metadata = {
  title: "LLM Experiments · OperateSolve",
  description: "Sequential vs parallel LLM execution experiment for OperateSolve.",
};

export default function ExperimentsPage() {
  return <ParallelExperimentPanel />;
}
