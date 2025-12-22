"""
Experiment 2: Context Window Size Impact

Tests how accuracy and latency change as context size grows.
"""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from src.experiments.base_experiment import BaseExperiment
from src.generators import generate_document_set
from src.models import get_model
from src.utils.config import EXP2_DOCUMENT_COUNTS, TRIALS_PER_CONDITION, GRAPH_DPI
from src.utils.helpers import Timer, count_tokens


class ContextSizeExperiment(BaseExperiment):
    """Experiment 2: Test accuracy/latency vs context size."""

    def __init__(self, model_backend: str = "mock"):
        super().__init__(
            name="exp2_context_size",
            description="Measure accuracy and latency as document count increases"
        )
        self.model = get_model(backend=model_backend)
        self.fact = "The annual revenue target is $50 million."
        self.question = "What is the annual revenue target?"
        self.expected_answer = "50 million"

    def run(self) -> Dict[str, Any]:
        """Run the context size experiment."""
        results = {count: [] for count in EXP2_DOCUMENT_COUNTS}

        for doc_count in EXP2_DOCUMENT_COUNTS:
            self.logger.info(f"Testing with {doc_count} documents...")

            for trial in range(TRIALS_PER_CONDITION):
                # Generate documents
                docs = generate_document_set(doc_count - 1, 200, "business")

                # Add document with fact in the middle
                from src.generators import generate_document
                fact_doc = generate_document(200, self.fact, "middle", "business")
                docs.insert(len(docs) // 2, fact_doc)

                # Concatenate all documents
                context = "\n\n".join(docs)
                tokens = count_tokens(context)

                # Query model
                with Timer() as timer:
                    response = self.model.query(context, self.question)

                # Evaluate
                is_correct = self.expected_answer.lower() in response.lower()

                results[doc_count].append({
                    "trial": trial + 1,
                    "correct": is_correct,
                    "latency_ms": timer.elapsed_ms,
                    "tokens_used": tokens
                })

        # Calculate summary
        summary = {}
        for count in EXP2_DOCUMENT_COUNTS:
            trials = results[count]
            correct_count = sum(1 for t in trials if t["correct"])
            summary[count] = {
                "accuracy": correct_count / len(trials) * 100,
                "avg_latency_ms": sum(t["latency_ms"] for t in trials) / len(trials),
                "avg_tokens": sum(t["tokens_used"] for t in trials) / len(trials)
            }

        return {"trials": results, "summary": summary}

    def generate_graphs(self, results: Dict[str, Any]) -> List[Path]:
        """Generate accuracy and latency vs size charts."""
        summary = results["summary"]
        doc_counts = list(summary.keys())
        accuracies = [summary[c]["accuracy"] for c in doc_counts]
        latencies = [summary[c]["avg_latency_ms"] for c in doc_counts]

        graphs = []

        # Accuracy vs Size
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(doc_counts, accuracies, "o-", color="#3498db", linewidth=2, markersize=8)
        ax.set_xlabel("Number of Documents")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Experiment 2: Accuracy Degradation vs Context Size")
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        graph1 = self.get_graph_path("accuracy_vs_size")
        plt.savefig(graph1, dpi=GRAPH_DPI)
        plt.close()
        graphs.append(graph1)

        # Latency vs Size
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(doc_counts, latencies, "o-", color="#e74c3c", linewidth=2, markersize=8)
        ax.set_xlabel("Number of Documents")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Experiment 2: Latency Increase vs Context Size")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        graph2 = self.get_graph_path("latency_vs_size")
        plt.savefig(graph2, dpi=GRAPH_DPI)
        plt.close()
        graphs.append(graph2)

        self.logger.info(f"Graphs saved: {graphs}")
        return graphs


def run_experiment(model_backend: str = "mock") -> Dict[str, Any]:
    """Run Experiment 2 and return results."""
    exp = ContextSizeExperiment(model_backend=model_backend)
    return exp.execute()
