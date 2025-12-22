"""
Experiment 3: RAG Impact

Compares Full Context vs RAG retrieval for accuracy and latency.
"""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from src.experiments.base_experiment import BaseExperiment
from src.generators import generate_hebrew_documents
from src.models import get_model
from src.rag import create_vector_store, add_documents, similarity_search, clear_collection
from src.utils.config import EXP3_NUM_DOCS, EXP3_TOPICS, EXP3_RAG_TOP_K, TRIALS_PER_CONDITION, GRAPH_DPI
from src.utils.helpers import Timer


class RAGImpactExperiment(BaseExperiment):
    """Experiment 3: Compare Full Context vs RAG retrieval."""

    def __init__(self, model_backend: str = "mock"):
        super().__init__(
            name="exp3_rag_impact",
            description="Compare Full Context vs RAG for accuracy and latency"
        )
        self.model = get_model(backend=model_backend)
        self.question = "מהן תופעות הלוואי של התרופה?"
        self.expected_keywords = ["סחרחורת", "בחילה"]

    def run(self) -> Dict[str, Any]:
        """Run the RAG impact experiment."""
        # Generate Hebrew documents
        self.logger.info(f"Generating {EXP3_NUM_DOCS} Hebrew documents...")
        docs_with_topics = generate_hebrew_documents(EXP3_NUM_DOCS, EXP3_TOPICS)
        documents = [doc for doc, _ in docs_with_topics]

        # Set up vector store
        self.logger.info("Setting up vector store...")
        collection = create_vector_store("rag_experiment")
        clear_collection(collection)
        add_documents(collection, documents)

        results = {"full_context": [], "rag": []}

        for trial in range(TRIALS_PER_CONDITION):
            self.logger.info(f"Trial {trial + 1}/{TRIALS_PER_CONDITION}")

            # Mode A: Full Context
            full_context = "\n\n".join(documents)
            with Timer() as timer_full:
                response_full = self.model.query(full_context, self.question)

            is_correct_full = any(kw in response_full for kw in self.expected_keywords)
            results["full_context"].append({
                "trial": trial + 1,
                "correct": is_correct_full,
                "latency_ms": timer_full.elapsed_ms,
                "context_length": len(full_context)
            })

            # Mode B: RAG
            with Timer() as timer_rag:
                relevant_docs = similarity_search(collection, self.question, k=EXP3_RAG_TOP_K)
                rag_context = "\n\n".join([doc for doc, _ in relevant_docs])
                response_rag = self.model.query(rag_context, self.question)

            is_correct_rag = any(kw in response_rag for kw in self.expected_keywords)
            results["rag"].append({
                "trial": trial + 1,
                "correct": is_correct_rag,
                "latency_ms": timer_rag.elapsed_ms,
                "context_length": len(rag_context)
            })

        # Calculate summary
        summary = {}
        for mode in ["full_context", "rag"]:
            trials = results[mode]
            correct_count = sum(1 for t in trials if t["correct"])
            summary[mode] = {
                "accuracy": correct_count / len(trials) * 100,
                "avg_latency_ms": sum(t["latency_ms"] for t in trials) / len(trials),
                "avg_context_length": sum(t["context_length"] for t in trials) / len(trials)
            }

        return {"trials": results, "summary": summary}

    def generate_graphs(self, results: Dict[str, Any]) -> List[Path]:
        """Generate comparison chart."""
        summary = results["summary"]

        modes = ["Full Context", "RAG"]
        accuracies = [summary["full_context"]["accuracy"], summary["rag"]["accuracy"]]
        latencies = [summary["full_context"]["avg_latency_ms"], summary["rag"]["avg_latency_ms"]]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Accuracy comparison
        bars1 = ax1.bar(modes, accuracies, color=["#e74c3c", "#2ecc71"])
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Accuracy: Full Context vs RAG")
        ax1.set_ylim(0, 100)
        for bar, acc in zip(bars1, accuracies):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                    f"{acc:.1f}%", ha="center", fontsize=11)

        # Latency comparison
        bars2 = ax2.bar(modes, latencies, color=["#e74c3c", "#2ecc71"])
        ax2.set_ylabel("Latency (ms)")
        ax2.set_title("Latency: Full Context vs RAG")
        for bar, lat in zip(bars2, latencies):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{lat:.1f}ms", ha="center", fontsize=11)

        plt.suptitle("Experiment 3: RAG Impact", fontsize=14)
        plt.tight_layout()

        graph_path = self.get_graph_path("performance_comparison")
        plt.savefig(graph_path, dpi=GRAPH_DPI)
        plt.close()

        self.logger.info(f"Graph saved to {graph_path}")
        return [graph_path]


def run_experiment(model_backend: str = "mock") -> Dict[str, Any]:
    """Run Experiment 3 and return results."""
    exp = RAGImpactExperiment(model_backend=model_backend)
    return exp.execute()
