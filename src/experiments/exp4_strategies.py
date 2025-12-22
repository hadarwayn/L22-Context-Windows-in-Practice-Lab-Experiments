"""
Experiment 4: Context Engineering Strategies

Compares Select, Compress, and Write strategies for context management.
"""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from src.experiments.base_experiment import BaseExperiment
from src.generators import generate_filler_text
from src.models import get_model
from src.rag import create_vector_store, add_documents, similarity_search, clear_collection
from src.utils.config import EXP4_NUM_ACTIONS, EXP4_STRATEGIES, GRAPH_DPI
from src.utils.helpers import Timer, count_tokens


class StrategiesExperiment(BaseExperiment):
    """Experiment 4: Compare context management strategies."""

    def __init__(self, model_backend: str = "mock"):
        super().__init__(
            name="exp4_strategies",
            description="Compare Select, Compress, Write strategies"
        )
        self.model = get_model(backend=model_backend)
        self.max_tokens = 4000

    def _generate_action_output(self, action_num: int) -> str:
        """Generate simulated agent action output."""
        output = generate_filler_text(100, "technology")
        # Add key fact every 3rd action
        if action_num % 3 == 0:
            fact = f"Key finding {action_num}: The system processed {action_num * 100} records."
            output = fact + " " + output
        return output

    def _select_strategy(self, history: List[str], query: str) -> str:
        """SELECT: Use RAG to retrieve relevant history."""
        collection = create_vector_store("strategy_select")
        clear_collection(collection)
        add_documents(collection, history)
        relevant = similarity_search(collection, query, k=min(5, len(history)))
        return "\n".join([doc for doc, _ in relevant])

    def _compress_strategy(self, history: List[str]) -> str:
        """COMPRESS: Summarize history when it gets too long."""
        full_history = "\n".join(history)
        if count_tokens(full_history) > self.max_tokens:
            # Simple compression: keep first and last few items
            compressed = history[:2] + ["[... compressed ...]"] + history[-2:]
            return "\n".join(compressed)
        return full_history

    def _write_strategy(self, history: List[str], scratchpad: Dict) -> str:
        """WRITE: Store key facts in external scratchpad."""
        # Extract key facts
        for i, item in enumerate(history):
            if "Key finding" in item:
                key = f"finding_{i}"
                scratchpad[key] = item.split(":")[1].strip() if ":" in item else item

        # Return scratchpad contents
        return "\n".join(f"- {v}" for v in scratchpad.values())

    def run(self) -> Dict[str, Any]:
        """Run the strategies experiment."""
        results = {strategy: [] for strategy in EXP4_STRATEGIES}
        query = "What were the key findings about record processing?"

        for strategy in EXP4_STRATEGIES:
            self.logger.info(f"Testing strategy: {strategy}")
            history = []
            scratchpad = {}

            for action in range(EXP4_NUM_ACTIONS):
                # Generate action output
                output = self._generate_action_output(action)
                history.append(output)

                # Apply strategy
                with Timer() as timer:
                    if strategy == "select":
                        context = self._select_strategy(history, query)
                    elif strategy == "compress":
                        context = self._compress_strategy(history)
                    else:  # write
                        context = self._write_strategy(history, scratchpad)

                    response = self.model.query(context, query)

                # Evaluate (check if key facts are retrievable)
                has_finding = "finding" in response.lower() or "record" in response.lower()

                results[strategy].append({
                    "action": action + 1,
                    "correct": has_finding,
                    "latency_ms": timer.elapsed_ms,
                    "context_tokens": count_tokens(context)
                })

        # Calculate summary
        summary = {}
        for strategy in EXP4_STRATEGIES:
            trials = results[strategy]
            correct_count = sum(1 for t in trials if t["correct"])
            summary[strategy] = {
                "accuracy": correct_count / len(trials) * 100,
                "avg_latency_ms": sum(t["latency_ms"] for t in trials) / len(trials),
                "avg_tokens": sum(t["context_tokens"] for t in trials) / len(trials)
            }

        return {"trials": results, "summary": summary}

    def generate_graphs(self, results: Dict[str, Any]) -> List[Path]:
        """Generate strategy performance chart."""
        summary = results["summary"]

        strategies = [s.upper() for s in EXP4_STRATEGIES]
        accuracies = [summary[s]["accuracy"] for s in EXP4_STRATEGIES]
        latencies = [summary[s]["avg_latency_ms"] for s in EXP4_STRATEGIES]
        tokens = [summary[s]["avg_tokens"] for s in EXP4_STRATEGIES]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))

        colors = ["#3498db", "#2ecc71", "#9b59b6"]

        # Accuracy
        axes[0].bar(strategies, accuracies, color=colors)
        axes[0].set_ylabel("Accuracy (%)")
        axes[0].set_title("Accuracy by Strategy")
        axes[0].set_ylim(0, 100)

        # Latency
        axes[1].bar(strategies, latencies, color=colors)
        axes[1].set_ylabel("Latency (ms)")
        axes[1].set_title("Average Latency")

        # Token usage
        axes[2].bar(strategies, tokens, color=colors)
        axes[2].set_ylabel("Tokens")
        axes[2].set_title("Average Context Size")

        plt.suptitle("Experiment 4: Context Engineering Strategies", fontsize=14)
        plt.tight_layout()

        graph_path = self.get_graph_path("strategy_performance")
        plt.savefig(graph_path, dpi=GRAPH_DPI)
        plt.close()

        self.logger.info(f"Graph saved to {graph_path}")
        return [graph_path]


def run_experiment(model_backend: str = "mock") -> Dict[str, Any]:
    """Run Experiment 4 and return results."""
    exp = StrategiesExperiment(model_backend=model_backend)
    return exp.execute()
