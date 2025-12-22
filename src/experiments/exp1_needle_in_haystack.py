"""
Experiment 1: Needle in Haystack (Lost in the Middle)

Tests if LLMs retrieve information differently based on position.
"""

from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt

from src.experiments.base_experiment import BaseExperiment
from src.generators import generate_document
from src.models import get_model
from src.utils.config import (
    EXP1_NUM_DOCS,
    EXP1_WORDS_PER_DOC,
    EXP1_POSITIONS,
    TRIALS_PER_CONDITION,
    GRAPH_DPI,
)
from src.utils.helpers import Timer


class NeedleInHaystackExperiment(BaseExperiment):
    """Experiment 1: Test accuracy by fact position."""

    def __init__(self, model_backend: str = "mock"):
        super().__init__(
            name="exp1_needle_in_haystack",
            description="Test information retrieval accuracy by position (start/middle/end)"
        )
        self.model = get_model(backend=model_backend)
        self.fact = "The company CEO is David Cohen."
        self.question = "Who is the company CEO?"
        self.expected_answer = "David Cohen"

    def run(self) -> Dict[str, Any]:
        """Run the needle in haystack experiment."""
        results = {pos: [] for pos in EXP1_POSITIONS}

        total_trials = len(EXP1_POSITIONS) * TRIALS_PER_CONDITION
        self.logger.info(f"Running {total_trials} trials...")

        for position in EXP1_POSITIONS:
            self.logger.info(f"Testing position: {position}")

            for trial in range(TRIALS_PER_CONDITION):
                # Generate document with fact at position
                document = generate_document(
                    total_words=EXP1_WORDS_PER_DOC,
                    fact=self.fact,
                    position=position,
                    topic="business"
                )

                # Query model
                with Timer() as timer:
                    response = self.model.query(document, self.question)

                # Evaluate accuracy
                is_correct = self.expected_answer.lower() in response.lower()

                results[position].append({
                    "trial": trial + 1,
                    "correct": is_correct,
                    "latency_ms": timer.elapsed_ms,
                    "response_preview": response[:100]
                })

        # Calculate summary statistics
        summary = {}
        for position in EXP1_POSITIONS:
            trials = results[position]
            correct_count = sum(1 for t in trials if t["correct"])
            summary[position] = {
                "accuracy": correct_count / len(trials) * 100,
                "avg_latency_ms": sum(t["latency_ms"] for t in trials) / len(trials)
            }

        return {"trials": results, "summary": summary}

    def generate_graphs(self, results: Dict[str, Any]) -> List[Path]:
        """Generate accuracy by position bar chart."""
        summary = results["summary"]

        positions = list(summary.keys())
        accuracies = [summary[p]["accuracy"] for p in positions]

        # Create bar chart
        fig, ax = plt.subplots(figsize=(8, 6))
        bars = ax.bar(positions, accuracies, color=["#2ecc71", "#e74c3c", "#2ecc71"])

        ax.set_xlabel("Fact Position in Document")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Experiment 1: Accuracy by Position (Lost in the Middle)")
        ax.set_ylim(0, 100)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 2,
                f"{acc:.1f}%",
                ha="center",
                fontsize=12
            )

        plt.tight_layout()
        graph_path = self.get_graph_path("accuracy_by_position")
        plt.savefig(graph_path, dpi=GRAPH_DPI)
        plt.close()

        self.logger.info(f"Graph saved to {graph_path}")
        return [graph_path]


def run_experiment(model_backend: str = "mock") -> Dict[str, Any]:
    """Run Experiment 1 and return results."""
    exp = NeedleInHaystackExperiment(model_backend=model_backend)
    return exp.execute()
