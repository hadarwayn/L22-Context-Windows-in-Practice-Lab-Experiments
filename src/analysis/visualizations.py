"""
Visualization module for Context Windows Lab.

Provides functions for generating experiment graphs.
"""

from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import seaborn as sns

from src.utils.config import GRAPH_DPI, GRAPHS_DIR


# Set style
sns.set_theme(style="whitegrid")


def create_accuracy_by_position_chart(
    results: Dict[str, float],
    output: Optional[Path] = None,
    title: str = "Accuracy by Position"
) -> Path:
    """
    Create bar chart showing accuracy by position.

    Args:
        results: Dict mapping position to accuracy
        output: Output path (auto-generated if None)
        title: Chart title

    Returns:
        Path to saved graph
    """
    if output is None:
        output = GRAPHS_DIR / "accuracy_by_position.png"

    positions = list(results.keys())
    accuracies = list(results.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2ecc71" if p in ["start", "end"] else "#e74c3c" for p in positions]
    bars = ax.bar(positions, accuracies, color=colors)

    ax.set_xlabel("Position")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)

    for bar, acc in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                f"{acc:.1f}%", ha="center")

    plt.tight_layout()
    plt.savefig(output, dpi=GRAPH_DPI)
    plt.close()
    return output


def create_accuracy_vs_size_chart(
    results: Dict[int, float],
    output: Optional[Path] = None,
    title: str = "Accuracy vs Context Size"
) -> Path:
    """
    Create line chart showing accuracy degradation.

    Args:
        results: Dict mapping doc count to accuracy
        output: Output path
        title: Chart title

    Returns:
        Path to saved graph
    """
    if output is None:
        output = GRAPHS_DIR / "accuracy_vs_size.png"

    x = list(results.keys())
    y = list(results.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o-", color="#3498db", linewidth=2, markersize=8)

    ax.set_xlabel("Number of Documents")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=GRAPH_DPI)
    plt.close()
    return output


def create_latency_vs_size_chart(
    results: Dict[int, float],
    output: Optional[Path] = None,
    title: str = "Latency vs Context Size"
) -> Path:
    """
    Create line chart showing latency increase.

    Args:
        results: Dict mapping doc count to latency
        output: Output path
        title: Chart title

    Returns:
        Path to saved graph
    """
    if output is None:
        output = GRAPHS_DIR / "latency_vs_size.png"

    x = list(results.keys())
    y = list(results.values())

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(x, y, "o-", color="#e74c3c", linewidth=2, markersize=8)

    ax.set_xlabel("Number of Documents")
    ax.set_ylabel("Latency (ms)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output, dpi=GRAPH_DPI)
    plt.close()
    return output


def create_comparison_chart(
    full_results: Dict[str, float],
    rag_results: Dict[str, float],
    output: Optional[Path] = None
) -> Path:
    """
    Create comparison chart for Full vs RAG.

    Args:
        full_results: Dict with full context metrics
        rag_results: Dict with RAG metrics
        output: Output path

    Returns:
        Path to saved graph
    """
    if output is None:
        output = GRAPHS_DIR / "rag_comparison.png"

    metrics = list(full_results.keys())
    full_values = list(full_results.values())
    rag_values = list(rag_results.values())

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar([i - width / 2 for i in x], full_values, width, label="Full Context", color="#e74c3c")
    ax.bar([i + width / 2 for i in x], rag_values, width, label="RAG", color="#2ecc71")

    ax.set_ylabel("Value")
    ax.set_title("Full Context vs RAG Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()

    plt.tight_layout()
    plt.savefig(output, dpi=GRAPH_DPI)
    plt.close()
    return output


def create_strategy_table(
    results: Dict[str, Dict[str, float]],
    output: Optional[Path] = None
) -> Path:
    """
    Create strategy performance visualization.

    Args:
        results: Dict of strategy -> metrics
        output: Output path

    Returns:
        Path to saved graph
    """
    if output is None:
        output = GRAPHS_DIR / "strategy_comparison.png"

    strategies = list(results.keys())
    metrics = ["Accuracy (%)", "Latency (ms)", "Tokens"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    colors = ["#3498db", "#2ecc71", "#9b59b6"]

    for i, (metric_key, metric_label) in enumerate([
        ("accuracy", "Accuracy (%)"),
        ("avg_latency_ms", "Latency (ms)"),
        ("avg_tokens", "Tokens")
    ]):
        values = [results[s].get(metric_key, 0) for s in strategies]
        axes[i].bar(strategies, values, color=colors)
        axes[i].set_ylabel(metric_label)
        axes[i].set_title(metric_label)

    plt.suptitle("Strategy Performance Comparison", fontsize=14)
    plt.tight_layout()
    plt.savefig(output, dpi=GRAPH_DPI)
    plt.close()
    return output
