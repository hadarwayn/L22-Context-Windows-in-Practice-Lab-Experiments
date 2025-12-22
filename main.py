#!/usr/bin/env python3
"""
Context Windows in Practice - Lab Experiments

Main entry point for running all 4 experiments on LLM context windows.

Usage:
    python main.py                    # Run all experiments
    python main.py --experiment 1     # Run specific experiment
    python main.py --backend ollama   # Use Ollama backend
    python main.py --help             # Show help
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from tqdm import tqdm

from src.utils.config import ensure_directories, EXPERIMENTS_DIR
from src.utils.logger import setup_logger
from src.experiments import run_exp1, run_exp2, run_exp3, run_exp4
from src.analysis import generate_full_report


EXPERIMENTS = {
    1: ("Needle in Haystack (Lost in the Middle)", run_exp1),
    2: ("Context Window Size Impact", run_exp2),
    3: ("RAG Impact", run_exp3),
    4: ("Context Engineering Strategies", run_exp4),
}


def print_banner():
    """Print welcome banner."""
    print("\n" + "=" * 60)
    print("  Context Windows in Practice - Lab Experiments")
    print("  LLM Context Window Empirical Research")
    print("=" * 60)
    print("\n4 Experiments:")
    for num, (name, _) in EXPERIMENTS.items():
        print(f"  {num}. {name}")
    print()


def run_single_experiment(exp_num: int, backend: str) -> dict:
    """Run a single experiment."""
    if exp_num not in EXPERIMENTS:
        print(f"Error: Experiment {exp_num} not found. Choose 1-4.")
        sys.exit(1)

    name, run_func = EXPERIMENTS[exp_num]
    print(f"\nRunning Experiment {exp_num}: {name}")
    print("-" * 50)

    return run_func(model_backend=backend)


def run_all_experiments(backend: str) -> dict:
    """Run all experiments with progress bar."""
    results = {}

    for exp_num in tqdm(EXPERIMENTS.keys(), desc="Experiments", unit="exp"):
        name, run_func = EXPERIMENTS[exp_num]
        tqdm.write(f"\n>>> Experiment {exp_num}: {name}")
        results[f"exp{exp_num}"] = run_func(model_backend=backend)

    return results


def print_summary(results: dict):
    """Print experiment summary with detailed explanations."""
    print("\n" + "=" * 60)
    print("  EXPERIMENT SUMMARY")
    print("=" * 60)

    for exp_key, exp_results in results.items():
        if "results" in exp_results:
            summary = exp_results["results"].get("summary", {})
            print(f"\n{exp_key.upper()}:")

            if isinstance(summary, dict):
                for key, value in summary.items():
                    if isinstance(value, dict):
                        acc = value.get("accuracy", "N/A")
                        print(f"  {key}: {acc:.1f}% accuracy" if isinstance(acc, float) else f"  {key}: {acc}")
                    else:
                        print(f"  {key}: {value}")

    # Generate and save detailed report
    report = generate_full_report(results)
    report_path = EXPERIMENTS_DIR / "full_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)

    print("\n" + "=" * 60)
    print("  OUTPUT FILES:")
    print(f"    Graphs: results/graphs/")
    print(f"    Results: results/experiments/")
    print(f"    Full Report: {report_path}")
    print("=" * 60)
    print("\n  View the full report for:")
    print("    - Detailed explanations of each experiment")
    print("    - Prompt engineering rules derived from results")
    print("=" * 60 + "\n")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Context Windows Lab - Run LLM Experiments"
    )
    parser.add_argument(
        "--experiment", "-e",
        type=int,
        choices=[1, 2, 3, 4],
        help="Run specific experiment (1-4)"
    )
    parser.add_argument(
        "--backend", "-b",
        type=str,
        default="mock",
        choices=["mock", "ollama", "claude", "gemini"],
        help="LLM backend: mock, ollama, claude, gemini (default: mock)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    # Setup
    logger = setup_logger(log_level="DEBUG" if args.verbose else "INFO")
    ensure_directories()

    print_banner()

    print(f"Backend: {args.backend}")
    print(f"Mode: {'Single experiment' if args.experiment else 'All experiments'}")

    # Run experiments
    if args.experiment:
        results = {f"exp{args.experiment}": run_single_experiment(args.experiment, args.backend)}
    else:
        results = run_all_experiments(args.backend)

    # Print summary
    print_summary(results)

    print("Done!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
