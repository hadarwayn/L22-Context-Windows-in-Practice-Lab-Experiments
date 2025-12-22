"""Experiment implementations for Context Windows Lab."""

from .base_experiment import BaseExperiment
from .exp1_needle_in_haystack import NeedleInHaystackExperiment, run_experiment as run_exp1
from .exp2_context_size import ContextSizeExperiment, run_experiment as run_exp2
from .exp3_rag_impact import RAGImpactExperiment, run_experiment as run_exp3
from .exp4_strategies import StrategiesExperiment, run_experiment as run_exp4

__all__ = [
    "BaseExperiment",
    "NeedleInHaystackExperiment",
    "ContextSizeExperiment",
    "RAGImpactExperiment",
    "StrategiesExperiment",
    "run_exp1",
    "run_exp2",
    "run_exp3",
    "run_exp4",
]


def run_all_experiments(model_backend: str = "mock"):
    """Run all 4 experiments and return results."""
    results = {}
    results["exp1"] = run_exp1(model_backend)
    results["exp2"] = run_exp2(model_backend)
    results["exp3"] = run_exp3(model_backend)
    results["exp4"] = run_exp4(model_backend)
    return results
