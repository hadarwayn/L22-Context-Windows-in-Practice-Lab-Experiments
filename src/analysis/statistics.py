"""
Statistical analysis functions for Context Windows Lab.

Provides functions for calculating accuracy, latency, and comparisons.
"""

from typing import List, Tuple

import numpy as np


def calculate_accuracy(results: List[bool]) -> float:
    """
    Calculate accuracy percentage from boolean results.

    Args:
        results: List of True/False values

    Returns:
        Accuracy as percentage (0-100)
    """
    if not results:
        return 0.0
    return sum(results) / len(results) * 100


def calculate_mean_latency(latencies: List[float]) -> float:
    """
    Calculate mean latency.

    Args:
        latencies: List of latency values in ms

    Returns:
        Mean latency
    """
    if not latencies:
        return 0.0
    return sum(latencies) / len(latencies)


def calculate_std(values: List[float]) -> float:
    """
    Calculate standard deviation.

    Args:
        values: List of numeric values

    Returns:
        Standard deviation
    """
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def calculate_confidence_interval(
    data: np.ndarray,
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    """
    Calculate confidence interval for data.

    Args:
        data: Array of values
        confidence: Confidence level (default 0.95)

    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    n = len(data)
    if n == 0:
        return (0.0, 0.0, 0.0)

    mean = float(np.mean(data))
    std_err = float(np.std(data, ddof=1) / np.sqrt(n)) if n > 1 else 0.0

    # Use z-score for 95% CI
    z = 1.96 if confidence == 0.95 else 2.576  # 99%

    lower = mean - z * std_err
    upper = mean + z * std_err

    return (mean, lower, upper)


def compare_groups(
    group1: np.ndarray,
    group2: np.ndarray
) -> dict:
    """
    Compare two groups statistically.

    Args:
        group1: First group data
        group2: Second group data

    Returns:
        Dictionary with comparison statistics
    """
    mean1 = float(np.mean(group1))
    mean2 = float(np.mean(group2))
    std1 = float(np.std(group1, ddof=1)) if len(group1) > 1 else 0.0
    std2 = float(np.std(group2, ddof=1)) if len(group2) > 1 else 0.0

    # Calculate effect size (Cohen's d)
    pooled_std = np.sqrt((std1**2 + std2**2) / 2) if (std1 + std2) > 0 else 1.0
    cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0.0

    return {
        "group1_mean": mean1,
        "group2_mean": mean2,
        "group1_std": std1,
        "group2_std": std2,
        "difference": mean2 - mean1,
        "percent_change": ((mean2 - mean1) / mean1 * 100) if mean1 != 0 else 0.0,
        "cohens_d": float(cohens_d)
    }


def find_threshold(x: np.ndarray, y: np.ndarray, target: float = 50.0) -> float:
    """
    Find x value where y crosses target threshold.

    Args:
        x: Independent variable values
        y: Dependent variable values
        target: Target y value to find threshold for

    Returns:
        Estimated x value at threshold
    """
    if len(x) != len(y) or len(x) < 2:
        return 0.0

    # Find where y crosses target
    for i in range(len(y) - 1):
        if (y[i] >= target and y[i + 1] < target) or \
           (y[i] <= target and y[i + 1] > target):
            # Linear interpolation
            slope = (y[i + 1] - y[i]) / (x[i + 1] - x[i]) if x[i + 1] != x[i] else 0
            if slope != 0:
                return float(x[i] + (target - y[i]) / slope)

    return float(x[-1])
