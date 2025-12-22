"""Tests for analysis functions."""

import pytest
import numpy as np

from src.analysis import (
    calculate_accuracy,
    calculate_mean_latency,
    calculate_std,
    calculate_confidence_interval,
    compare_groups,
    find_threshold,
)


class TestAccuracyCalculation:
    """Tests for accuracy calculation."""

    def test_perfect_accuracy(self):
        """Test 100% accuracy."""
        results = [True, True, True, True, True]
        assert calculate_accuracy(results) == 100.0

    def test_zero_accuracy(self):
        """Test 0% accuracy."""
        results = [False, False, False]
        assert calculate_accuracy(results) == 0.0

    def test_partial_accuracy(self):
        """Test partial accuracy."""
        results = [True, True, False, False]
        assert calculate_accuracy(results) == 50.0

    def test_empty_list(self):
        """Test empty list returns 0."""
        assert calculate_accuracy([]) == 0.0


class TestLatencyCalculation:
    """Tests for latency calculation."""

    def test_mean_latency(self):
        """Test mean latency calculation."""
        latencies = [100.0, 200.0, 300.0]
        assert calculate_mean_latency(latencies) == 200.0

    def test_single_value(self):
        """Test single value."""
        latencies = [150.0]
        assert calculate_mean_latency(latencies) == 150.0

    def test_empty_list(self):
        """Test empty list returns 0."""
        assert calculate_mean_latency([]) == 0.0


class TestStatisticalFunctions:
    """Tests for statistical functions."""

    def test_std_calculation(self):
        """Test standard deviation calculation."""
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        std = calculate_std(values)
        assert std > 0
        assert abs(std - np.std(values, ddof=1)) < 0.001

    def test_confidence_interval(self):
        """Test confidence interval calculation."""
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        mean, lower, upper = calculate_confidence_interval(data)
        assert mean == 3.0
        assert lower < mean
        assert upper > mean

    def test_compare_groups(self):
        """Test group comparison."""
        group1 = np.array([1.0, 2.0, 3.0])
        group2 = np.array([4.0, 5.0, 6.0])
        result = compare_groups(group1, group2)

        assert result["group1_mean"] == 2.0
        assert result["group2_mean"] == 5.0
        assert result["difference"] == 3.0

    def test_find_threshold(self):
        """Test threshold finding."""
        x = np.array([0, 10, 20, 30, 40])
        y = np.array([100, 80, 60, 40, 20])
        threshold = find_threshold(x, y, target=50.0)
        # Should be between 20 and 30
        assert 20 <= threshold <= 30


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
