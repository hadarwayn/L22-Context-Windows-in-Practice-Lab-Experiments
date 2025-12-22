"""Statistical analysis and visualization modules."""

from .statistics import (
    calculate_accuracy,
    calculate_mean_latency,
    calculate_std,
    calculate_confidence_interval,
    compare_groups,
    find_threshold,
)
from .visualizations import (
    create_accuracy_by_position_chart,
    create_accuracy_vs_size_chart,
    create_latency_vs_size_chart,
    create_comparison_chart,
    create_strategy_table,
)
from .explanations import (
    explain_needle_results,
    explain_context_size_results,
    explain_rag_results,
    explain_strategy_results,
    derive_prompt_rules,
    generate_full_report,
)

__all__ = [
    "calculate_accuracy",
    "calculate_mean_latency",
    "calculate_std",
    "calculate_confidence_interval",
    "compare_groups",
    "find_threshold",
    "create_accuracy_by_position_chart",
    "create_accuracy_vs_size_chart",
    "create_latency_vs_size_chart",
    "create_comparison_chart",
    "create_strategy_table",
    "explain_needle_results",
    "explain_context_size_results",
    "explain_rag_results",
    "explain_strategy_results",
    "derive_prompt_rules",
    "generate_full_report",
]
