"""Evaluation framework for multi-agent RL experiments."""

from evaluation.plotting import (
    create_scalability_report,
    load_results_from_json,
    plot_comparison,
    plot_multiple_metrics,
    plot_scalability_curve,
    save_results_to_json,
)
from evaluation.utils import (
    create_eval_env_with_normalization,
    evaluate_on_multiple_sizes,
    evaluate_policy,
    load_model_with_normalization,
    print_scalability_summary,
)

__all__ = [
    # Utils
    "load_model_with_normalization",
    "create_eval_env_with_normalization",
    "evaluate_policy",
    "evaluate_on_multiple_sizes",
    "print_scalability_summary",
    # Plotting
    "plot_scalability_curve",
    "plot_multiple_metrics",
    "plot_comparison",
    "save_results_to_json",
    "load_results_from_json",
    "create_scalability_report",
]
