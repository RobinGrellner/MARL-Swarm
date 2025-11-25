"""
Plotting utilities for multi-agent RL experiments.

This module provides functions for:
- Generating scalability plots (performance vs swarm size)
- Reading and visualizing TensorBoard logs
- Creating publication-ready figures for thesis
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np


def plot_scalability_curve(
    results: Dict[int, Dict[str, float]],
    metric: str = "mean_reward",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (10, 6),
) -> plt.Figure:
    """Plot a scalability curve showing metric vs swarm size.

    Args:
        results: Dictionary mapping swarm size to evaluation metrics
        metric: Metric to plot (e.g., 'mean_reward', 'success_rate', 'mean_final_max_dist')
        ylabel: Y-axis label (defaults to metric name)
        title: Plot title
        save_path: Path to save figure (if provided)
        show: Whether to display the plot
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure object
    """
    swarm_sizes = sorted(results.keys())
    means = [results[size][metric] for size in swarm_sizes]

    # Get standard deviation if available
    std_key = metric.replace("mean_", "std_")
    if std_key in results[swarm_sizes[0]]:
        stds = [results[size][std_key] for size in swarm_sizes]
    else:
        stds = None

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot with error bars if std is available
    if stds is not None:
        ax.errorbar(
            swarm_sizes,
            means,
            yerr=stds,
            marker="o",
            markersize=8,
            linewidth=2,
            capsize=5,
            capthick=2,
            label=metric.replace("_", " ").title(),
        )
    else:
        ax.plot(
            swarm_sizes,
            means,
            marker="o",
            markersize=8,
            linewidth=2,
            label=metric.replace("_", " ").title(),
        )

    # Formatting
    ax.set_xlabel("Number of Agents", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel or metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.set_title(title or f"{metric.replace('_', ' ').title()} vs Swarm Size", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10)

    # Set x-axis to show all swarm sizes
    ax.set_xticks(swarm_sizes)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved figure to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_multiple_metrics(
    results: Dict[int, Dict[str, float]],
    metrics: List[str],
    titles: Optional[List[str]] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (15, 10),
) -> plt.Figure:
    """Plot multiple metrics in subplots.

    Args:
        results: Dictionary mapping swarm size to evaluation metrics
        metrics: List of metrics to plot
        titles: List of subplot titles (optional)
        save_path: Path to save figure
        show: Whether to display the plot
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    n_metrics = len(metrics)
    n_cols = min(3, n_metrics)
    n_rows = (n_metrics + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    swarm_sizes = sorted(results.keys())

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        means = [results[size][metric] for size in swarm_sizes]

        # Get standard deviation if available
        std_key = metric.replace("mean_", "std_")
        if std_key in results[swarm_sizes[0]]:
            stds = [results[size][std_key] for size in swarm_sizes]
            ax.errorbar(
                swarm_sizes,
                means,
                yerr=stds,
                marker="o",
                markersize=6,
                linewidth=2,
                capsize=4,
                capthick=1.5,
            )
        else:
            ax.plot(swarm_sizes, means, marker="o", markersize=6, linewidth=2)

        # Formatting
        ax.set_xlabel("Number of Agents", fontsize=10)
        ax.set_ylabel(metric.replace("_", " ").title(), fontsize=10)
        if titles and idx < len(titles):
            ax.set_title(titles[idx], fontsize=11, fontweight="bold")
        else:
            ax.set_title(metric.replace("_", " ").title(), fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xticks(swarm_sizes)

    # Hide unused subplots
    for idx in range(n_metrics, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved figure to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def plot_comparison(
    results_dict: Dict[str, Dict[int, Dict[str, float]]],
    metric: str = "mean_reward",
    ylabel: Optional[str] = None,
    title: Optional[str] = None,
    save_path: Optional[Path] = None,
    show: bool = True,
    figsize: Tuple[int, int] = (12, 7),
) -> plt.Figure:
    """Plot comparison of multiple experiments.

    Args:
        results_dict: Dictionary mapping experiment name to results
        metric: Metric to plot
        ylabel: Y-axis label
        title: Plot title
        save_path: Path to save figure
        show: Whether to display the plot
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    for exp_name, results in results_dict.items():
        swarm_sizes = sorted(results.keys())
        means = [results[size][metric] for size in swarm_sizes]

        # Get standard deviation if available
        std_key = metric.replace("mean_", "std_")
        if std_key in results[swarm_sizes[0]]:
            stds = [results[size][std_key] for size in swarm_sizes]
            ax.errorbar(
                swarm_sizes,
                means,
                yerr=stds,
                marker="o",
                markersize=7,
                linewidth=2,
                capsize=4,
                capthick=1.5,
                label=exp_name,
            )
        else:
            ax.plot(
                swarm_sizes,
                means,
                marker="o",
                markersize=7,
                linewidth=2,
                label=exp_name,
            )

    # Formatting
    ax.set_xlabel("Number of Agents", fontsize=12, fontweight="bold")
    ax.set_ylabel(ylabel or metric.replace("_", " ").title(), fontsize=12, fontweight="bold")
    ax.set_title(title or f"Comparison: {metric.replace('_', ' ').title()}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(fontsize=10, loc="best")

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"✓ Saved figure to {save_path}")

    # Show if requested
    if show:
        plt.show()

    return fig


def save_results_to_json(
    results: Dict[int, Dict[str, float]],
    output_path: Path,
    metadata: Optional[Dict] = None,
) -> None:
    """Save evaluation results to JSON file.

    Args:
        results: Evaluation results dictionary
        output_path: Path to save JSON
        metadata: Optional metadata to include
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert results to serializable format
    output_data = {}
    for size, metrics in results.items():
        output_data[f"{size}_agents"] = metrics

    # Add metadata if provided
    if metadata:
        output_data["metadata"] = metadata

    # Save to JSON
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Results saved to {output_path}")


def load_results_from_json(json_path: Path) -> Tuple[Dict[int, Dict[str, float]], Optional[Dict]]:
    """Load evaluation results from JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        Tuple of (results dictionary, metadata if present)
    """
    json_path = Path(json_path)

    with open(json_path) as f:
        data = json.load(f)

    # Extract metadata if present
    metadata = data.pop("metadata", None)

    # Convert back to integer keys
    results = {}
    for key, value in data.items():
        if key.endswith("_agents"):
            num_agents = int(key.replace("_agents", ""))
            results[num_agents] = value

    return results, metadata


def create_scalability_report(
    results: Dict[int, Dict[str, float]],
    output_dir: Path,
    experiment_name: str = "scalability_experiment",
) -> None:
    """Generate a complete scalability report with multiple plots.

    Args:
        results: Evaluation results
        output_dir: Directory to save plots
        experiment_name: Name for the experiment (used in filenames)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating scalability report in {output_dir}/...")

    # Plot 1: Mean reward
    plot_scalability_curve(
        results,
        metric="mean_reward",
        ylabel="Mean Episode Reward",
        title="Scalability: Reward vs Swarm Size",
        save_path=output_dir / f"{experiment_name}_reward.png",
        show=False,
    )

    # Plot 2: Success rate
    plot_scalability_curve(
        results,
        metric="success_rate",
        ylabel="Success Rate",
        title="Scalability: Success Rate vs Swarm Size",
        save_path=output_dir / f"{experiment_name}_success_rate.png",
        show=False,
    )

    # Plot 3: Final max distance
    plot_scalability_curve(
        results,
        metric="mean_final_max_dist",
        ylabel="Final Maximum Pairwise Distance",
        title="Scalability: Final Distance vs Swarm Size",
        save_path=output_dir / f"{experiment_name}_final_distance.png",
        show=False,
    )

    # Plot 4: Multiple metrics overview
    plot_multiple_metrics(
        results,
        metrics=["mean_reward", "success_rate", "mean_final_max_dist", "mean_length"],
        save_path=output_dir / f"{experiment_name}_overview.png",
        show=False,
    )

    # Save results to JSON
    save_results_to_json(
        results,
        output_path=output_dir / f"{experiment_name}_results.json",
    )

    print(f"✓ Scalability report generated in {output_dir}/")
