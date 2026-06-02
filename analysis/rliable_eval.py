"""Reliable statistical evaluation of MARL-Swarm experiments.

Implements the evaluation protocol of Agarwal et al. (2021, "Deep Reinforcement
Learning at the Edge of the Statistical Precipice") on top of the ``rliable``
library: interquartile-mean (IQM) aggregate scores with stratified-bootstrap
confidence intervals, performance profiles, and probability-of-improvement
comparisons between configurations.

A *method* is an experiment configuration (e.g. ``embed_dim16``) and a *task* is
an evaluation swarm size. Scores for a method are arranged as an array of shape
``(n_seeds, n_tasks)``, the layout expected by rliable.

The module is importable and is driven by the ``analysis.run_analysis`` and
``analysis.run_comparison`` CLIs and by ``analyze_experiments.ipynb``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy.random import RandomState
from rliable import library as rly
from rliable import metrics, plot_utils

ScoreDict = Dict[str, np.ndarray]
IntervalEstimates = Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]
PathLike = Union[str, Path]

DEFAULT_REPS = 50_000
DEFAULT_CONFIDENCE = 0.95
AGGREGATE_METRIC_NAMES: Tuple[str, ...] = ("Median", "IQM", "Mean", "Optimality Gap")


def normalize_scores(
    scores: Mapping[str, np.ndarray],
    *,
    method: str = "min_max",
    reference: Optional[str] = None,
) -> ScoreDict:
    """Normalize scores per task so they share a common interval across tasks.

    Args:
        scores: Mapping from config name to a ``(n_seeds, n_tasks)`` matrix. All
            matrices must share the same task axis (number of columns).
        method: ``"min_max"`` rescales each task to ``[0, 1]`` using the minimum
            and maximum across all configs and seeds; ``"reference"`` divides each
            task by the per-task mean of ``reference`` (so the reference config
            scores around ``1.0``); ``"none"`` returns copies of the raw scores.
        reference: Reference config name, required for ``method="reference"``.

    Returns:
        Mapping from config name to the normalized score matrix.

    Raises:
        ValueError: If the matrices disagree on the number of tasks, the method is
            unknown, or a reference is required but missing/unknown.
    """
    configs = list(scores)
    if not configs:
        return {}

    n_tasks = scores[configs[0]].shape[1]
    if any(scores[config].shape[1] != n_tasks for config in configs):
        raise ValueError("All configs must share the same number of tasks to normalize.")

    if method == "none":
        return {config: np.array(scores[config], dtype=np.float64) for config in configs}

    if method == "min_max":
        stacked = np.concatenate([scores[config] for config in configs], axis=0)
        task_min = stacked.min(axis=0)
        task_range = stacked.max(axis=0) - task_min
        task_range[task_range == 0.0] = 1.0
        return {config: (scores[config] - task_min) / task_range for config in configs}

    if method == "reference":
        if reference is None:
            raise ValueError("normalize='reference' requires a reference config name.")
        if reference not in scores:
            raise ValueError(f"Reference config '{reference}' is not present in the scores.")
        task_mean = scores[reference].mean(axis=0)
        task_mean = np.where(task_mean == 0.0, 1.0, task_mean)
        return {config: scores[config] / task_mean for config in configs}

    raise ValueError(f"Unknown normalization method: {method!r}")


def _aggregate_metrics(scores: np.ndarray) -> np.ndarray:
    """Return Median, IQM, Mean and Optimality Gap for a score matrix."""
    return np.array(
        [
            metrics.aggregate_median(scores),
            metrics.aggregate_iqm(scores),
            metrics.aggregate_mean(scores),
            metrics.aggregate_optimality_gap(scores),
        ]
    )


def aggregate_iqm(
    scores_dict: Mapping[str, np.ndarray],
    *,
    reps: int = DEFAULT_REPS,
    confidence_interval_size: float = DEFAULT_CONFIDENCE,
    random_state: Optional[RandomState] = None,
) -> IntervalEstimates:
    """Aggregate point estimates and stratified-bootstrap confidence intervals.

    Computes Median, IQM, Mean and Optimality Gap (in the order of
    :data:`AGGREGATE_METRIC_NAMES`) for every config via
    :func:`rliable.library.get_interval_estimates`.

    Args:
        scores_dict: Mapping from config name to a ``(n_seeds, n_tasks)`` matrix.
        reps: Number of stratified-bootstrap replications.
        confidence_interval_size: Coverage of the confidence interval.
        random_state: Optional RNG for reproducible intervals.

    Returns:
        ``point_estimates`` mapping each config to an array of shape ``(4,)`` and
        ``interval_estimates`` mapping each config to an array of shape ``(2, 4)``
        whose rows hold the lower and upper interval bounds.
    """
    return rly.get_interval_estimates(
        dict(scores_dict),
        _aggregate_metrics,
        reps=reps,
        confidence_interval_size=confidence_interval_size,
        random_state=random_state,
    )


def performance_profile(
    scores_dict: Mapping[str, np.ndarray],
    taus: Sequence[float],
    *,
    reps: int = DEFAULT_REPS,
    confidence_interval_size: float = DEFAULT_CONFIDENCE,
) -> IntervalEstimates:
    """Compute score-distribution performance profiles with confidence bands.

    Args:
        scores_dict: Mapping from config name to a ``(n_seeds, n_tasks)`` matrix.
        taus: Thresholds at which the profile (fraction of runs with score > tau)
            is evaluated.
        reps: Number of stratified-bootstrap replications.
        confidence_interval_size: Coverage of the confidence interval.

    Returns:
        ``score_distributions`` mapping each config to a profile of shape
        ``(len(taus),)`` and ``score_distribution_cis`` mapping each config to the
        confidence band of shape ``(2, len(taus))``.
    """
    tau_array = np.asarray(taus, dtype=np.float64)
    return rly.create_performance_profile(
        dict(scores_dict),
        tau_array,
        reps=reps,
        confidence_interval_size=confidence_interval_size,
    )


def _improvement_probability(scores_x: np.ndarray, scores_y: np.ndarray) -> np.ndarray:
    """Wrap ``metrics.probability_of_improvement`` to return a 1D array."""
    return np.array([metrics.probability_of_improvement(scores_x, scores_y)])


def probability_of_improvement(
    scores_dict: Mapping[str, np.ndarray],
    pairs: Sequence[Tuple[str, str]],
    *,
    reps: int = DEFAULT_REPS,
    confidence_interval_size: float = DEFAULT_CONFIDENCE,
    random_state: Optional[RandomState] = None,
) -> IntervalEstimates:
    """Pairwise probability that config X outperforms config Y, with CIs.

    For each ``(x, y)`` pair, computes ``P(X > Y)`` averaged across tasks
    (Mann-Whitney based, see :func:`rliable.metrics.probability_of_improvement`)
    together with its stratified-bootstrap confidence interval.

    Args:
        scores_dict: Mapping from config name to a ``(n_seeds, n_tasks)`` matrix.
        pairs: Sequence of ``(config_x, config_y)`` config-name pairs to compare.
        reps: Number of stratified-bootstrap replications.
        confidence_interval_size: Coverage of the confidence interval.
        random_state: Optional RNG for reproducible intervals.

    Returns:
        ``point_estimates`` mapping ``"x,y"`` to an array of shape ``(1,)`` and
        ``interval_estimates`` mapping ``"x,y"`` to an array of shape ``(2, 1)``.

    Raises:
        ValueError: If any pair references an unknown config.
    """
    pair_scores: Dict[str, List[np.ndarray]] = {}
    for config_x, config_y in pairs:
        for config in (config_x, config_y):
            if config not in scores_dict:
                raise ValueError(f"Unknown config in improvement pair: {config!r}")
        pair_scores[f"{config_x},{config_y}"] = [scores_dict[config_x], scores_dict[config_y]]

    return rly.get_interval_estimates(
        pair_scores,
        _improvement_probability,
        reps=reps,
        confidence_interval_size=confidence_interval_size,
        random_state=random_state,
    )


def default_taus(scores_dict: Mapping[str, np.ndarray], num: int = 101) -> np.ndarray:
    """Return ``num`` evenly spaced thresholds spanning the observed score range."""
    stacked = np.concatenate([matrix.reshape(-1) for matrix in scores_dict.values()])
    low, high = float(np.min(stacked)), float(np.max(stacked))
    if low == high:
        high = low + 1.0
    return np.linspace(low, high, num)


def aggregate_summary_frame(
    point_estimates: Mapping[str, np.ndarray],
    interval_estimates: Mapping[str, np.ndarray],
    metric_names: Sequence[str] = AGGREGATE_METRIC_NAMES,
) -> pd.DataFrame:
    """Build a tidy ``config x metric`` table of point estimates and CI bounds."""
    rows = []
    for config in point_estimates:
        for index, metric_name in enumerate(metric_names):
            lower, upper = interval_estimates[config][:, index]
            rows.append(
                {
                    "config": config,
                    "metric": metric_name,
                    "point": float(point_estimates[config][index]),
                    "ci_low": float(lower),
                    "ci_high": float(upper),
                }
            )
    return pd.DataFrame(rows, columns=["config", "metric", "point", "ci_low", "ci_high"])


def probability_of_improvement_frame(
    point_estimates: Mapping[str, np.ndarray],
    interval_estimates: Mapping[str, np.ndarray],
) -> pd.DataFrame:
    """Build a table of pairwise improvement probabilities and CI bounds."""
    rows = []
    for key in point_estimates:
        config_x, config_y = key.split(",", 1)
        lower, upper = interval_estimates[key][:, 0]
        rows.append(
            {
                "config_x": config_x,
                "config_y": config_y,
                "p_improvement": float(point_estimates[key][0]),
                "ci_low": float(lower),
                "ci_high": float(upper),
            }
        )
    return pd.DataFrame(rows, columns=["config_x", "config_y", "p_improvement", "ci_low", "ci_high"])


def raw_scores_frame(scores_dict: Mapping[str, np.ndarray], task_labels: Sequence[str]) -> pd.DataFrame:
    """Flatten score matrices into a long ``config x run x task x score`` table."""
    rows = []
    for config, matrix in scores_dict.items():
        n_runs, n_tasks = matrix.shape
        for run in range(n_runs):
            for task in range(n_tasks):
                rows.append(
                    {
                        "config": config,
                        "run": run,
                        "task": task_labels[task],
                        "score": float(matrix[run, task]),
                    }
                )
    return pd.DataFrame(rows, columns=["config", "run", "task", "score"])


def summary_pivot(
    point_estimates: Mapping[str, np.ndarray],
    interval_estimates: Mapping[str, np.ndarray],
    metric_names: Sequence[str] = AGGREGATE_METRIC_NAMES,
) -> pd.DataFrame:
    """Pivot the aggregate summary into ``config x metric`` cells ``point [lo, hi]``."""
    frame = aggregate_summary_frame(point_estimates, interval_estimates, metric_names)
    frame["cell"] = frame.apply(
        lambda row: f"{row['point']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}]", axis=1
    )
    pivot = frame.pivot(index="config", columns="metric", values="cell").reindex(columns=list(metric_names))
    pivot.columns.name = None
    return pivot


def summary_text(
    point_estimates: Mapping[str, np.ndarray],
    interval_estimates: Mapping[str, np.ndarray],
    *,
    header: str = "",
    probability_of_improvement: Optional[pd.DataFrame] = None,
) -> str:
    """Render a plain-text summary: aggregate pivot plus optional improvement table."""
    lines: List[str] = []
    if header:
        lines += [header, "=" * len(header), ""]
    lines.append("Aggregate scores (point [95% CI]):")
    lines.append(summary_pivot(point_estimates, interval_estimates).to_string())
    if probability_of_improvement is not None and not probability_of_improvement.empty:
        lines += ["", "Probability of improvement vs reference:", probability_of_improvement.to_string(index=False)]
    return "\n".join(lines) + "\n"


def _save_figure(fig: plt.Figure, output_path: PathLike) -> Path:
    """Save ``fig`` as a raster image (format from the path suffix) and close it."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_aggregate_intervals(
    point_estimates: Mapping[str, np.ndarray],
    interval_estimates: Mapping[str, np.ndarray],
    *,
    output_path: PathLike,
    algorithms: Optional[Sequence[str]] = None,
    metric_names: Sequence[str] = AGGREGATE_METRIC_NAMES,
    xlabel: str = "Normalized Score",
) -> Path:
    """Plot aggregate metrics with 95% CIs and save the figure as an image."""
    algorithm_list = list(algorithms) if algorithms is not None else list(point_estimates)
    fig, _ = plot_utils.plot_interval_estimates(
        dict(point_estimates),
        dict(interval_estimates),
        metric_names=list(metric_names),
        algorithms=algorithm_list,
        xlabel=xlabel,
    )
    return _save_figure(fig, output_path)


def plot_profiles(
    score_distributions: Mapping[str, np.ndarray],
    score_distribution_cis: Mapping[str, np.ndarray],
    taus: Sequence[float],
    *,
    output_path: PathLike,
    xlabel: str = r"Normalized Score ($\tau$)",
) -> Path:
    """Plot performance profiles with confidence bands and save as an image."""
    fig, ax = plt.subplots(figsize=(8, 5))
    plot_utils.plot_performance_profiles(
        dict(score_distributions),
        np.asarray(taus, dtype=np.float64),
        performance_profile_cis=dict(score_distribution_cis),
        ax=ax,
        xlabel=xlabel,
    )
    ax.legend(loc="upper right")
    return _save_figure(fig, output_path)


def _iqm_only(scores: np.ndarray) -> np.ndarray:
    """Return the IQM as a 1D array (for per-task interval estimation)."""
    return np.array([metrics.aggregate_iqm(scores)])


def iqm_by_task(
    scores_dict: Mapping[str, np.ndarray],
    *,
    reps: int = DEFAULT_REPS,
    confidence_interval_size: float = DEFAULT_CONFIDENCE,
    random_state: Optional[RandomState] = None,
) -> IntervalEstimates:
    """Per-task IQM point estimate and CI for each config.

    Bootstraps each task column independently, giving one IQM (and interval) per
    task. Useful for showing how a config behaves across the task axis (e.g. swarm
    size).

    Returns:
        ``point_estimates`` mapping config to an array of shape ``(n_tasks,)`` and
        ``interval_estimates`` mapping config to an array of shape ``(2, n_tasks)``.
    """
    methods = list(scores_dict)
    if not methods:
        return {}, {}
    n_tasks = scores_dict[methods[0]].shape[1]
    keyed = {
        f"{method}@@{task}": scores_dict[method][:, [task]]
        for method in methods
        for task in range(n_tasks)
    }
    point, interval = rly.get_interval_estimates(
        keyed,
        _iqm_only,
        reps=reps,
        confidence_interval_size=confidence_interval_size,
        random_state=random_state,
    )
    point_by_method = {
        method: np.array([point[f"{method}@@{task}"][0] for task in range(n_tasks)]) for method in methods
    }
    interval_by_method = {
        method: np.stack([interval[f"{method}@@{task}"][:, 0] for task in range(n_tasks)], axis=1)
        for method in methods
    }
    return point_by_method, interval_by_method


def plot_iqm_by_task(
    point_estimates: Mapping[str, np.ndarray],
    interval_estimates: Mapping[str, np.ndarray],
    task_labels: Sequence[str],
    *,
    output_path: PathLike,
    xlabel: str = "Swarm size",
    ylabel: str = "IQM normalized score",
) -> Path:
    """Plot per-task IQM with confidence bands versus the task axis, saved as an image."""
    fig, ax = plt.subplots(figsize=(7, 5))
    positions = np.arange(len(task_labels))
    for method, point in point_estimates.items():
        lower, upper = interval_estimates[method]
        line, = ax.plot(positions, point, marker="o", label=method)
        ax.fill_between(positions, lower, upper, alpha=0.15, color=line.get_color())
    ax.set_xticks(positions)
    ax.set_xticklabels(task_labels)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="best")
    return _save_figure(fig, output_path)
