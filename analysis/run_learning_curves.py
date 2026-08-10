"""Chapter 6 learning-curve figures: IQM + bootstrap CI over training iterations.

Figures:

* fig_6_1 -- embedding_scaling: 2 rows (rendezvous, pursuit_evasion) x 4 columns
  (N in {4,16,50,100}), 6 lines per panel (embed_dim).
* fig_6_2 -- architecture_scalability, one figure per task (fig_6_2_..._rendezvous,
  fig_6_2_..._pursuit_evasion): 4 rows (N in {4,16,50,100}) x 3 columns
  (phi_layers L in {1,2,4}), 3 lines per panel (phi_hidden_width w). Split by task
  rather than faceted into one figure so a panel never has to carry more than 3
  lines + CI bands.

Also writes ``learning_curves.csv`` (the per-iteration IQM/CI data behind both
figures) and ``seed_coverage.csv`` (per-run completeness for every config in
scope, including the ones only needed by the Part B final-score pipeline).

Example::

    python -m analysis.run_learning_curves
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import log_loading as ll
from analysis import rliable_eval as rl

_ARCH_VARIANT_RE = re.compile(r"^phi_layers(\d+)_phi_hidden_width(\d+)$")

DEFAULT_REPS = 2000
SUBSAMPLE_STEP = 10
FONT_SIZE = 9

# Okabe-Ito colorblind-safe palette, black reserved (not used for data lines).
PALETTE = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", "#D55E00", "#CC79A7"]

TASKS = ["rendezvous", "pursuit_evasion"]
SIZES = [4, 16, 50, 100]
ARCH_TASKS = TASKS
ARCH_SIZES = SIZES
DEPTHS = [1, 2, 4]
WIDTHS = [32, 64, 128]

EMBED_DIMS = [4, 8, 16, 32, 64, 128]
EMBED_COLOR = {f"embed_dim{k}": PALETTE[i] for i, k in enumerate(EMBED_DIMS)}
WIDTH_COLOR = {w: PALETTE[i] for i, w in enumerate(WIDTHS)}


def embedding_config(task: str, size: int) -> str:
    return f"embedding_scaling_{task}_{size}agents_ppo"


def architecture_config(task: str, size: int) -> str:
    return f"architecture_scalability_{task}_{size}agents"


EMBEDDING_CONFIGS = [embedding_config(task, size) for task in TASKS for size in SIZES]
ARCHITECTURE_CONFIGS = [architecture_config(task, size) for task in ARCH_TASKS for size in ARCH_SIZES]
ALL_CONFIGS = EMBEDDING_CONFIGS + ARCHITECTURE_CONFIGS


def subsample_indices(n_points: int, step: int = SUBSAMPLE_STEP) -> np.ndarray:
    """Every ``step``-th index into a length-``n_points`` axis, always keeping the last."""
    idx = np.arange(0, n_points, step)
    if idx[-1] != n_points - 1:
        idx = np.append(idx, n_points - 1)
    return idx


def cellwide_minmax_normalize(curves: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Normalize every variant by one min/max pooled over all variants, runs and iterations."""
    all_values = np.concatenate([matrix.ravel() for matrix in curves.values()])
    lo, hi = float(all_values.min()), float(all_values.max())
    scale = hi - lo if hi > lo else 1.0
    return {variant: (matrix - lo) / scale for variant, matrix in curves.items()}


def pointwise_iqm(
    curves: Dict[str, np.ndarray],
    iterations: np.ndarray,
    *,
    reps: int = DEFAULT_REPS,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], np.ndarray]:
    """Subsample then per-iteration IQM/CI (via :func:`rl.iqm_by_task`, iteration = task)."""
    idx = subsample_indices(len(iterations))
    subsampled = {variant: matrix[:, idx] for variant, matrix in curves.items()}
    point, interval = rl.iqm_by_task(subsampled, reps=reps)
    return point, interval, iterations[idx]


def config_curve_summary(
    config_name: str,
    *,
    logs_dir: str = "logs",
    configs_dir: str = "training/configs",
    reps: int = DEFAULT_REPS,
    min_runs: int = 2,
) -> Dict[str, object]:
    """Load, normalize and summarize one config's learning curves."""
    result = ll.load_config_curves(config_name, logs_dir=logs_dir, configs_dir=configs_dir, min_runs=min_runs)
    normalized = cellwide_minmax_normalize(result.curves)
    point, interval, sub_iterations = pointwise_iqm(normalized, result.iterations, reps=reps)
    return {
        "config": config_name,
        "meta": result.meta,
        "anomalies": result.anomalies,
        "point": point,
        "interval": interval,
        "iterations": sub_iterations,
    }


def rc_context_kwargs() -> dict:
    return {
        "font.size": FONT_SIZE,
        "axes.labelsize": FONT_SIZE,
        "axes.titlesize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 1,
        "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1,
    }


def _plot_panel(ax, summary: Dict[str, object], variant_order: Sequence[str], color_of) -> None:
    for variant in variant_order:
        if variant not in summary["point"]:
            continue
        point = summary["point"][variant]
        lower, upper = summary["interval"][variant]
        color = color_of(variant)
        ax.plot(summary["iterations"], point, color=color, linewidth=1.0, label=variant)
        ax.fill_between(summary["iterations"], lower, upper, color=color, alpha=0.15, linewidth=0)
    ax.set_ylim(-0.02, 1.02)


def save_figure_png_pdf(fig, output_prefix: Path) -> List[Path]:
    output_prefix.parent.mkdir(parents=True, exist_ok=True)
    png_path = output_prefix.with_suffix(".png")
    pdf_path = output_prefix.with_suffix(".pdf")
    fig.savefig(png_path, dpi=200, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return [png_path, pdf_path]


def plot_embedding_grid(summaries: Dict[Tuple[str, int], dict], output_prefix: Path) -> List[Path]:
    """2 rows (task) x 4 cols (N), 6 lines (embed_dim) per panel, shared legend."""
    variant_order = [f"embed_dim{k}" for k in EMBED_DIMS]
    with plt.rc_context(rc_context_kwargs()):
        fig, axes = plt.subplots(2, 4, figsize=(9.0, 4.6), sharey=True)
        for row, task in enumerate(TASKS):
            for col, size in enumerate(SIZES):
                ax = axes[row, col]
                _plot_panel(ax, summaries[(task, size)], variant_order, lambda v: EMBED_COLOR[v])
                ax.set_title(f"{task}, N={size}")
                if row == 1:
                    ax.set_xlabel("Iteration")
                if col == 0:
                    ax.set_ylabel("Normalized reward")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(
            handles, labels, loc="lower center", ncol=len(variant_order), frameon=False, bbox_to_anchor=(0.5, -0.02)
        )
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        return save_figure_png_pdf(fig, output_prefix)


def plot_architecture_grid(summaries: Dict[int, dict], output_prefix: Path) -> List[Path]:
    """One task's architecture grid: len(ARCH_SIZES) rows (N) x 3 cols (depth L),

    3 lines (width w) per panel, shared legend. Called once per task so no panel
    ever has to carry more than 3 lines + CI bands.
    """
    n_rows = len(ARCH_SIZES)
    with plt.rc_context(rc_context_kwargs()):
        fig, axes = plt.subplots(n_rows, 3, figsize=(7.5, 2.4 * n_rows), sharey=True)
        for row, size in enumerate(ARCH_SIZES):
            summary = summaries[size]
            for col, depth in enumerate(DEPTHS):
                ax = axes[row, col]
                variant_order = [f"phi_layers{depth}_phi_hidden_width{w}" for w in WIDTHS]
                _plot_panel(ax, summary, variant_order, lambda v: WIDTH_COLOR[int(v.rsplit("width", 1)[1])])
                ax.set_title(f"N={size}, L={depth}")
                if row == n_rows - 1:
                    ax.set_xlabel("Iteration")
                if col == 0:
                    ax.set_ylabel("Normalized reward")
        handles_raw, labels_raw = axes[0, 0].get_legend_handles_labels()
        labels = [f"w={w}" for w in WIDTHS]
        fig.legend(
            handles_raw, labels, loc="lower center", ncol=len(WIDTHS), frameon=False, bbox_to_anchor=(0.5, -0.02)
        )
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        return save_figure_png_pdf(fig, output_prefix)


def _summary_to_rows(summary: Dict[str, object], figure: str, extra: Dict[str, object], reps: int) -> List[dict]:
    rows = []
    meta = summary["meta"]
    for variant in summary["point"]:
        point = summary["point"][variant]
        lower, upper = summary["interval"][variant]
        for i, iteration in enumerate(summary["iterations"]):
            rows.append(
                {
                    "figure": figure,
                    "config": summary["config"],
                    "environment": meta["environment"],
                    **extra,
                    "variant": variant,
                    "iteration": int(iteration),
                    "iqm": float(point[i]),
                    "ci_low": float(lower[i]),
                    "ci_high": float(upper[i]),
                    "reps": reps,
                }
            )
    return rows


def build_curves_csv(
    embedding_summaries: Dict[Tuple[str, int], dict],
    architecture_summaries: Dict[Tuple[str, int], dict],
    reps: int,
) -> pd.DataFrame:
    rows: List[dict] = []
    for (task, size), summary in embedding_summaries.items():
        rows += _summary_to_rows(summary, "fig_6_1_embedding", {"size": size}, reps)
    for (task, size), summary in architecture_summaries.items():
        rows += _summary_to_rows(summary, "fig_6_2_architecture", {"size": size}, reps)
    columns = [
        "figure",
        "config",
        "environment",
        "size",
        "variant",
        "iteration",
        "iqm",
        "ci_low",
        "ci_high",
        "reps",
    ]
    return pd.DataFrame(rows, columns=columns)


def build_seed_coverage(logs_dir: str = "logs", configs_dir: str = "training/configs") -> pd.DataFrame:
    """Per-(config,variant,run) completeness for every in-scope config (both Part A and Part B)."""
    frames = [ll.scan_config_completeness(config, logs_dir=logs_dir, configs_dir=configs_dir) for config in ALL_CONFIGS]
    return pd.concat(frames, ignore_index=True)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chapter 6 learning-curve figures and data.")
    parser.add_argument("--output-dir", default="results", help="Root output directory.")
    parser.add_argument("--logs-dir", default="logs", help="Root of the TensorBoard logs.")
    parser.add_argument("--configs-dir", default="training/configs", help="Experiment-config JSON directory.")
    parser.add_argument("--reps", type=int, default=DEFAULT_REPS, help="Bootstrap replications per iteration point.")
    parser.add_argument("--min-runs", type=int, default=2, help="Drop variants with fewer usable runs.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    plt.switch_backend("Agg")
    print(f"reps={args.reps}")

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"

    embedding_summaries: Dict[Tuple[str, int], dict] = {}
    for task in TASKS:
        for size in SIZES:
            config = embedding_config(task, size)
            print(f"Loading curves: {config}")
            embedding_summaries[(task, size)] = config_curve_summary(
                config,
                logs_dir=args.logs_dir,
                configs_dir=args.configs_dir,
                reps=args.reps,
                min_runs=args.min_runs,
            )

    architecture_summaries: Dict[Tuple[str, int], dict] = {}
    for task in ARCH_TASKS:
        for size in ARCH_SIZES:
            config = architecture_config(task, size)
            print(f"Loading curves: {config}")
            architecture_summaries[(task, size)] = config_curve_summary(
                config,
                logs_dir=args.logs_dir,
                configs_dir=args.configs_dir,
                reps=args.reps,
                min_runs=args.min_runs,
            )

    fig1_paths = plot_embedding_grid(embedding_summaries, figures_dir / "fig_6_1_embedding_learning_curves")
    fig2_paths: List[Path] = []
    for task in ARCH_TASKS:
        per_task_summaries = {size: architecture_summaries[(task, size)] for size in ARCH_SIZES}
        fig2_paths += plot_architecture_grid(
            per_task_summaries, figures_dir / f"fig_6_2_architecture_learning_curves_{task}"
        )

    curves_csv = build_curves_csv(embedding_summaries, architecture_summaries, args.reps)
    curves_path = output_dir / "learning_curves.csv"
    curves_csv.to_csv(curves_path, index=False)

    coverage = build_seed_coverage(logs_dir=args.logs_dir, configs_dir=args.configs_dir)
    coverage_path = output_dir / "seed_coverage.csv"
    coverage.to_csv(coverage_path, index=False)

    all_anomalies = [a for s in embedding_summaries.values() for a in s["anomalies"]]
    all_anomalies += [a for s in architecture_summaries.values() for a in s["anomalies"]]
    print(f"\n{len(all_anomalies)} anomalies found (see {coverage_path} for the full completeness table):")
    for anomaly in all_anomalies:
        print(f"  {anomaly.config_name}/{anomaly.variant}/{anomaly.run_label}: {anomaly.kind} - {anomaly.detail}")

    print(f"\nFigures: {', '.join(str(p) for p in fig1_paths + fig2_paths)}")
    print(f"Data: {curves_path}, {coverage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
