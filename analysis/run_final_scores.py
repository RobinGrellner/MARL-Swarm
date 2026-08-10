"""Chapter 6 final-score numbers: seed completeness, per-config IQM/CI, and figures.

Runs the existing rliable final-score pipeline (:func:`analysis.run_analysis.analyze`)
over every embedding_scaling and architecture_scalability config (both tasks, all N),
with truncated runs excluded up front (see :mod:`analysis.log_loading`). Writes
``final_scores_summary.json`` and figures under ``results/figures/``.

Example::

    python -m analysis.run_final_scores
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import pandas as pd

from analysis import log_loading as ll
from analysis import rliable_eval as rl
from analysis.run_analysis import analyze
from analysis.run_learning_curves import (
    ALL_CONFIGS,
    ARCH_SIZES,
    ARCH_TASKS,
    DEPTHS,
    EMBED_DIMS,
    SIZES,
    TASKS,
    WIDTH_COLOR,
    WIDTHS,
    architecture_config,
    embedding_config,
    rc_context_kwargs,
    save_figure_png_pdf,
)

REFERENCE_EMBED_DIM = 64
TASK_COLOR = {"rendezvous": "#E69F00", "pursuit_evasion": "#0072B2"}


def seed_completeness_matrix(
    logs_dir: str = "logs",
    configs_dir: str = "training/configs",
    configs: Sequence[str] = ALL_CONFIGS,
) -> pd.DataFrame:
    """Per-(config,variant) usable-run count out of the runs found, for B1."""
    frames = [ll.scan_config_completeness(config, logs_dir=logs_dir, configs_dir=configs_dir) for config in configs]
    coverage = pd.concat(frames, ignore_index=True)
    grouped = coverage.groupby(["config", "environment", "size", "variant"], as_index=False).agg(
        n_usable=("usable", "sum"), n_total=("usable", "size")
    )
    grouped["flagged"] = grouped["n_usable"] < 5
    return grouped


def run_all_analyses(
    *,
    logs_dir: str,
    configs_dir: str,
    output_dir: str,
    reps: int,
    min_runs: int,
    configs: Sequence[str] = ALL_CONFIGS,
) -> Dict[str, dict]:
    figures_dir = Path(output_dir) / "figures" / "configs"
    results = {}
    for config in configs:
        print(f"Analyzing: {config}")
        results[config] = analyze(
            config,
            output_dir=str(figures_dir),
            logs_dir=logs_dir,
            configs_dir=configs_dir,
            reps=reps,
            min_runs=min_runs,
            require_complete=True,
        )
    return results


def embedding_iqm_table(results_by_config: Dict[str, dict]) -> List[dict]:
    """B3(a)+(c): per (task,N,embed_dim) IQM, CI bounds and CI width."""
    iqm_index = rl.AGGREGATE_METRIC_NAMES.index("IQM")
    rows = []
    for task in TASKS:
        for size in SIZES:
            result = results_by_config[embedding_config(task, size)]
            for k in EMBED_DIMS:
                variant = f"embed_dim{k}"
                if variant not in result["point_estimates"]:
                    continue
                point = float(result["point_estimates"][variant][iqm_index])
                lower, upper = result["interval_estimates"][variant][:, iqm_index]
                rows.append(
                    {
                        "task": task,
                        "size": size,
                        "embed_dim": k,
                        "iqm": point,
                        "ci_low": float(lower),
                        "ci_high": float(upper),
                        "ci_width": float(upper - lower),
                    }
                )
    return rows


def smallest_overlapping_k(embedding_rows: List[dict]) -> List[dict]:
    """B3(b): per (task,N), smallest embed_dim whose CI overlaps embed_dim=64's, with per-k decisions."""
    by_cell: Dict[tuple, Dict[int, dict]] = {}
    for row in embedding_rows:
        by_cell.setdefault((row["task"], row["size"]), {})[row["embed_dim"]] = row

    results = []
    for (task, size), by_k in sorted(by_cell.items()):
        if REFERENCE_EMBED_DIM not in by_k:
            continue
        reference = by_k[REFERENCE_EMBED_DIM]
        decisions = {}
        smallest = None
        for k in sorted(by_k):
            row = by_k[k]
            overlaps = row["ci_low"] <= reference["ci_high"] and reference["ci_low"] <= row["ci_high"]
            decisions[k] = overlaps
            if overlaps and smallest is None:
                smallest = k
        results.append(
            {
                "task": task,
                "size": size,
                "reference_embed_dim": REFERENCE_EMBED_DIM,
                "smallest_overlapping_k": smallest,
                "per_k_overlap": decisions,
            }
        )
    return results


def architecture_iqm_table(results_by_config: Dict[str, dict]) -> List[dict]:
    """B3(d): per (task,N,L,w) IQM and CI bounds."""
    iqm_index = rl.AGGREGATE_METRIC_NAMES.index("IQM")
    rows = []
    for task in ARCH_TASKS:
        for size in ARCH_SIZES:
            result = results_by_config[architecture_config(task, size)]
            for depth in DEPTHS:
                for width in WIDTHS:
                    variant = f"phi_layers{depth}_phi_hidden_width{width}"
                    if variant not in result["point_estimates"]:
                        continue
                    point = float(result["point_estimates"][variant][iqm_index])
                    lower, upper = result["interval_estimates"][variant][:, iqm_index]
                    rows.append(
                        {
                            "task": task,
                            "size": size,
                            "depth": depth,
                            "width": width,
                            "iqm": point,
                            "ci_low": float(lower),
                            "ci_high": float(upper),
                        }
                    )
    return rows


def total_timesteps_table(configs_dir: str = "training/configs", configs: Sequence[str] = ALL_CONFIGS) -> List[dict]:
    """B3(e): T_total = n_iterations * n_steps * n_agents * num_vec_envs, per config."""
    rows = []
    for config in configs:
        spec = json.loads((Path(configs_dir) / f"{config}.json").read_text())
        train_config = spec["defaults"]["train_config"]
        env_config = spec["defaults"]["env_config"]
        n_agents = env_config.get("num_agents", env_config.get("num_pursuers"))
        n_iterations = train_config["n_iterations"]
        n_steps = train_config["n_steps"]
        num_vec_envs = train_config["num_vec_envs"]
        rows.append(
            {
                "config": config,
                "n_iterations": n_iterations,
                "n_steps": n_steps,
                "n_agents": n_agents,
                "num_vec_envs": num_vec_envs,
                "total_timesteps": n_iterations * n_steps * n_agents * num_vec_envs,
            }
        )
    return rows


def plot_embedding_final_scores(embedding_rows: List[dict], output_prefix: Path) -> List[Path]:
    with plt.rc_context(rc_context_kwargs()):
        fig, axes = plt.subplots(1, len(SIZES), figsize=(9.0, 2.8), sharey=True)
        for col, size in enumerate(SIZES):
            ax = axes[col]
            for task in TASKS:
                rows = sorted(
                    (r for r in embedding_rows if r["task"] == task and r["size"] == size),
                    key=lambda r: r["embed_dim"],
                )
                if not rows:
                    continue
                xs = [r["embed_dim"] for r in rows]
                ys = [r["iqm"] for r in rows]
                lo = [r["ci_low"] for r in rows]
                hi = [r["ci_high"] for r in rows]
                color = TASK_COLOR[task]
                ax.plot(xs, ys, marker="o", color=color, linewidth=1.0, label=task)
                ax.fill_between(xs, lo, hi, color=color, alpha=0.15, linewidth=0)
            ax.set_xscale("log", base=2)
            ax.set_xticks(EMBED_DIMS)
            ax.set_xticklabels([str(k) for k in EMBED_DIMS])
            ax.set_title(f"N={size}")
            ax.set_xlabel("embed_dim")
            if col == 0:
                ax.set_ylabel("IQM normalized return")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.08))
        fig.tight_layout(rect=(0, 0.14, 1, 1))
        return save_figure_png_pdf(fig, output_prefix)


def plot_architecture_final_scores(architecture_rows: List[dict], output_prefix: Path) -> List[Path]:
    """2 rows (task) x len(ARCH_SIZES) cols (N), 3 lines (width w) per panel."""
    with plt.rc_context(rc_context_kwargs()):
        fig, axes = plt.subplots(
            len(ARCH_TASKS), len(ARCH_SIZES), figsize=(2.4 * len(ARCH_SIZES), 2.8 * len(ARCH_TASKS)), sharey=True
        )
        for row, task in enumerate(ARCH_TASKS):
            for col, size in enumerate(ARCH_SIZES):
                ax = axes[row, col]
                for width in WIDTHS:
                    rows = sorted(
                        (r for r in architecture_rows if r["task"] == task and r["size"] == size and r["width"] == width),
                        key=lambda r: r["depth"],
                    )
                    if not rows:
                        continue
                    xs = [r["depth"] for r in rows]
                    ys = [r["iqm"] for r in rows]
                    lo_err = [y - r["ci_low"] for y, r in zip(ys, rows)]
                    hi_err = [r["ci_high"] - y for y, r in zip(ys, rows)]
                    color = WIDTH_COLOR[width]
                    ax.errorbar(
                        xs,
                        ys,
                        yerr=[lo_err, hi_err],
                        marker="o",
                        color=color,
                        linewidth=1.0,
                        capsize=2,
                        label=f"w={width}",
                    )
                ax.set_xticks(DEPTHS)
                ax.set_title(f"{task}, N={size}")
                if row == len(ARCH_TASKS) - 1:
                    ax.set_xlabel("phi_layers (depth)")
                if col == 0:
                    ax.set_ylabel("IQM normalized return")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=len(WIDTHS), frameon=False, bbox_to_anchor=(0.5, -0.05))
        fig.tight_layout(rect=(0, 0.1, 1, 1))
        return save_figure_png_pdf(fig, output_prefix)


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chapter 6 final-score numbers and figures.")
    parser.add_argument("--output-dir", default="results", help="Root output directory.")
    parser.add_argument("--logs-dir", default="logs", help="Root of the TensorBoard logs.")
    parser.add_argument("--configs-dir", default="training/configs", help="Experiment-config JSON directory.")
    parser.add_argument("--reps", type=int, default=rl.DEFAULT_REPS, help="Bootstrap replications.")
    parser.add_argument("--min-runs", type=int, default=2, help="Drop variants with fewer usable runs.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    plt.switch_backend("Agg")
    print(f"reps={args.reps}")

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"

    print("Scanning seed completeness...")
    completeness = seed_completeness_matrix(logs_dir=args.logs_dir, configs_dir=args.configs_dir)
    flagged = completeness[completeness["flagged"]]
    print(f"{len(flagged)} (config,variant) cells with < 5 usable runs:")
    flagged_records = []
    for row in flagged.itertuples():
        print(f"  {row.config}/{row.variant}: {row.n_usable}/{row.n_total} usable")
        flagged_records.append(
            {
                "config": str(row.config),
                "environment": str(row.environment),
                "size": int(row.size) if row.size is not None else None,
                "variant": str(row.variant),
                "n_usable": int(row.n_usable),
                "n_total": int(row.n_total),
            }
        )

    results_by_config = run_all_analyses(
        logs_dir=args.logs_dir,
        configs_dir=args.configs_dir,
        output_dir=args.output_dir,
        reps=args.reps,
        min_runs=args.min_runs,
    )

    embedding_rows = embedding_iqm_table(results_by_config)
    overlap_rows = smallest_overlapping_k(embedding_rows)
    architecture_rows = architecture_iqm_table(results_by_config)
    timesteps_rows = total_timesteps_table(configs_dir=args.configs_dir)

    fig_embed_paths = plot_embedding_final_scores(embedding_rows, figures_dir / "fig_embedding_final_scores")
    fig_arch_paths = plot_architecture_final_scores(architecture_rows, figures_dir / "fig_architecture_final_scores")

    numbers = {
        "reps": args.reps,
        "embedding_final_scores": embedding_rows,
        "embedding_smallest_overlapping_k": overlap_rows,
        "architecture_final_scores": architecture_rows,
        "total_timesteps": timesteps_rows,
        "seed_completeness_flags": flagged_records,
    }
    numbers_path = output_dir / "final_scores_summary.json"
    numbers_path.write_text(json.dumps(numbers, indent=2))

    print(f"\nWrote {numbers_path}")
    print(f"Figures: {', '.join(str(p) for p in fig_embed_paths + fig_arch_paths)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
