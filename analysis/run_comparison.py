"""Cross-config rliable comparison CLI.

Pools several configs of one family (typically the same study/environment across
swarm sizes) onto a single task axis: each matrix variant is a *method* and each
config's swarm size is a *task*. Produces, under ``<output-dir>/<name>/``:

* ``aggregate_summary.csv`` — IQM/Median/Mean/Optimality Gap with 95% CIs per
  variant, pooled across sizes;
* ``summary.txt`` — the same numbers as a readable table (point [95% CI]);
* ``iqm_by_size.csv`` and ``figures/iqm_by_size.png`` — per-size IQM (scale curve);
* ``raw_scores.csv``, ``probability_of_improvement.csv``;
* ``figures/aggregate_intervals.png`` and ``figures/performance_profiles.png``.

Example::

    python -m analysis.run_comparison --configs \\
        embedding_scaling_rendezvous_4agents_ppo \\
        embedding_scaling_rendezvous_50agents_ppo \\
        embedding_scaling_rendezvous_100agents_ppo
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Sequence

import pandas as pd

from analysis import rliable_eval as rl
from analysis.log_loading import DEFAULT_SCORE_TAG, load_comparison_scores, read_config_meta
from analysis.run_analysis import _pick_reference


def _iqm_by_task_frame(
    point_estimates: Dict[str, "rl.np.ndarray"],
    interval_estimates: Dict[str, "rl.np.ndarray"],
    task_labels: Sequence[str],
) -> pd.DataFrame:
    """Tidy ``config x task`` table of per-task IQM and CI bounds."""
    rows = []
    for config, point in point_estimates.items():
        for index, task in enumerate(task_labels):
            rows.append(
                {
                    "config": config,
                    "task": task,
                    "iqm": float(point[index]),
                    "ci_low": float(interval_estimates[config][0, index]),
                    "ci_high": float(interval_estimates[config][1, index]),
                }
            )
    return pd.DataFrame(rows, columns=["config", "task", "iqm", "ci_low", "ci_high"])


def compare(
    config_names: Sequence[str],
    *,
    output_dir: str = "results",
    name: Optional[str] = None,
    logs_dir: str = "logs",
    configs_dir: str = "training/configs",
    algorithm: Optional[str] = None,
    score_tag: str = DEFAULT_SCORE_TAG,
    reduction: str = "last_k_mean",
    last_k: int = 10,
    min_runs: int = 2,
    normalize: str = "min_max",
    reference: Optional[str] = None,
    reps: int = rl.DEFAULT_REPS,
    confidence: float = rl.DEFAULT_CONFIDENCE,
) -> Dict[str, object]:
    """Run the cross-config (scale) comparison and write all artifacts."""
    algorithm = algorithm or str(read_config_meta(config_names[0], configs_dir)["algorithm"]).upper()

    raw_scores, task_labels, metas = load_comparison_scores(
        config_names,
        logs_dir=logs_dir,
        configs_dir=configs_dir,
        algorithm=algorithm,
        score_tag=score_tag,
        reduction=reduction,
        last_k=last_k,
        min_runs=min_runs,
    )
    normalized = rl.normalize_scores(raw_scores, method=normalize, reference=reference)

    head = metas[0]
    if name is None:
        name = f"comparison_{head['study']}_{head['environment']}_{algorithm.lower()}"
    out_dir = Path(output_dir) / name
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    point, interval = rl.aggregate_iqm(normalized, reps=reps, confidence_interval_size=confidence)
    summary = rl.aggregate_summary_frame(point, interval)
    summary.to_csv(out_dir / "aggregate_summary.csv", index=False)
    rl.raw_scores_frame(raw_scores, task_labels).to_csv(out_dir / "raw_scores.csv", index=False)

    task_point, task_interval = rl.iqm_by_task(normalized, reps=reps, confidence_interval_size=confidence)
    iqm_frame = _iqm_by_task_frame(task_point, task_interval, task_labels)
    iqm_frame.to_csv(out_dir / "iqm_by_size.csv", index=False)

    reference_variant = _pick_reference(list(normalized), reference)
    pairs = [(method, reference_variant) for method in normalized if method != reference_variant]
    poi = None
    if pairs:
        poi_point, poi_interval = rl.probability_of_improvement(
            normalized, pairs, reps=reps, confidence_interval_size=confidence
        )
        poi = rl.probability_of_improvement_frame(poi_point, poi_interval)
        poi.to_csv(out_dir / "probability_of_improvement.csv", index=False)

    header = f"{name}  [{algorithm}, {head['study']}, {head['environment']}, sizes {task_labels}]"
    (out_dir / "summary.txt").write_text(
        rl.summary_text(point, interval, header=header, probability_of_improvement=poi)
    )

    intervals_png = rl.plot_aggregate_intervals(point, interval, output_path=figures_dir / "aggregate_intervals.png")
    scale_png = rl.plot_iqm_by_task(
        task_point, task_interval, task_labels, output_path=figures_dir / "iqm_by_size.png"
    )
    taus = rl.default_taus(normalized)
    profiles, profile_cis = rl.performance_profile(normalized, taus, reps=reps, confidence_interval_size=confidence)
    profiles_png = rl.plot_profiles(profiles, profile_cis, taus, output_path=figures_dir / "performance_profiles.png")

    return {
        "config_names": list(config_names),
        "metas": metas,
        "algorithm": algorithm,
        "task_labels": task_labels,
        "raw_scores": raw_scores,
        "normalized_scores": normalized,
        "point_estimates": point,
        "interval_estimates": interval,
        "summary": summary,
        "iqm_by_size": iqm_frame,
        "probability_of_improvement": poi,
        "reference": reference_variant,
        "output_dir": out_dir,
        "figures": [intervals_png, scale_png, profiles_png],
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="rliable comparison across configs (e.g. swarm sizes).")
    parser.add_argument("--configs", nargs="+", required=True, help="Config stems to compare (>= 2).")
    parser.add_argument("--output-dir", default="results", help="Root output directory.")
    parser.add_argument("--name", default=None, help="Output subdirectory name (default: derived from metadata).")
    parser.add_argument("--logs-dir", default="logs", help="Root of the TensorBoard logs.")
    parser.add_argument("--configs-dir", default="training/configs", help="Experiment-config JSON directory.")
    parser.add_argument("--algorithm", default=None, choices=["PPO", "TRPO"], help="Run algorithm to read.")
    parser.add_argument("--score-tag", default=DEFAULT_SCORE_TAG, help="Scalar tag used as the per-run score.")
    parser.add_argument(
        "--reduction",
        default="last_k_mean",
        choices=["last_k_mean", "last", "best"],
        help="Reduction of the scalar series into a per-run score.",
    )
    parser.add_argument("--last-k", type=int, default=10, help="Window for reduction=last_k_mean.")
    parser.add_argument("--min-runs", type=int, default=2, help="Drop variants with fewer usable runs per config.")
    parser.add_argument(
        "--normalize",
        default="min_max",
        choices=["min_max", "reference", "none"],
        help="Per-task score normalization mode.",
    )
    parser.add_argument("--reference", default=None, help="Reference variant (default: embed_dim64).")
    parser.add_argument("--reps", type=int, default=rl.DEFAULT_REPS, help="Bootstrap replications.")
    parser.add_argument("--confidence", type=float, default=rl.DEFAULT_CONFIDENCE, help="CI coverage.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the cross-config comparison."""
    args = _build_arg_parser().parse_args(argv)
    rl.plt.switch_backend("Agg")

    result = compare(
        args.configs,
        output_dir=args.output_dir,
        name=args.name,
        logs_dir=args.logs_dir,
        configs_dir=args.configs_dir,
        algorithm=args.algorithm,
        score_tag=args.score_tag,
        reduction=args.reduction,
        last_k=args.last_k,
        min_runs=args.min_runs,
        normalize=args.normalize,
        reference=args.reference,
        reps=args.reps,
        confidence=args.confidence,
    )

    print(
        f"Compared {len(result['config_names'])} configs across sizes {result['task_labels']} "
        f"[{result['algorithm']}]"
    )
    print(f"Shared variants: {', '.join(result['point_estimates'])}")
    print(f"\nPooled aggregate (point [95% CI]) written to {result['output_dir']}:\n")
    print(rl.summary_pivot(result["point_estimates"], result["interval_estimates"]).to_string())
    print(f"\nFigures: {', '.join(str(path) for path in result['figures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
