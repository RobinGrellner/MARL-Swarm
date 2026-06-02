"""Single-config rliable analysis CLI.

Analyzes one experiment config (one swarm size): each matrix variant (e.g.
``embed_dim16``) is a *method* and its repeated runs are aggregated with the
rliable protocol. Produces the thesis-ready outputs under
``<output-dir>/<config>/``:

* ``aggregate_summary.csv`` — IQM, Median, Mean, Optimality Gap with 95%
  stratified-bootstrap CIs per variant;
* ``summary.txt`` — the same numbers as a readable table (point [95% CI]);
* ``raw_scores.csv`` — the per-run scores behind the aggregates;
* ``probability_of_improvement.csv`` — each variant versus the reference variant;
* ``figures/aggregate_intervals.png`` and ``figures/performance_profiles.png``.

Example::

    python -m analysis.run_analysis --config embedding_scaling_rendezvous_100agents_ppo
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Optional, Sequence

from analysis import rliable_eval as rl
from analysis.log_loading import DEFAULT_SCORE_TAG, load_config_scores, read_config_meta


def _pick_reference(methods: Sequence[str], explicit: Optional[str]) -> str:
    """Choose the baseline variant for probability-of-improvement comparisons."""
    if explicit is not None:
        if explicit not in methods:
            raise ValueError(f"Reference '{explicit}' not among variants: {sorted(methods)}")
        return explicit
    if "embed_dim64" in methods:
        return "embed_dim64"
    embed_dims = [(int(m.group(1)), m.string) for m in (re.search(r"embed_dim(\d+)", x) for x in methods) if m]
    if embed_dims:
        return max(embed_dims)[1]
    return sorted(methods)[-1]


def analyze(
    config_name: str,
    *,
    output_dir: str = "results",
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
    """Run the full single-config analysis and write all artifacts.

    Returns a dict with the loaded scores, metadata, aggregate point/interval
    estimates, the summary and probability-of-improvement DataFrames, and the
    output paths (useful when called from a notebook).
    """
    meta = read_config_meta(config_name, configs_dir)
    algorithm = algorithm or str(meta["algorithm"]).upper()

    raw_scores, task_labels, meta = load_config_scores(
        config_name,
        logs_dir=logs_dir,
        configs_dir=configs_dir,
        algorithm=algorithm,
        score_tag=score_tag,
        reduction=reduction,
        last_k=last_k,
        min_runs=min_runs,
    )
    normalized = rl.normalize_scores(raw_scores, method=normalize, reference=reference)

    out_dir = Path(output_dir) / config_name
    figures_dir = out_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    point, interval = rl.aggregate_iqm(normalized, reps=reps, confidence_interval_size=confidence)
    summary = rl.aggregate_summary_frame(point, interval)
    summary.to_csv(out_dir / "aggregate_summary.csv", index=False)

    rl.raw_scores_frame(raw_scores, task_labels).to_csv(out_dir / "raw_scores.csv", index=False)

    poi = None
    reference_variant = _pick_reference(list(normalized), reference)
    pairs = [(method, reference_variant) for method in normalized if method != reference_variant]
    if pairs:
        poi_point, poi_interval = rl.probability_of_improvement(
            normalized, pairs, reps=reps, confidence_interval_size=confidence
        )
        poi = rl.probability_of_improvement_frame(poi_point, poi_interval)
        poi.to_csv(out_dir / "probability_of_improvement.csv", index=False)

    header = f"{config_name}  [{algorithm}, {meta['study']}, {meta['environment']}, size {meta['size']}]"
    (out_dir / "summary.txt").write_text(
        rl.summary_text(point, interval, header=header, probability_of_improvement=poi)
    )

    intervals_png = rl.plot_aggregate_intervals(
        point, interval, output_path=figures_dir / "aggregate_intervals.png"
    )
    taus = rl.default_taus(normalized)
    profiles, profile_cis = rl.performance_profile(normalized, taus, reps=reps, confidence_interval_size=confidence)
    profiles_png = rl.plot_profiles(profiles, profile_cis, taus, output_path=figures_dir / "performance_profiles.png")

    return {
        "config_name": config_name,
        "meta": meta,
        "algorithm": algorithm,
        "task_labels": task_labels,
        "raw_scores": raw_scores,
        "normalized_scores": normalized,
        "point_estimates": point,
        "interval_estimates": interval,
        "summary": summary,
        "probability_of_improvement": poi,
        "reference": reference_variant,
        "output_dir": out_dir,
        "figures": [intervals_png, profiles_png],
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="rliable analysis of a single experiment config.")
    parser.add_argument("--config", required=True, help="Config/log directory stem to analyze.")
    parser.add_argument("--output-dir", default="results", help="Root output directory.")
    parser.add_argument("--logs-dir", default="logs", help="Root of the TensorBoard logs.")
    parser.add_argument("--configs-dir", default="training/configs", help="Experiment-config JSON directory.")
    parser.add_argument(
        "--algorithm",
        default=None,
        choices=["PPO", "TRPO"],
        help="Run algorithm to read (default: the config's declared algorithm).",
    )
    parser.add_argument("--score-tag", default=DEFAULT_SCORE_TAG, help="Scalar tag used as the per-run score.")
    parser.add_argument(
        "--reduction",
        default="last_k_mean",
        choices=["last_k_mean", "last", "best"],
        help="Reduction of the scalar series into a per-run score.",
    )
    parser.add_argument("--last-k", type=int, default=10, help="Window for reduction=last_k_mean.")
    parser.add_argument("--min-runs", type=int, default=2, help="Drop variants with fewer usable runs.")
    parser.add_argument(
        "--normalize",
        default="min_max",
        choices=["min_max", "reference", "none"],
        help="Per-task score normalization mode.",
    )
    parser.add_argument(
        "--reference",
        default=None,
        help="Reference variant for normalization and improvement comparisons (default: embed_dim64).",
    )
    parser.add_argument("--reps", type=int, default=rl.DEFAULT_REPS, help="Bootstrap replications.")
    parser.add_argument("--confidence", type=float, default=rl.DEFAULT_CONFIDENCE, help="CI coverage.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point for the single-config analysis."""
    args = _build_arg_parser().parse_args(argv)
    rl.plt.switch_backend("Agg")

    result = analyze(
        args.config,
        output_dir=args.output_dir,
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

    meta = result["meta"]
    print(
        f"Analyzed '{args.config}' [{result['algorithm']}, {meta['study']}, {meta['environment']}, "
        f"size {meta['size']}, {meta['n_runs']} runs/variant]"
    )
    print(f"Reference variant: {result['reference']}")
    print(f"\nImportant variables (point [95% CI]) written to {result['output_dir']}:\n")
    print(rl.summary_pivot(result["point_estimates"], result["interval_estimates"]).to_string())
    print(f"\nFigures: {', '.join(str(path) for path in result['figures'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
