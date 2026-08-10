"""Zero-shot cross-size generalization CLI.

Evaluates mean-embedding policies trained at one swarm size on a grid of *test*
swarm sizes (no further training) and reports how performance transfers. Per
config it writes ``<output-dir>/generalization_<config>/``:

* ``raw_episodes.csv``   -- per-episode rollout records (the eval cache);
* ``aggregated.csv``     -- mean metrics per (variant, run, test_size) cell;
* ``summary.txt``        -- readable per-(variant, test_size) reward + convergence;
* ``figures/transfer_reward.png``      -- View A: IQM episode reward vs test size,
                                          one line per variant (CI over runs);
* ``figures/transfer_reward_norm.png`` -- View A: same on min-max normalized
                                          reward (Gl. 57);
* ``figures/transfer_convergence.png`` -- View A: scale-free convergence rate vs
                                          test size, one line per variant
                                          (pursuit-evasion instead writes
                                          ``transfer_capture_rate.png`` /
                                          ``transfer_capture_time.png``).

Both rendezvous and pursuit-evasion configs are supported; the task is read
from the config's ``env_config.environment``. The variant axis (the rliable
"method" dimension) is read from the config's ``matrix_parameters``: either
``embed_dim`` or a ``phi_layers``/``phi_hidden_width`` grid.

When two or more configs are passed (ideally the full train-size set) it also
writes a cross-config cube under ``<output-dir>/generalization_cube/``:

* View B -- per variant, a train_size x test_size IQM-reward heatmap;
* View C -- per variant, a native-retention heatmap (transferred reward / the
            reward of the policy native to each test size) plus a gap summary;
* View D -- per variant, the additive generalization gap Delta(N_test) =
            G(N_train) - G(N_test) on Gl.-57 normalized returns with IQM +
            bootstrap CIs (``delta_iqm.csv`` + ``viewD_delta_*.png``).

Example::

    python -m analysis.run_generalization --configs embedding_scaling_rendezvous_16agents_ppo
    python -m analysis.run_generalization --configs \
        embedding_scaling_rendezvous_{4,16,50,100}agents_ppo --eval-sizes 4 8 16 32 50 100
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from analysis import rliable_eval as rl
from analysis.generalization_loading import aggregate, run_or_load_raw, to_score_dict
from analysis.generalization_resolver import ConfigSpec, load_config_spec, variant_sort_key

DEFAULT_EVAL_SIZES = (4, 8, 16, 32, 50, 100)
DEFAULT_EVAL_SEEDS = (0, 1, 2)


def analyze_generalization(
    config: str,
    *,
    eval_sizes: Sequence[int] = DEFAULT_EVAL_SIZES,
    n_episodes: int = 20,
    eval_seeds: Sequence[int] = DEFAULT_EVAL_SEEDS,
    model_root: str = "model",
    output_dir: str = "results",
    configs_dir: str = "training/configs",
    device: str = "cpu",
    reps: int = rl.DEFAULT_REPS,
    confidence: float = rl.DEFAULT_CONFIDENCE,
    force: bool = False,
    workers: int = 4,
    position: Optional[tuple] = None,
) -> Dict[str, object]:
    """Run the per-config transfer analysis (View A) and write its artifacts."""
    spec = load_config_spec(config, configs_dir)
    # A policy can only be tested up to the agent count baked into its obs
    # (max_agents / max_pursuers).
    max_agents = spec.max_size
    sizes = [s for s in eval_sizes if s <= max_agents]
    dropped = [s for s in eval_sizes if s > max_agents]

    tag = f"[{position[0]}/{position[1]}] " if position else ""
    print(f"\n{'=' * 70}\n{tag}{spec.stem} (train size {spec.train_size})\n{'=' * 70}", flush=True)

    out_dir = Path(output_dir) / f"generalization_{spec.stem}"
    figures_dir = out_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    df = run_or_load_raw(
        spec,
        test_sizes=sizes,
        n_episodes=n_episodes,
        eval_seeds=eval_seeds,
        cache_path=out_dir / "raw_episodes.csv",
        model_root=model_root,
        device=device,
        force=force,
        workers=workers,
    )
    agg = aggregate(df)
    agg.to_csv(out_dir / "aggregated.csv", index=False)

    labels = [str(s) for s in sizes]
    figures: List[Path] = []

    # View A.1 -- IQM episode reward vs test size (raw; reward scale differs by
    # size, so read the convergence curve alongside it).
    reward_scores, _ = to_score_dict(agg, metric="ep_reward")
    r_point, r_interval = rl.iqm_by_task(reward_scores, reps=reps, confidence_interval_size=confidence)
    figures.append(
        rl.plot_iqm_by_task(
            r_point, r_interval, labels,
            output_path=figures_dir / "transfer_reward.png",
            xlabel="Test swarm size",
            ylabel="IQM episode reward",
        )
    )

    # View A.1b -- Gl.-57 min-max normalized reward (each test-size column
    # normalized over its observed range, so sizes share one scale).
    norm_scores = rl.normalize_scores(reward_scores, method="min_max")
    n_point, n_interval = rl.iqm_by_task(norm_scores, reps=reps, confidence_interval_size=confidence)
    figures.append(
        rl.plot_iqm_by_task(
            n_point, n_interval, labels,
            output_path=figures_dir / "transfer_reward_norm.png",
            xlabel="Test swarm size",
            ylabel="IQM min-max normalized reward",
        )
    )

    # View A.2 -- early-termination rate vs test size (scale-free transfer
    # signal): rendezvous convergence, pursuit-evasion capture rate (Fangrate).
    is_pursuit = spec.task == "pursuit_evasion"
    conv_scores, _ = to_score_dict(agg, metric="converged")
    c_point, c_interval = rl.iqm_by_task(conv_scores, reps=reps, confidence_interval_size=confidence)
    figures.append(
        rl.plot_iqm_by_task(
            c_point, c_interval, labels,
            output_path=figures_dir / ("transfer_capture_rate.png" if is_pursuit else "transfer_convergence.png"),
            xlabel="Test swarm size",
            ylabel="Capture rate (Fangrate)" if is_pursuit else "Convergence rate",
        )
    )

    # View A.3 (pursuit-evasion only) -- capture time (Fangzeit) vs test size;
    # cells where no episode captured stay NaN and leave gaps in the plot.
    t_point = None
    if is_pursuit:
        time_scores, _ = to_score_dict(agg, metric="capture_time")
        t_point, t_interval = rl.iqm_by_task(time_scores, reps=reps, confidence_interval_size=confidence)
        figures.append(
            rl.plot_iqm_by_task(
                t_point, t_interval, labels,
                output_path=figures_dir / "transfer_capture_time.png",
                xlabel="Test swarm size",
                ylabel="Capture time (Fangzeit, steps)",
            )
        )

    (out_dir / "summary.txt").write_text(
        _summary_text(spec, sizes, r_point, c_point, dropped, time_point=t_point)
    )

    return {
        "spec": spec,
        "test_sizes": sizes,
        "aggregated": agg,
        "reward_point": r_point,
        "convergence_point": c_point,
        "output_dir": out_dir,
        "figures": figures,
    }


def _summary_text(
    spec: ConfigSpec,
    sizes: Sequence[int],
    reward_point: Dict[str, np.ndarray],
    conv_point: Dict[str, np.ndarray],
    dropped: Sequence[int],
    *,
    time_point: Optional[Dict[str, np.ndarray]] = None,
) -> str:
    conv_tag = "[fang%]" if spec.task == "pursuit_evasion" else "[conv%]"
    lines = [
        f"Zero-shot cross-size generalization: {spec.stem}",
        f"Trained at swarm size {spec.train_size}; tested at {[int(s) for s in sizes]}.",
        "Score = IQM over the 5 PPO runs (task = test swarm size).",
        "",
    ]
    if dropped:
        lines.append(f"NOTE: dropped test sizes above the obs-space cap: {[int(s) for s in dropped]}")
        lines.append("")
    header = " " * 20 + "".join(f"{int(s):>10}" for s in sizes)
    lines.append(header)
    for variant in sorted(reward_point, key=variant_sort_key):
        rew = "".join(f"{v:>10.2f}" for v in reward_point[variant])
        lines.append(f"[reward] {variant:<11}{rew}")
    lines.append("")
    lines.append(header)
    for variant in sorted(conv_point, key=variant_sort_key):
        cov = "".join(f"{v:>10.2f}" for v in conv_point[variant])
        lines.append(f"{conv_tag}  {variant:<11}{cov}")
    if time_point is not None:
        lines.append("")
        lines.append(header)
        for variant in sorted(time_point, key=variant_sort_key):
            tim = "".join(f"{v:>10.1f}" for v in time_point[variant])
            lines.append(f"[fangT]  {variant:<11}{tim}")
    return "\n".join(lines) + "\n"


def analyze_cube(
    configs: Sequence[str],
    *,
    output_dir: str = "results",
    reps: int = rl.DEFAULT_REPS,
    confidence: float = rl.DEFAULT_CONFIDENCE,
    **per_config_kwargs,
) -> Dict[str, object]:
    """Run every config (View A) and build the cross-config cube (Views B & C)."""
    per_config = {}
    frames = []
    n = len(configs)
    for i, config in enumerate(configs, 1):
        result = analyze_generalization(
            config, output_dir=output_dir, reps=reps, confidence=confidence,
            position=(i, n), **per_config_kwargs,
        )
        per_config[config] = result
        agg = result["aggregated"].copy()
        agg["train_size"] = result["spec"].train_size
        frames.append(agg)

    tasks = {result["spec"].task for result in per_config.values()}
    if len(tasks) > 1:
        raise ValueError(
            f"Refusing to build one cube from mixed tasks {sorted(tasks)}: "
            "ep_reward is not comparable across tasks. Pass --configs from a single task."
        )

    print(f"\n{'=' * 70}\nBuilding cross-config cube (Views B/C)\n{'=' * 70}", flush=True)
    cube = pd.concat(frames, ignore_index=True)
    cube_dir = Path(output_dir) / "generalization_cube"
    figures_dir = cube_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    cube.to_csv(cube_dir / "cube.csv", index=False)

    figures = _plot_cube_views(cube, figures_dir)
    delta_df, delta_figures = _delta_analysis(
        cube, cube_dir, figures_dir, reps=reps, confidence=confidence
    )
    figures.extend(delta_figures)
    (cube_dir / "summary.txt").write_text(_cube_summary_text(cube))

    return {
        "cube": cube, "per_config": per_config, "output_dir": cube_dir,
        "figures": figures, "delta": delta_df,
    }


def _iqm_grid(cube: pd.DataFrame, variant: str, metric: str = "ep_reward"):
    """Return (train_sizes, test_sizes, matrix) of mean ``metric`` for one variant.

    Matrix entry [i, j] aggregates over runs (mean) the cell (train_sizes[i],
    test_sizes[j]); NaN where that train/test combination was not evaluated.
    """
    sub = cube[cube["variant"] == variant]
    train_sizes = sorted(sub["train_size"].unique())
    test_sizes = sorted(sub["test_size"].unique())
    mat = np.full((len(train_sizes), len(test_sizes)), np.nan)
    for i, tr in enumerate(train_sizes):
        for j, te in enumerate(test_sizes):
            vals = sub[(sub["train_size"] == tr) & (sub["test_size"] == te)][metric]
            if len(vals):
                mat[i, j] = float(vals.mean())
    return train_sizes, test_sizes, mat


def _plot_cube_views(cube: pd.DataFrame, figures_dir: Path) -> List[Path]:
    rl.plt.switch_backend("Agg")
    figures: List[Path] = []
    for variant in sorted(cube["variant"].unique(), key=variant_sort_key):
        train_sizes, test_sizes, mat = _iqm_grid(cube, variant, "ep_reward")

        # View B -- raw reward heatmap (train x test).
        figures.append(
            _heatmap(
                mat, train_sizes, test_sizes,
                title=f"View B: mean episode reward -- {variant}",
                output_path=figures_dir / f"viewB_reward_{variant}.png",
                fmt="{:.1f}",
            )
        )

        # View C -- native retention: reward / reward of the policy native to
        # each test size (the diagonal-trained policy at that size).
        retention = np.full_like(mat, np.nan)
        for j, te in enumerate(test_sizes):
            if te in train_sizes:
                native = mat[train_sizes.index(te), j]
                if np.isfinite(native) and native != 0:
                    retention[:, j] = mat[:, j] / native
        figures.append(
            _heatmap(
                retention, train_sizes, test_sizes,
                title=f"View C: native retention -- {variant}",
                output_path=figures_dir / f"viewC_retention_{variant}.png",
                fmt="{:.2f}",
            )
        )
    return figures


def _delta_analysis(
    cube: pd.DataFrame,
    cube_dir: Path,
    figures_dir: Path,
    *,
    reps: int,
    confidence: float,
):
    """View D -- additive generalization gap on Gl.-57 normalized returns.

    Per run, Delta(N_test) = G_norm(N_train) - G_norm(N_test) (Gl. 56), where
    G_norm min-max normalizes each test-size column over every (variant,
    train_size, run) observation in that cell (Gl. 57, same scheme as
    ``rliable_eval.normalize_scores``). Aggregated per (variant, train_size,
    test_size) with IQM + stratified bootstrap CIs over runs.
    """
    # Test sizes evaluated for every train size, so the score matrices align.
    common = None
    for _, sub in cube.groupby("train_size"):
        sizes = {int(s) for s in sub["test_size"].unique()}
        common = sizes if common is None else common & sizes
    test_sizes = sorted(common)

    scores: Dict[str, np.ndarray] = {}
    train_of: Dict[str, int] = {}
    for (variant, train_size), sub in cube.groupby(["variant", "train_size"]):
        key = f"{variant}@train{int(train_size)}"
        mat = (
            sub.pivot(index="run", columns="test_size", values="ep_reward")
            .reindex(columns=test_sizes)
            .sort_index()
        )
        scores[key] = mat.to_numpy(dtype=float)
        train_of[key] = int(train_size)

    normalized = rl.normalize_scores(scores, method="min_max")

    deltas: Dict[str, np.ndarray] = {}
    for key, mat in normalized.items():
        if train_of[key] not in test_sizes:
            print(f"  delta: skipping {key} (train size not among common test sizes)", flush=True)
            continue
        j = test_sizes.index(train_of[key])
        deltas[key] = mat[:, [j]] - mat

    point, interval = rl.iqm_by_task(deltas, reps=reps, confidence_interval_size=confidence)

    rows = []
    for key in deltas:
        variant = key.split("@train")[0]
        for j, te in enumerate(test_sizes):
            rows.append(
                {
                    "variant": variant,
                    "train_size": train_of[key],
                    "test_size": te,
                    "delta_iqm": point[key][j],
                    "ci_low": interval[key][0, j],
                    "ci_high": interval[key][1, j],
                }
            )
    delta_df = pd.DataFrame(rows)
    delta_df.to_csv(cube_dir / "delta_iqm.csv", index=False)

    figures: List[Path] = []
    labels = [str(s) for s in test_sizes]
    for variant in sorted({k.split("@train")[0] for k in deltas}, key=variant_sort_key):
        keys = [k for k in deltas if k.split("@train")[0] == variant]
        v_point = {f"train{train_of[k]}": point[k] for k in keys}
        v_interval = {f"train{train_of[k]}": interval[k] for k in keys}
        figures.append(
            rl.plot_iqm_by_task(
                v_point, v_interval, labels,
                output_path=figures_dir / f"viewD_delta_{variant}.png",
                xlabel="Test swarm size",
                ylabel="IQM generalization gap (normalized reward)",
            )
        )
    return delta_df, figures


def _heatmap(mat, row_labels, col_labels, *, title, output_path, fmt="{:.2f}") -> Path:
    fig, ax = rl.plt.subplots(figsize=(1.4 * len(col_labels) + 2, 1.0 * len(row_labels) + 2))
    im = ax.imshow(mat, aspect="auto", cmap="viridis")
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    ax.set_xlabel("Test swarm size")
    ax.set_ylabel("Train swarm size")
    ax.set_title(title)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, fmt.format(mat[i, j]), ha="center", va="center",
                        color="white", fontsize=8)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    rl.plt.close(fig)
    return output_path


def _cube_summary_text(cube: pd.DataFrame) -> str:
    lines = ["Cross-size generalization cube", ""]
    for variant in sorted(cube["variant"].unique(), key=variant_sort_key):
        train_sizes, test_sizes, mat = _iqm_grid(cube, variant, "converged")
        lines.append(f"{variant}: mean convergence rate (rows=train, cols=test {[int(s) for s in test_sizes]})")
        for i, tr in enumerate(train_sizes):
            row = "".join(f"{mat[i, j]:>8.2f}" for j in range(len(test_sizes)))
            lines.append(f"  train {tr:>4}: {row}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Zero-shot cross-size generalization analysis.")
    parser.add_argument("--configs", nargs="+", required=True, help="Train config stem(s) to evaluate.")
    parser.add_argument("--eval-sizes", nargs="+", type=int, default=list(DEFAULT_EVAL_SIZES),
                        help="Test swarm sizes (capped at each policy's max_agents).")
    parser.add_argument("--n-episodes", type=int, default=20, help="Episodes per (run, test_size, seed) cell.")
    parser.add_argument("--eval-seeds", nargs="+", type=int, default=list(DEFAULT_EVAL_SEEDS),
                        help="Evaluation seeds (averaged within each cell).")
    parser.add_argument("--model-root", default="model", help="Root of the saved per-run model folders.")
    parser.add_argument("--output-dir", default="results", help="Root output directory.")
    parser.add_argument("--configs-dir", default="training/configs", help="Experiment-config JSON directory.")
    parser.add_argument("--device", default="cpu", help="Torch device for policy inference.")
    parser.add_argument("--reps", type=int, default=rl.DEFAULT_REPS, help="Bootstrap replications.")
    parser.add_argument("--confidence", type=float, default=rl.DEFAULT_CONFIDENCE, help="CI coverage.")
    parser.add_argument("--force", action="store_true", help="Recompute rollouts, ignoring the CSV cache.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel worker processes for model rollouts (1 = sequential, in-process).")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)
    rl.plt.switch_backend("Agg")

    print(
        f"Generalization eval: {len(args.configs)} config(s), "
        f"eval sizes {args.eval_sizes}, {args.n_episodes} episodes x "
        f"{len(args.eval_seeds)} seeds/cell. Cached configs are skipped.",
        flush=True,
    )
    t0 = time.perf_counter()

    common = dict(
        eval_sizes=args.eval_sizes,
        n_episodes=args.n_episodes,
        eval_seeds=args.eval_seeds,
        model_root=args.model_root,
        configs_dir=args.configs_dir,
        device=args.device,
        force=args.force,
        workers=args.workers,
    )

    if len(args.configs) == 1:
        result = analyze_generalization(
            args.configs[0], output_dir=args.output_dir,
            reps=args.reps, confidence=args.confidence, position=(1, 1), **common,
        )
        print(f"\nWrote {result['output_dir']} in {time.perf_counter() - t0:.0f}s")
        print(f"Figures: {', '.join(str(p) for p in result['figures'])}")
        return 0

    result = analyze_cube(
        args.configs, output_dir=args.output_dir,
        reps=args.reps, confidence=args.confidence, **common,
    )
    print(f"\nPer-config + cube written under {args.output_dir} in {time.perf_counter() - t0:.0f}s")
    print(f"Cube: {result['output_dir']} ({len(result['figures'])} figures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
