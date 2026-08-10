"""Run/cache cross-size eval rollouts and shape them for rliable.

Pipeline:
  1. ``run_or_load_raw`` -- resolve all (variant, run) policy zips for a config,
     roll each out across the test-size grid, and persist per-episode rows to a
     CSV cache (re-runs reuse the cache unless ``force=True``). Checkpoints are
     evaluated across ``workers`` parallel processes.
  2. ``aggregate`` -- collapse episodes x eval-seeds to one score per
     (variant, run, test_size) cell.
  3. ``to_score_dict`` -- pivot a chosen metric into the rliable
     ``{variant: (n_runs, n_test_sizes)}`` matrices, with variants as the
     method axis and the test sizes as the task axis.

The score that matches the training analysis (``rollout/ep_rew_mean``) is
``ep_reward``. NOTE: raw episode reward is NOT comparable across test sizes (the
env's reward normalisation is built from ``num_agents``); per-test-size
normalisation for the transfer views lives in ``run_generalization``.
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

from analysis.generalization_eval import EpisodeResult, evaluate_variant
from analysis.generalization_resolver import (
    ConfigSpec,
    ResolvedModel,
    resolve_models,
    variant_sort_key,
)

ScoreDict = Dict[str, np.ndarray]

# Columns that uniquely identify an evaluation cell vs. the per-episode metrics.
_CELL_KEYS = ["variant", "run", "test_size"]
_METRICS = [
    "ep_reward", "mean_pairwise_distance", "max_pairwise_distance", "distance_to_com",
    "min_distance_to_evader", "capture_time", "ep_length", "converged",
]


def _init_worker() -> None:
    # Process-level parallelism already saturates the machine; per-process
    # BLAS auto-threading on top of that only adds contention.
    torch.set_num_threads(1)


def _evaluate_model(
    model: ResolvedModel,
    *,
    env_config: Dict[str, Any],
    config: str,
    train_size: int,
    test_sizes: Sequence[int],
    task: str,
    n_episodes: int,
    eval_seeds: Sequence[int],
    device: str,
    max_agents: int,
) -> List[EpisodeResult]:
    """Evaluate one resolved checkpoint; the unit of work for each pool worker."""
    return evaluate_variant(
        model.zip_path, env_config, config=config, variant=model.variant, run=model.run,
        train_size=train_size, test_sizes=test_sizes, task=task, n_episodes=n_episodes,
        eval_seeds=eval_seeds, device=device, max_agents=max_agents,
    )


def run_or_load_raw(
    spec: ConfigSpec,
    *,
    test_sizes: Sequence[int],
    n_episodes: int,
    eval_seeds: Sequence[int],
    cache_path: Path,
    model_root: str | Path = "model",
    device: str = "cpu",
    variants: Optional[Sequence[str]] = None,
    force: bool = False,
    verbose: bool = True,
    workers: int = 4,
) -> pd.DataFrame:
    """Return per-episode eval records, computing (and caching) them if needed."""
    cache_path = Path(cache_path)
    if cache_path.exists() and not force:
        if verbose:
            print(f"  cache hit -> {cache_path} (skipping rollouts)", flush=True)
        return pd.read_csv(cache_path)

    use_variants = list(variants) if variants is not None else spec.variants
    models = resolve_models(spec.stem, model_root=model_root, variants=use_variants)
    if not models:
        raise FileNotFoundError(
            f"No trained models found for '{spec.stem}' under '{model_root}'. "
            "Check the config stem and that the sweep finished training."
        )

    if verbose:
        print(
            f"  resolved {len(models)} models; test sizes {list(test_sizes)}; "
            f"{n_episodes} episodes x {len(eval_seeds)} seeds per cell; {workers} worker(s)",
            flush=True,
        )

    common_kwargs = dict(
        env_config=spec.env_config, config=spec.stem, train_size=spec.train_size,
        test_sizes=test_sizes, task=spec.task, n_episodes=n_episodes,
        eval_seeds=eval_seeds, device=device, max_agents=spec.max_size,
    )

    t0 = time.perf_counter()
    if workers <= 1:
        pairs = [(m, _evaluate_model(m, **common_kwargs)) for m in models]
    else:
        os.environ.setdefault("OMP_NUM_THREADS", "1")
        os.environ.setdefault("MKL_NUM_THREADS", "1")
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            futures = {pool.submit(_evaluate_model, m, **common_kwargs): m for m in models}
            pairs = [(futures[f], f.result()) for f in as_completed(futures)]

    records: List[EpisodeResult] = []
    for idx, (m, recs) in enumerate(pairs, 1):
        records.extend(recs)
        if verbose:
            print(f"  [{idx:>3}/{len(models)}] {m.variant:<12} run {m.run} -> {len(recs):>4} eps", flush=True)

    df = pd.DataFrame([asdict(r) for r in records])
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_path, index=False)
    if verbose:
        print(
            f"  cached {len(df)} rows -> {cache_path} "
            f"({time.perf_counter() - t0:.0f}s total)",
            flush=True,
        )
    return df


def aggregate(df: pd.DataFrame, metrics: Sequence[str] = tuple(_METRICS)) -> pd.DataFrame:
    """Mean over episodes and eval-seeds -> one row per (variant, run, test_size)."""
    present = [m for m in metrics if m in df.columns]
    return df.groupby(_CELL_KEYS, as_index=False)[present].mean()


def to_score_dict(
    agg: pd.DataFrame, metric: str = "ep_reward"
) -> Tuple[ScoreDict, List[int]]:
    """Pivot an aggregated frame into ``{variant: (n_runs, n_test_sizes)}``.

    Rows are runs (sorted ascending), columns are test sizes (sorted ascending).
    Missing (run, test_size) cells become NaN so callers can detect gaps.
    """
    test_sizes = sorted(int(s) for s in agg["test_size"].unique())
    variants = sorted(agg["variant"].unique(), key=variant_sort_key)
    score_dict: ScoreDict = {}
    for v in variants:
        sub = agg[agg["variant"] == v]
        mat = (
            sub.pivot(index="run", columns="test_size", values=metric)
            .reindex(columns=test_sizes)
            .sort_index()
        )
        score_dict[v] = mat.to_numpy(dtype=float)
    return score_dict, test_sizes
