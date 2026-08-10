"""Build rliable score matrices from the project's TensorBoard run logs.

The training pipeline writes one event file per run at::

    logs/<config>/<variant>/<ALGO>_<run>/events.out.tfevents.*

where ``<config>`` is the experiment-config stem (one swarm size), ``<variant>``
is a matrix-parameter combination (e.g. ``embed_dim16``), and ``<ALGO>_<run>``
holds the repeated runs (``PPO_1`` … ``PPO_5``). This module reads a scalar
(``rollout/ep_rew_mean`` by default) from each run and assembles the
``{method: ndarray(n_runs, n_tasks)}`` dictionaries consumed by
:mod:`analysis.rliable_eval`.

Two views are provided:

* :func:`load_config_scores` — a single config: ``method`` = variant, the only
  ``task`` is that config's swarm size (matrix shape ``(n_runs, 1)``).
* :func:`load_comparison_scores` — several configs of one family: ``method`` =
  variant, ``task`` = swarm size, giving ``(n_runs, n_tasks)`` matrices for the
  variants present in every config.

Scores are returned **raw** (un-normalized); callers apply
:func:`analysis.rliable_eval.normalize_scores` as needed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union

import numpy as np
import pandas as pd
from tensorboard.backend.event_processing import event_file_loader

from analysis.rliable_eval import ScoreDict

PathLike = Union[str, Path]

DEFAULT_SCORE_TAG = "rollout/ep_rew_mean"
DEFAULT_ALGORITHM = "PPO"
_RUN_DIR_RE = re.compile(r"^(PPO|TRPO)_(\d+)$")
_SEED_TOKEN_RE = re.compile(r"^seed\d+$")
_ACTIVATION_TOKEN_RE = re.compile(r"^activation[a-z0-9]+$", re.IGNORECASE)
_SIZE_RE = re.compile(r"_(\d+)agents")

# Step = SB3 iteration index (1, 2, 3, ...), unlike DEFAULT_SCORE_TAG (step = num_timesteps).
BY_ITER_SCORE_TAG = "by_iter/rollout/ep_rew_mean"

# n_iterations per environment (same for every swarm size within a task).
FINAL_ITERATION_BY_ENVIRONMENT: Dict[str, int] = {"rendezvous": 500, "pursuit_evasion": 1000}


def read_config_meta(config_name: str, configs_dir: PathLike = "training/configs") -> Dict[str, object]:
    """Resolve environment, swarm size, study type and algorithm for a config.

    Prefers the matching ``training/configs/<config_name>.json`` and falls back to
    parsing the config name when the file is unavailable.

    Returns:
        Dict with keys ``environment``, ``size``, ``study`` (``"embedding"`` /
        ``"architecture"`` / ``"unknown"``), ``algorithm`` and ``config_name``.
    """
    meta: Dict[str, object] = {
        "config_name": config_name,
        "environment": None,
        "size": None,
        "study": None,
        "algorithm": None,
    }
    path = Path(configs_dir) / f"{config_name}.json"
    if path.exists():
        config = json.loads(path.read_text())
        env_config = config.get("defaults", {}).get("env_config", {})
        train_config = config.get("defaults", {}).get("train_config", {})
        matrix = config.get("matrix_parameters", {})
        meta["environment"] = env_config.get("environment")
        meta["size"] = env_config.get("num_agents", env_config.get("num_pursuers"))
        meta["algorithm"] = train_config.get("algorithm")
        if "embed_dim" in matrix:
            meta["study"] = "embedding"
        elif "phi_layers" in matrix or "phi_hidden_width" in matrix:
            meta["study"] = "architecture"

    if meta["environment"] is None:
        meta["environment"] = "pursuit_evasion" if "pursuit_evasion" in config_name else "rendezvous"
    if meta["size"] is None:
        match = _SIZE_RE.search(config_name)
        meta["size"] = int(match.group(1)) if match else None
    if meta["study"] is None:
        if config_name.startswith("embedding_scaling"):
            meta["study"] = "embedding"
        elif config_name.startswith("architecture_scalability"):
            meta["study"] = "architecture"
        else:
            meta["study"] = "unknown"
    if meta["algorithm"] is None:
        meta["algorithm"] = DEFAULT_ALGORITHM.lower()
    return meta


@dataclass
class RunSeries:
    """A deduplicated scalar series (one value per step) from a run's event file(s).

    On a duplicate step (a restarted run reusing the same directory), the value
    from the last-loaded event wins.
    """

    steps: List[int]
    values: List[float]
    had_restart: bool
    n_event_files: int


@dataclass
class RunAnomaly:
    """A data-quality note about one run or variant, for reporting."""

    config_name: str
    variant: str
    run_label: str
    kind: str
    detail: str


def infer_final_iteration(meta: Mapping[str, object]) -> int:
    """Look up the configured ``n_iterations`` for a config's environment."""
    environment = meta.get("environment")
    if environment not in FINAL_ITERATION_BY_ENVIRONMENT:
        raise ValueError(f"No known final iteration for environment {environment!r}.")
    return FINAL_ITERATION_BY_ENVIRONMENT[str(environment)]


def _read_run_series(run_dir: Path, tag: str) -> Optional[RunSeries]:
    """Read one scalar tag from a run directory's event file(s), deduplicated by step."""
    event_files = sorted(run_dir.glob("events.out.tfevents.*"))
    if not event_files:
        return None

    n_raw = 0
    series: Dict[int, float] = {}
    for event_file in event_files:
        loader = event_file_loader.LegacyEventFileLoader(str(event_file))
        for event in loader.Load():
            if not event.HasField("summary"):
                continue
            for value in event.summary.value:
                if value.tag == tag:
                    n_raw += 1
                    series[event.step] = value.simple_value

    if not series:
        return None
    ordered_steps = sorted(series)
    return RunSeries(
        steps=ordered_steps,
        values=[series[step] for step in ordered_steps],
        had_restart=n_raw != len(series),
        n_event_files=len(event_files),
    )


def _run_is_complete(series: RunSeries, final_iteration: int) -> bool:
    """Reaches ``final_iteration`` with no gaps from the run's own minimum step.

    Contiguity is checked from the run's own start (not iteration 1) to tolerate
    a systematic logging offset (pursuit_evasion's ``by_iter`` tag starts at
    iteration 3 for every run) while still catching truncated/crashed runs.
    """
    step_set = set(series.steps)
    if max(step_set) < final_iteration:
        return False
    expected = set(range(min(step_set), final_iteration + 1))
    return expected <= step_set


def _scan_config_runs(
    config_name: str,
    *,
    logs_dir: PathLike = "logs",
    configs_dir: PathLike = "training/configs",
    algorithm: str = DEFAULT_ALGORITHM,
    tag: str = BY_ITER_SCORE_TAG,
    final_iteration: Optional[int] = None,
) -> Tuple[Dict[str, List[Tuple[str, Optional[RunSeries], bool]]], Dict[str, object], int]:
    """Single-pass scan shared by :func:`scan_config_completeness` and :func:`load_config_curves`.

    Returns ``variant -> [(run_label, series_or_None, usable), ...]`` (run order),
    the resolved config metadata, and the final iteration used.
    """
    config_dir = Path(logs_dir) / config_name
    if not config_dir.is_dir():
        raise FileNotFoundError(f"No log directory for config '{config_name}': {config_dir}")

    meta = read_config_meta(config_name, configs_dir)
    final_iter = final_iteration if final_iteration is not None else infer_final_iteration(meta)

    result: Dict[str, List[Tuple[str, Optional[RunSeries], bool]]] = {}
    for variant_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
        variant = _method_label(variant_dir.name)
        rows: List[Tuple[str, Optional[RunSeries], bool]] = []
        for run_dir in _ordered_run_dirs(variant_dir, algorithm):
            series = _read_run_series(run_dir, tag)
            usable = series is not None and _run_is_complete(series, final_iter)
            rows.append((run_dir.name, series, usable))
        result[variant] = rows
    return result, meta, final_iter


def scan_config_completeness(
    config_name: str,
    *,
    logs_dir: PathLike = "logs",
    configs_dir: PathLike = "training/configs",
    algorithm: str = DEFAULT_ALGORITHM,
    tag: str = BY_ITER_SCORE_TAG,
    final_iteration: Optional[int] = None,
) -> pd.DataFrame:
    """Per-(variant, run) completeness table for one config (nothing dropped).

    Returns:
        DataFrame with columns ``config, environment, study, size, variant, run,
        usable, had_restart, max_iteration, n_points, note``.
    """
    scanned, meta, final_iter = _scan_config_runs(
        config_name,
        logs_dir=logs_dir,
        configs_dir=configs_dir,
        algorithm=algorithm,
        tag=tag,
        final_iteration=final_iteration,
    )
    rows = []
    for variant, run_rows in scanned.items():
        for run_label, series, usable in run_rows:
            if series is None:
                rows.append(
                    {
                        "config": config_name,
                        "environment": meta["environment"],
                        "study": meta["study"],
                        "size": meta["size"],
                        "variant": variant,
                        "run": run_label,
                        "usable": False,
                        "had_restart": False,
                        "max_iteration": None,
                        "n_points": 0,
                        "note": f"no event data for tag '{tag}'",
                    }
                )
                continue
            note = ""
            if not usable:
                note = f"max_iteration={max(series.steps)} < final_iteration={final_iter} or gaps present"
            elif series.had_restart:
                note = "multiple event files / duplicate steps merged; kept last write per step"
            rows.append(
                {
                    "config": config_name,
                    "environment": meta["environment"],
                    "study": meta["study"],
                    "size": meta["size"],
                    "variant": variant,
                    "run": run_label,
                    "usable": usable,
                    "had_restart": series.had_restart,
                    "max_iteration": max(series.steps),
                    "n_points": len(series.steps),
                    "note": note,
                }
            )
    columns = [
        "config",
        "environment",
        "study",
        "size",
        "variant",
        "run",
        "usable",
        "had_restart",
        "max_iteration",
        "n_points",
        "note",
    ]
    return pd.DataFrame(rows, columns=columns)


def _method_label(variant_dir_name: str) -> str:
    """Variant directory name with any ``seed<n>`` and ``activation<value>`` token removed.

    ``activation<value>`` (e.g. ``activationrelu``) is a fixed default baked into
    some older log dir names, not a swept matrix axis; stripping it keeps variant
    identity consistent with configs where the same fixed default isn't in the name
    (e.g. pursuit_evasion N=4 vs N=16/50/100 architecture-scalability dirs).
    """
    parts = [
        part
        for part in variant_dir_name.split("_")
        if not _SEED_TOKEN_RE.match(part) and not _ACTIVATION_TOKEN_RE.match(part)
    ]
    return "_".join(parts)


def _ordered_run_dirs(variant_dir: Path, algorithm: str) -> List[Path]:
    """Return the run directories of ``algorithm`` ordered by run index."""
    indexed: Dict[int, Path] = {}
    for run_dir in variant_dir.iterdir():
        if not run_dir.is_dir():
            continue
        match = _RUN_DIR_RE.match(run_dir.name)
        if match and match.group(1).upper() == algorithm.upper():
            indexed[int(match.group(2))] = run_dir
    return [indexed[index] for index in sorted(indexed)]


def _read_run_score(run_dir: Path, score_tag: str, reduction: str, last_k: int) -> Optional[float]:
    """Read and reduce ``score_tag`` from a single run's event file."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    if not any(run_dir.glob("events.out.tfevents.*")):
        return None
    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    if score_tag not in accumulator.Tags().get("scalars", []):
        return None
    values = np.asarray([event.value for event in accumulator.Scalars(score_tag)], dtype=np.float64)
    if values.size == 0:
        return None
    if reduction == "last":
        return float(values[-1])
    if reduction == "best":
        return float(np.max(values))
    if reduction == "last_k_mean":
        return float(np.mean(values[-min(last_k, values.size) :]))
    raise ValueError(f"Unknown reduction: {reduction!r}")


def _collect_variant_scores(
    config_dir: Path,
    algorithm: str,
    score_tag: str,
    reduction: str,
    last_k: int,
    *,
    allowed_run_names: Optional[Mapping[str, Set[str]]] = None,
) -> Dict[str, List[float]]:
    """Map each variant of a config to its list of per-run scores.

    ``allowed_run_names``, when given, restricts each variant to the named run
    directories (e.g. runs that passed a completeness check).
    """
    variant_scores: Dict[str, List[float]] = {}
    for variant_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
        variant = _method_label(variant_dir.name)
        allowed = allowed_run_names.get(variant) if allowed_run_names is not None else None
        scores: List[float] = []
        for run_dir in _ordered_run_dirs(variant_dir, algorithm):
            if allowed is not None and run_dir.name not in allowed:
                continue
            score = _read_run_score(run_dir, score_tag, reduction, last_k)
            if score is not None:
                scores.append(score)
        if scores:
            variant_scores[variant] = scores
    return variant_scores


def load_config_scores(
    config_name: str,
    *,
    logs_dir: PathLike = "logs",
    configs_dir: PathLike = "training/configs",
    algorithm: str = DEFAULT_ALGORITHM,
    score_tag: str = DEFAULT_SCORE_TAG,
    reduction: str = "last_k_mean",
    last_k: int = 10,
    min_runs: int = 2,
    require_complete: bool = False,
    complete_tag: str = BY_ITER_SCORE_TAG,
    final_iteration: Optional[int] = None,
) -> Tuple[ScoreDict, List[str], Dict[str, object]]:
    """Load per-variant scores for a single config (one swarm size).

    Args:
        config_name: Config/log directory stem (e.g.
            ``embedding_scaling_rendezvous_100agents_ppo``).
        logs_dir: Root directory of the TensorBoard logs.
        configs_dir: Directory holding the experiment-config JSON files.
        algorithm: Run algorithm to read (``"PPO"`` or ``"TRPO"``).
        score_tag: Scalar tag used as the per-run score.
        reduction: ``"last_k_mean"``, ``"last"`` or ``"best"`` over the scalar.
        last_k: Window length for ``reduction="last_k_mean"``.
        min_runs: Drop variants with fewer than this many usable runs.
        require_complete: Exclude runs that don't reach the final iteration
            (see :func:`_run_is_complete`) before scoring, instead of silently
            including a truncated run's premature score.
        complete_tag: Tag used for the completeness check.
        final_iteration: Override the inferred final iteration.

    Returns:
        ``scores`` mapping variant to a ``(n_runs, 1)`` matrix (each variant keeps
        its own usable run count -- one variant losing a seed does not truncate
        the others), ``task_labels`` (the single swarm size as a string), and the
        resolved metadata (with an added ``n_runs`` key mapping variant to count).

    Raises:
        FileNotFoundError: If the config log directory does not exist.
        ValueError: If no variant has at least ``min_runs`` runs of ``algorithm``.
    """
    config_dir = Path(logs_dir) / config_name
    if not config_dir.is_dir():
        raise FileNotFoundError(f"No log directory for config '{config_name}': {config_dir}")

    meta = read_config_meta(config_name, configs_dir)
    allowed_run_names = None
    if require_complete:
        scanned, _, _ = _scan_config_runs(
            config_name,
            logs_dir=logs_dir,
            configs_dir=configs_dir,
            algorithm=algorithm,
            tag=complete_tag,
            final_iteration=final_iteration,
        )
        allowed_run_names = {
            variant: {run_label for run_label, _, usable in rows if usable} for variant, rows in scanned.items()
        }
    variant_scores = _collect_variant_scores(
        config_dir, algorithm, score_tag, reduction, last_k, allowed_run_names=allowed_run_names
    )
    variant_scores = {method: values for method, values in variant_scores.items() if len(values) >= min_runs}
    if not variant_scores:
        raise ValueError(f"No variant of '{config_name}' has >= {min_runs} {algorithm} runs with tag '{score_tag}'.")

    scores: ScoreDict = {
        method: np.asarray(values, dtype=np.float64).reshape(len(values), 1)
        for method, values in sorted(variant_scores.items())
    }
    meta["n_runs"] = {method: len(values) for method, values in variant_scores.items()}
    return scores, [str(meta["size"])], meta


def load_comparison_scores(
    config_names: Sequence[str],
    *,
    logs_dir: PathLike = "logs",
    configs_dir: PathLike = "training/configs",
    algorithm: str = DEFAULT_ALGORITHM,
    score_tag: str = DEFAULT_SCORE_TAG,
    reduction: str = "last_k_mean",
    last_k: int = 10,
    min_runs: int = 2,
) -> Tuple[ScoreDict, List[str], List[Dict[str, object]]]:
    """Load scores for several configs and align them on a swarm-size task axis.

    ``method`` is the variant and ``task`` is the swarm size of each config. Only
    variants present in *every* config are kept, and every cell is truncated to a
    common run count so the resulting matrices are rectangular.

    Args:
        config_names: Config stems to compare (typically one family, varying size).
        logs_dir: Root directory of the TensorBoard logs.
        configs_dir: Directory holding the experiment-config JSON files.
        algorithm: Run algorithm to read.
        score_tag: Scalar tag used as the per-run score.
        reduction: Reduction applied to the scalar series.
        last_k: Window length for ``reduction="last_k_mean"``.
        min_runs: Drop variants with fewer than this many usable runs per config.

    Returns:
        ``scores`` mapping variant to a ``(n_runs, n_tasks)`` matrix, ``task_labels``
        (swarm sizes as strings, ascending), and the per-config metadata.

    Raises:
        ValueError: If fewer than two configs are given or no variant is shared.
        FileNotFoundError: If any config log directory is missing.
    """
    if len(config_names) < 2:
        raise ValueError("load_comparison_scores expects at least two configs.")

    metas = [read_config_meta(name, configs_dir) for name in config_names]
    order = sorted(range(len(config_names)), key=lambda i: metas[i]["size"])
    ordered_names = [config_names[i] for i in order]
    ordered_metas = [metas[i] for i in order]

    per_config: List[Dict[str, List[float]]] = []
    for name in ordered_names:
        config_dir = Path(logs_dir) / name
        if not config_dir.is_dir():
            raise FileNotFoundError(f"No log directory for config '{name}': {config_dir}")
        variant_scores = _collect_variant_scores(config_dir, algorithm, score_tag, reduction, last_k)
        per_config.append({m: v for m, v in variant_scores.items() if len(v) >= min_runs})

    shared_methods = set(per_config[0])
    for variant_scores in per_config[1:]:
        shared_methods &= set(variant_scores)
    if not shared_methods:
        raise ValueError("No variant is present in every config with enough runs.")

    n_runs = min(len(per_config[column][method]) for column in range(len(ordered_names)) for method in shared_methods)
    scores: ScoreDict = {}
    for method in sorted(shared_methods):
        matrix = np.empty((n_runs, len(ordered_names)), dtype=np.float64)
        for column, variant_scores in enumerate(per_config):
            matrix[:, column] = variant_scores[method][:n_runs]
        scores[method] = matrix

    task_labels = [str(meta["size"]) for meta in ordered_metas]
    return scores, task_labels, ordered_metas


@dataclass
class ConfigCurves:
    """Per-iteration learning curves for one config: variant -> (n_runs, n_iterations)."""

    curves: Dict[str, np.ndarray]
    iterations: np.ndarray
    meta: Dict[str, object]
    anomalies: List[RunAnomaly]


def load_config_curves(
    config_name: str,
    *,
    logs_dir: PathLike = "logs",
    configs_dir: PathLike = "training/configs",
    algorithm: str = DEFAULT_ALGORITHM,
    tag: str = BY_ITER_SCORE_TAG,
    final_iteration: Optional[int] = None,
    min_runs: int = 2,
) -> ConfigCurves:
    """Load full per-iteration learning curves for one config.

    Drops incomplete runs and variants left with fewer than ``min_runs`` usable
    runs, then builds the iteration axis as the intersection of the remaining
    runs' logged steps (handles pursuit_evasion's runs never logging iterations
    1-2, and any run with a mid-series gap).

    Raises:
        FileNotFoundError: If the config log directory does not exist.
        ValueError: If no variant is left with >= ``min_runs`` usable runs.
    """
    scanned, meta, final_iter = _scan_config_runs(
        config_name,
        logs_dir=logs_dir,
        configs_dir=configs_dir,
        algorithm=algorithm,
        tag=tag,
        final_iteration=final_iteration,
    )

    anomalies: List[RunAnomaly] = []
    kept: Dict[str, List[RunSeries]] = {}
    for variant, rows in scanned.items():
        usable_series = []
        for run_label, series, usable in rows:
            if series is None:
                anomalies.append(RunAnomaly(config_name, variant, run_label, "missing", f"no data for tag '{tag}'"))
                continue
            if not usable:
                anomalies.append(
                    RunAnomaly(
                        config_name,
                        variant,
                        run_label,
                        "truncated",
                        f"max_iteration={max(series.steps)}, final_iteration={final_iter}",
                    )
                )
                continue
            if series.had_restart:
                anomalies.append(
                    RunAnomaly(config_name, variant, run_label, "restarted", "duplicate steps merged, kept last write")
                )
            usable_series.append(series)
        if len(usable_series) < min_runs:
            anomalies.append(
                RunAnomaly(
                    config_name,
                    variant,
                    "*",
                    "dropped_variant",
                    f"only {len(usable_series)} usable runs (< min_runs={min_runs})",
                )
            )
            continue
        kept[variant] = usable_series

    if not kept:
        raise ValueError(f"No variant of '{config_name}' has >= {min_runs} usable runs.")

    grid: Optional[Set[int]] = None
    for series_list in kept.values():
        for series in series_list:
            step_set = set(series.steps)
            grid = step_set if grid is None else (grid & step_set)
    iterations = np.array(sorted(grid), dtype=np.int64)

    curves: Dict[str, np.ndarray] = {}
    for variant, series_list in sorted(kept.items()):
        matrix = np.empty((len(series_list), len(iterations)), dtype=np.float64)
        for row, series in enumerate(series_list):
            lookup = dict(zip(series.steps, series.values))
            matrix[row] = [lookup[step] for step in iterations]
        curves[variant] = matrix

    meta = dict(meta)
    meta["n_runs"] = {variant: matrix.shape[0] for variant, matrix in curves.items()}
    meta["final_iteration"] = final_iter
    return ConfigCurves(curves=curves, iterations=iterations, meta=meta, anomalies=anomalies)
