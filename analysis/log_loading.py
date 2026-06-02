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
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from analysis.rliable_eval import ScoreDict

PathLike = Union[str, Path]

DEFAULT_SCORE_TAG = "rollout/ep_rew_mean"
DEFAULT_ALGORITHM = "PPO"
_RUN_DIR_RE = re.compile(r"^(PPO|TRPO)_(\d+)$")
_SEED_TOKEN_RE = re.compile(r"^seed\d+$")
_SIZE_RE = re.compile(r"_(\d+)agents")


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


def _method_label(variant_dir_name: str) -> str:
    """Variant directory name with any ``seed<n>`` token removed."""
    parts = [part for part in variant_dir_name.split("_") if not _SEED_TOKEN_RE.match(part)]
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
        return float(np.mean(values[-min(last_k, values.size):]))
    raise ValueError(f"Unknown reduction: {reduction!r}")


def _collect_variant_scores(
    config_dir: Path,
    algorithm: str,
    score_tag: str,
    reduction: str,
    last_k: int,
) -> Dict[str, List[float]]:
    """Map each variant of a config to its list of per-run scores."""
    variant_scores: Dict[str, List[float]] = {}
    for variant_dir in sorted(p for p in config_dir.iterdir() if p.is_dir()):
        scores: List[float] = []
        for run_dir in _ordered_run_dirs(variant_dir, algorithm):
            score = _read_run_score(run_dir, score_tag, reduction, last_k)
            if score is not None:
                scores.append(score)
        if scores:
            variant_scores[_method_label(variant_dir.name)] = scores
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

    Returns:
        ``scores`` mapping variant to a ``(n_runs, 1)`` matrix, ``task_labels``
        (the single swarm size as a string), and the resolved metadata (with an
        added ``n_runs`` key).

    Raises:
        FileNotFoundError: If the config log directory does not exist.
        ValueError: If no variant has at least ``min_runs`` runs of ``algorithm``.
    """
    config_dir = Path(logs_dir) / config_name
    if not config_dir.is_dir():
        raise FileNotFoundError(f"No log directory for config '{config_name}': {config_dir}")

    meta = read_config_meta(config_name, configs_dir)
    variant_scores = _collect_variant_scores(config_dir, algorithm, score_tag, reduction, last_k)
    variant_scores = {method: values for method, values in variant_scores.items() if len(values) >= min_runs}
    if not variant_scores:
        raise ValueError(
            f"No variant of '{config_name}' has >= {min_runs} {algorithm} runs with tag '{score_tag}'."
        )

    n_runs = min(len(values) for values in variant_scores.values())
    scores: ScoreDict = {
        method: np.asarray(values[:n_runs], dtype=np.float64).reshape(n_runs, 1)
        for method, values in sorted(variant_scores.items())
    }
    meta["n_runs"] = n_runs
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

    n_runs = min(
        len(per_config[column][method]) for column in range(len(ordered_names)) for method in shared_methods
    )
    scores: ScoreDict = {}
    for method in sorted(shared_methods):
        matrix = np.empty((n_runs, len(ordered_names)), dtype=np.float64)
        for column, variant_scores in enumerate(per_config):
            matrix[:, column] = variant_scores[method][:n_runs]
        scores[method] = matrix

    task_labels = [str(meta["size"]) for meta in ordered_metas]
    return scores, task_labels, ordered_metas
