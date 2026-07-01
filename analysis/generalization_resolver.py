"""Resolve a training-config stem to its trained-model zip paths on disk.

The experiment configs and the saved model directories use slightly different
naming, so this module bridges them:

    config stem:  embedding_scaling_rendezvous_16agents_ppo
    model dir:    model/embedding_scaling_rendezvous_16_ppo_<run>/embed_dim<d>.zip

Transforms applied (config -> model-dir prefix):
  * ``<N>agents`` -> ``<N>``           (e.g. ``16agents`` -> ``16``)
  * ``architecture_scalability`` -> ``architecture_schaling``  (a known typo
    baked into the saved directory names)

Resolution is data-driven: after computing the expected prefix we glob for
``<prefix>_<run>`` run directories and discover the ``embed_dim*.zip`` variants
that actually exist, so partially-trained sweeps resolve to whatever is present.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

# Misspellings frozen into the saved model directory names.
_KNOWN_DIR_TYPOS = {"architecture_scalability": "architecture_schaling"}

_VARIANT_RE = re.compile(r"^embed_dim(\d+)$")


@dataclass(frozen=True)
class ResolvedModel:
    """One trained policy zip for a (config, variant, run)."""

    config: str
    variant: str        # e.g. "embed_dim16"
    embed_dim: int
    run: int
    zip_path: Path


def model_prefix_for_config(config_stem: str) -> str:
    """Map a config stem to the saved model-directory prefix (run suffix excluded)."""
    prefix = re.sub(r"(\d+)agents", r"\1", config_stem)
    for canonical, typo in _KNOWN_DIR_TYPOS.items():
        prefix = prefix.replace(canonical, typo)
    return prefix


def _glob_runs(prefix: str, root: Path) -> Dict[int, Path]:
    runs: Dict[int, Path] = {}
    for d in root.glob(f"{prefix}_*"):
        if not d.is_dir():
            continue
        suffix = d.name[len(prefix) + 1:]
        if suffix.isdigit():
            runs[int(suffix)] = d
    return runs


def resolve_run_dirs(config_stem: str, model_root: str | Path = "model") -> Dict[int, Path]:
    """Find ``{run_id: dir}`` for a config, by globbing ``<prefix>_<run>``.

    Some config stems omit the algorithm suffix that the saved dirs carry (e.g.
    ``architecture_scalability_rendezvous_4agents`` vs ``..._4_ppo_<run>``); if
    the bare prefix matches nothing, retry with ``_ppo`` appended.
    """
    prefix = model_prefix_for_config(config_stem)
    root = Path(model_root)
    runs = _glob_runs(prefix, root)
    if not runs and not prefix.endswith(("_ppo", "_trpo")):
        runs = _glob_runs(f"{prefix}_ppo", root)
    return dict(sorted(runs.items()))


def discover_variants(run_dir: Path) -> List[int]:
    """List embed_dim variants present in a run dir, ascending."""
    dims: List[int] = []
    for zp in run_dir.glob("embed_dim*.zip"):
        m = _VARIANT_RE.match(zp.stem)
        if m:
            dims.append(int(m.group(1)))
    return sorted(dims)


def resolve_models(
    config_stem: str,
    *,
    model_root: str | Path = "model",
    variants: Optional[Sequence[int]] = None,
) -> List[ResolvedModel]:
    """Resolve all (variant, run) -> zip paths that exist for a config.

    Args:
        config_stem: e.g. ``embedding_scaling_rendezvous_16agents_ppo``.
        model_root: directory holding the per-run model folders.
        variants: restrict to these embed_dims; default = discover from disk.

    Returns:
        Existing ``ResolvedModel`` records only (missing zips are skipped).
    """
    run_dirs = resolve_run_dirs(config_stem, model_root)
    out: List[ResolvedModel] = []
    for run, run_dir in run_dirs.items():
        dims = list(variants) if variants is not None else discover_variants(run_dir)
        for ed in dims:
            zp = run_dir / f"embed_dim{ed}.zip"
            if zp.exists():
                out.append(
                    ResolvedModel(
                        config=config_stem,
                        variant=f"embed_dim{ed}",
                        embed_dim=ed,
                        run=run,
                        zip_path=zp,
                    )
                )
    return out


@dataclass(frozen=True)
class ConfigSpec:
    """The parts of a training config the generalization pipeline needs."""

    stem: str
    env_config: Dict
    train_size: int
    variants: List[int]


def load_config_spec(config_path: str | Path, configs_dir: str | Path = "training/configs") -> ConfigSpec:
    """Read a config JSON into a ``ConfigSpec`` (stem, env_config, train size, variants).

    Accepts either a bare stem or a path; resolves under ``configs_dir`` if needed.
    """
    path = Path(config_path)
    if path.suffix != ".json":
        path = Path(configs_dir) / f"{path.name}.json"
    data = json.loads(path.read_text())
    defaults = data.get("defaults", {})
    env_config = defaults.get("env_config", {})
    train_size = int(env_config.get("num_agents"))
    variants = [int(d) for d in data.get("matrix_parameters", {}).get("embed_dim", [])]
    return ConfigSpec(
        stem=path.stem,
        env_config=env_config,
        train_size=train_size,
        variants=sorted(variants),
    )
