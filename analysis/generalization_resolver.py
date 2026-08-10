"""Resolve a training-config stem to its trained-model zip paths on disk.

    config stem:  embedding_scaling_rendezvous_16agents_ppo
    model dir:    model/embedding_scaling_rendezvous_16_ppo_<run>/embed_dim<d>.zip

    config stem:  architecture_scalability_rendezvous_4agents
    model dir:    model/architecture_schaling_rendezvous_4_ppo_<run>/phi_layers<l>_phi_hidden_width<w>.zip

    config stem:  architecture_scalability_pursuit_evasion_4agents
    model dir:    model/architecture_scaling_pusuit_evasion_4_ppo_<run>/phi_layers<l>_phi_hidden_width<w>_seed0.zip

Resolution is data-driven: after computing candidate prefixes we glob for
``<prefix>_<run>`` run directories and discover the checkpoint zips that
actually exist, so partially-trained sweeps resolve to whatever is present.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# On-disk directory-naming schemes, tried in order until one has matches.
_KNOWN_DIR_NAMING_SCHEMES: Tuple[Tuple[Tuple[str, str], ...], ...] = (
    (),
    (("architecture_scalability", "architecture_schaling"),),
    (
        ("architecture_scalability", "architecture_scaling"),
        ("pursuit_evasion", "pusuit_evasion"),
    ),
)

_EMBED_DIM_LABEL_RE = re.compile(r"^embed_dim(\d+)$")
_PHI_LABEL_RE = re.compile(r"^phi(\d+)_w(\d+)$")

_EMBED_DIM_FILE_RE = re.compile(r"^embed_dim(\d+)$")
_PHI_FILE_RE = re.compile(r"^(?:activationrelu_)?phi_layers(\d+)_phi_hidden_width(\d+)(?:_seed\d+)?$")


@dataclass(frozen=True)
class ResolvedModel:
    config: str
    variant: str        # e.g. "embed_dim16" or "phi2_w64"
    run: int
    zip_path: Path


def variant_sort_key(variant: str) -> Tuple[int, ...]:
    m = _EMBED_DIM_LABEL_RE.match(variant)
    if m:
        return (int(m.group(1)),)
    m = _PHI_LABEL_RE.match(variant)
    if m:
        return (int(m.group(1)), int(m.group(2)))
    raise ValueError(f"Unrecognized variant label: {variant!r}")


def _variant_label_from_zip_stem(stem: str) -> Optional[str]:
    m = _EMBED_DIM_FILE_RE.match(stem)
    if m:
        return f"embed_dim{int(m.group(1))}"
    m = _PHI_FILE_RE.match(stem)
    if m:
        return f"phi{int(m.group(1))}_w{int(m.group(2))}"
    return None


def model_prefix_for_config(config_stem: str) -> str:
    return re.sub(r"(\d+)agents", r"\1", config_stem)


def _prefix_candidates(prefix: str) -> List[str]:
    candidates = []
    for substitutions in _KNOWN_DIR_NAMING_SCHEMES:
        candidate = prefix
        for canonical, replacement in substitutions:
            candidate = candidate.replace(canonical, replacement)
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


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
    prefix = model_prefix_for_config(config_stem)
    root = Path(model_root)
    for candidate in _prefix_candidates(prefix):
        for probe in (candidate, f"{candidate}_ppo"):
            runs = _glob_runs(probe, root)
            if runs:
                return dict(sorted(runs.items()))
    return {}


def discover_variants(run_dir: Path) -> List[str]:
    labels = (_variant_label_from_zip_stem(zp.stem) for zp in run_dir.glob("*.zip"))
    return sorted((label for label in labels if label is not None), key=variant_sort_key)


def resolve_models(
    config_stem: str,
    *,
    model_root: str | Path = "model",
    variants: Optional[Sequence[str]] = None,
) -> List[ResolvedModel]:
    run_dirs = resolve_run_dirs(config_stem, model_root)
    wanted = set(variants) if variants is not None else None
    out: List[ResolvedModel] = []
    for run, run_dir in run_dirs.items():
        for zp in sorted(run_dir.glob("*.zip")):
            label = _variant_label_from_zip_stem(zp.stem)
            if label is None or (wanted is not None and label not in wanted):
                continue
            out.append(ResolvedModel(config=config_stem, variant=label, run=run, zip_path=zp))
    return out


@dataclass(frozen=True)
class ConfigSpec:
    stem: str
    env_config: Dict
    train_size: int
    variants: List[str]  # e.g. ["embed_dim4", ...] or ["phi1_w32", ...]
    task: str            # "rendezvous" | "pursuit_evasion"
    max_size: int        # obs-space cap (max_agents / max_pursuers)


def _variants_from_matrix(matrix_parameters: Dict) -> List[str]:
    if "embed_dim" in matrix_parameters:
        dims = sorted(int(d) for d in matrix_parameters["embed_dim"])
        return [f"embed_dim{d}" for d in dims]
    if "phi_layers" in matrix_parameters and "phi_hidden_width" in matrix_parameters:
        layers = sorted(int(x) for x in matrix_parameters["phi_layers"])
        widths = sorted(int(x) for x in matrix_parameters["phi_hidden_width"])
        return [f"phi{layer}_w{width}" for layer in layers for width in widths]
    return []


def load_config_spec(config_path: str | Path, configs_dir: str | Path = "training/configs") -> ConfigSpec:
    path = Path(config_path)
    if path.suffix != ".json":
        path = Path(configs_dir) / f"{path.name}.json"
    data = json.loads(path.read_text())
    defaults = data.get("defaults", {})
    env_config = defaults.get("env_config", {})
    task = env_config.get("environment", "rendezvous")
    if task == "pursuit_evasion":
        train_size = int(env_config.get("num_pursuers"))
        max_size = int(env_config.get("max_pursuers", train_size))
    else:
        train_size = int(env_config.get("num_agents"))
        max_size = int(env_config.get("max_agents", train_size))
    variants = _variants_from_matrix(data.get("matrix_parameters", {}))
    return ConfigSpec(
        stem=path.stem,
        env_config=env_config,
        train_size=train_size,
        variants=variants,
        task=task,
        max_size=max_size,
    )
