"""Tests for the chapter6 final-score numbers pipeline (pure table-building functions)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from analysis import log_loading as ll
from analysis import rliable_eval as rl
from analysis import run_final_scores as r6

IQM_INDEX = rl.AGGREGATE_METRIC_NAMES.index("IQM")


def _fake_result(variant_ci: dict) -> dict:
    """variant_ci: {variant: (iqm, ci_low, ci_high)}"""
    point_estimates = {}
    interval_estimates = {}
    for variant, (iqm, lo, hi) in variant_ci.items():
        point = np.zeros(len(rl.AGGREGATE_METRIC_NAMES))
        point[IQM_INDEX] = iqm
        point_estimates[variant] = point
        interval = np.zeros((2, len(rl.AGGREGATE_METRIC_NAMES)))
        interval[0, IQM_INDEX] = lo
        interval[1, IQM_INDEX] = hi
        interval_estimates[variant] = interval
    return {"point_estimates": point_estimates, "interval_estimates": interval_estimates}


# --------------------------------------------------------------------------- #
# embedding_iqm_table
# --------------------------------------------------------------------------- #


def test_embedding_iqm_table_extracts_all_present_variants() -> None:
    results_by_config = {}
    for task in r6.TASKS:
        for size in r6.SIZES:
            variant_ci = {f"embed_dim{k}": (0.1 * k, 0.05 * k, 0.15 * k) for k in r6.EMBED_DIMS}
            results_by_config[r6.embedding_config(task, size)] = _fake_result(variant_ci)

    rows = r6.embedding_iqm_table(results_by_config)
    assert len(rows) == len(r6.TASKS) * len(r6.SIZES) * len(r6.EMBED_DIMS)
    row = next(r for r in rows if r["task"] == "rendezvous" and r["size"] == 4 and r["embed_dim"] == 8)
    assert row["iqm"] == 0.8
    assert row["ci_low"] == pytest.approx(0.4)
    assert row["ci_high"] == pytest.approx(1.2)
    assert row["ci_width"] == pytest.approx(0.8)


def test_embedding_iqm_table_skips_dropped_variants() -> None:
    results_by_config = {}
    for task in r6.TASKS:
        for size in r6.SIZES:
            variant_ci = {f"embed_dim{k}": (1.0, 0.9, 1.1) for k in r6.EMBED_DIMS if k != 4}
            results_by_config[r6.embedding_config(task, size)] = _fake_result(variant_ci)

    rows = r6.embedding_iqm_table(results_by_config)
    assert all(row["embed_dim"] != 4 for row in rows)
    assert len(rows) == len(r6.TASKS) * len(r6.SIZES) * (len(r6.EMBED_DIMS) - 1)


# --------------------------------------------------------------------------- #
# smallest_overlapping_k
# --------------------------------------------------------------------------- #


def test_smallest_overlapping_k_finds_first_overlap() -> None:
    embedding_rows = [
        {"task": "rendezvous", "size": 100, "embed_dim": 4, "ci_low": 0.0, "ci_high": 0.1},
        {"task": "rendezvous", "size": 100, "embed_dim": 8, "ci_low": 0.2, "ci_high": 0.3},
        {"task": "rendezvous", "size": 100, "embed_dim": 16, "ci_low": 0.35, "ci_high": 0.55},
        {"task": "rendezvous", "size": 100, "embed_dim": 32, "ci_low": 0.5, "ci_high": 0.7},
        {"task": "rendezvous", "size": 100, "embed_dim": 64, "ci_low": 0.6, "ci_high": 0.8},
        {"task": "rendezvous", "size": 100, "embed_dim": 128, "ci_low": 0.6, "ci_high": 0.8},
    ]
    result = r6.smallest_overlapping_k(embedding_rows)
    assert len(result) == 1
    entry = result[0]
    assert entry["task"] == "rendezvous"
    assert entry["size"] == 100
    assert entry["reference_embed_dim"] == 64
    # embed_dim=32 [0.5,0.7] overlaps embed_dim=64 [0.6,0.8]; 16 and below do not.
    assert entry["smallest_overlapping_k"] == 32
    assert entry["per_k_overlap"] == {4: False, 8: False, 16: False, 32: True, 64: True, 128: True}


def test_smallest_overlapping_k_skips_cell_missing_reference() -> None:
    embedding_rows = [
        {"task": "pursuit_evasion", "size": 4, "embed_dim": 8, "ci_low": 0.0, "ci_high": 1.0},
    ]
    assert r6.smallest_overlapping_k(embedding_rows) == []


# --------------------------------------------------------------------------- #
# architecture_iqm_table
# --------------------------------------------------------------------------- #


def test_architecture_iqm_table_extracts_all_present_variants() -> None:
    results_by_config = {}
    for task in r6.ARCH_TASKS:
        for size in r6.ARCH_SIZES:
            variant_ci = {
                f"phi_layers{d}_phi_hidden_width{w}": (d + w / 100, d, d + w / 50)
                for d in r6.DEPTHS
                for w in r6.WIDTHS
            }
            results_by_config[r6.architecture_config(task, size)] = _fake_result(variant_ci)

    rows = r6.architecture_iqm_table(results_by_config)
    assert len(rows) == len(r6.ARCH_TASKS) * len(r6.ARCH_SIZES) * len(r6.DEPTHS) * len(r6.WIDTHS)
    row = next(
        r for r in rows if r["task"] == "rendezvous" and r["size"] == 50 and r["depth"] == 2 and r["width"] == 64
    )
    assert row["iqm"] == 2.64

    pe_row = next(
        r for r in rows if r["task"] == "pursuit_evasion" and r["size"] == 4 and r["depth"] == 1 and r["width"] == 32
    )
    assert pe_row["task"] == "pursuit_evasion"


# --------------------------------------------------------------------------- #
# total_timesteps_table
# --------------------------------------------------------------------------- #


def test_total_timesteps_table_computes_expected_formula(tmp_path) -> None:
    config_name = "fake_config"
    spec = {
        "defaults": {
            "env_config": {"num_agents": 4},
            "train_config": {"n_iterations": 1000, "n_steps": 500, "num_vec_envs": 8},
        }
    }
    (tmp_path / f"{config_name}.json").write_text(json.dumps(spec))

    rows = r6.total_timesteps_table(configs_dir=tmp_path, configs=[config_name])
    assert len(rows) == 1
    assert rows[0]["total_timesteps"] == 1000 * 500 * 4 * 8


def test_total_timesteps_table_pursuit_evasion_uses_num_pursuers(tmp_path) -> None:
    config_name = "fake_pursuit_config"
    spec = {
        "defaults": {
            "env_config": {"num_pursuers": 50},
            "train_config": {"n_iterations": 1000, "n_steps": 500, "num_vec_envs": 1},
        }
    }
    (tmp_path / f"{config_name}.json").write_text(json.dumps(spec))

    rows = r6.total_timesteps_table(configs_dir=tmp_path, configs=[config_name])
    assert rows[0]["n_agents"] == 50
    assert rows[0]["total_timesteps"] == 1000 * 500 * 50 * 1


# --------------------------------------------------------------------------- #
# seed_completeness_matrix
# --------------------------------------------------------------------------- #


def _write_series(run_dir: Path, tag: str, steps, values) -> None:
    from torch.utils.tensorboard import SummaryWriter

    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    for step, value in zip(steps, values):
        writer.add_scalar(tag, value, global_step=step)
    writer.close()


def test_seed_completeness_matrix_flags_cells_below_five(tmp_path) -> None:
    logs_root = tmp_path / "logs"
    config = "embedding_scaling_rendezvous_100agents_ppo"
    full = list(range(1, 21))
    for run_name in ("PPO_1", "PPO_2", "PPO_3"):  # only 3 usable runs, no config json -> name-inferred meta
        _write_series(logs_root / config / "embed_dim4" / run_name, ll.BY_ITER_SCORE_TAG, full, [1.0] * 20)

    matrix = r6.seed_completeness_matrix(logs_dir=logs_root, configs_dir=tmp_path / "no_configs", configs=[config])
    row = matrix[matrix.variant == "embed_dim4"].iloc[0]
    assert row.n_usable == 0  # final_iteration inferred as 500 for rendezvous; none of these reach it
    assert bool(row.flagged) is True
