"""Tests for TensorBoard log loading: dedup, completeness, and learning curves."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pytest
from torch.utils.tensorboard import SummaryWriter

from analysis import log_loading as ll


def _write_series(run_dir: Path, tag: str, steps: Iterable[int], values: Iterable[float]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(run_dir))
    for step, value in zip(steps, values):
        writer.add_scalar(tag, value, global_step=step)
    writer.close()


def _make_config(
    logs_root: Path,
    config: str,
    variants: dict,
    tag: str = ll.BY_ITER_SCORE_TAG,
) -> None:
    """variants: {variant_name: {run_name: (steps, values)}}"""
    for variant, runs in variants.items():
        for run_name, (steps, values) in runs.items():
            _write_series(logs_root / config / variant / run_name, tag, steps, values)


# --------------------------------------------------------------------------- #
# _read_run_series
# --------------------------------------------------------------------------- #


def test_read_run_series_basic(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_series(run_dir, ll.BY_ITER_SCORE_TAG, range(1, 6), [float(i) for i in range(1, 6)])
    series = ll._read_run_series(run_dir, ll.BY_ITER_SCORE_TAG)
    assert series.steps == [1, 2, 3, 4, 5]
    assert series.values == pytest.approx([1.0, 2.0, 3.0, 4.0, 5.0])
    assert series.had_restart is False
    assert series.n_event_files == 1


def test_read_run_series_missing_dir_returns_none(tmp_path) -> None:
    assert ll._read_run_series(tmp_path / "nope", ll.BY_ITER_SCORE_TAG) is None


def test_read_run_series_wrong_tag_returns_none(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_series(run_dir, "some/other/tag", [1], [1.0])
    assert ll._read_run_series(run_dir, ll.BY_ITER_SCORE_TAG) is None


def test_read_run_series_restart_keeps_last_write(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_series(run_dir, ll.BY_ITER_SCORE_TAG, range(1, 6), [1.0] * 5)
    _write_series(run_dir, ll.BY_ITER_SCORE_TAG, range(1, 6), [2.0] * 5)
    series = ll._read_run_series(run_dir, ll.BY_ITER_SCORE_TAG)
    assert series.n_event_files == 2
    assert series.had_restart is True
    assert series.steps == [1, 2, 3, 4, 5]
    assert series.values == pytest.approx([2.0] * 5)


# --------------------------------------------------------------------------- #
# _run_is_complete
# --------------------------------------------------------------------------- #


def test_run_is_complete_true_for_full_range() -> None:
    series = ll.RunSeries(steps=list(range(1, 501)), values=[0.0] * 500, had_restart=False, n_event_files=1)
    assert ll._run_is_complete(series, 500) is True


def test_run_is_complete_false_when_truncated() -> None:
    series = ll.RunSeries(steps=list(range(1, 317)), values=[0.0] * 316, had_restart=False, n_event_files=1)
    assert ll._run_is_complete(series, 500) is False


def test_run_is_complete_true_with_systematic_offset() -> None:
    # pursuit_evasion-style: starts at 3, no gaps, reaches final iteration.
    series = ll.RunSeries(steps=list(range(3, 1001)), values=[0.0] * 998, had_restart=False, n_event_files=1)
    assert ll._run_is_complete(series, 1000) is True


def test_run_is_complete_false_with_gap() -> None:
    steps = [s for s in range(1, 501) if s != 250]
    series = ll.RunSeries(steps=steps, values=[0.0] * len(steps), had_restart=False, n_event_files=1)
    assert ll._run_is_complete(series, 500) is False


# --------------------------------------------------------------------------- #
# load_config_curves
# --------------------------------------------------------------------------- #


def test_load_config_curves_grid_intersection_and_shape(tmp_path) -> None:
    logs_root = tmp_path / "logs"
    config = "embedding_scaling_pursuit_evasion_4agents_ppo"
    steps_a = list(range(3, 21))  # ragged start, like real pursuit_evasion
    steps_b = list(range(1, 21))
    variants = {
        "embed_dim4": {
            "PPO_1": (steps_a, [float(s) for s in steps_a]),
            "PPO_2": (steps_b, [float(s) for s in steps_b]),
        },
    }
    _make_config(logs_root, config, variants)

    result = ll.load_config_curves(config, logs_dir=logs_root, final_iteration=20, min_runs=1)
    assert list(result.iterations) == list(range(3, 21))  # intersection excludes 1-2
    matrix = result.curves["embed_dim4"]
    assert matrix.shape == (2, 18)
    assert matrix[0].tolist() == [float(s) for s in range(3, 21)]
    assert matrix[1].tolist() == [float(s) for s in range(3, 21)]


def test_load_config_curves_drops_truncated_run(tmp_path) -> None:
    logs_root = tmp_path / "logs"
    config = "embedding_scaling_rendezvous_100agents_ppo"
    full = list(range(1, 21))
    variants = {
        "embed_dim4": {
            "PPO_1": (full, [1.0] * 20),
            "PPO_2": (full, [1.0] * 20),
            "PPO_3": (list(range(1, 11)), [1.0] * 10),  # truncated: never reaches 20
        },
    }
    _make_config(logs_root, config, variants)

    result = ll.load_config_curves(config, logs_dir=logs_root, final_iteration=20, min_runs=2)
    assert result.curves["embed_dim4"].shape == (2, 20)
    kinds = {a.kind for a in result.anomalies}
    assert "truncated" in kinds


def test_load_config_curves_keeps_independent_run_counts_per_variant(tmp_path) -> None:
    logs_root = tmp_path / "logs"
    config = "embedding_scaling_rendezvous_100agents_ppo"
    full = list(range(1, 21))
    variants = {
        "embed_dim4": {
            "PPO_1": (full, [1.0] * 20),
            "PPO_2": (full, [1.0] * 20),
            "PPO_3": (list(range(1, 11)), [1.0] * 10),  # truncated -> only 2 usable
        },
        "embed_dim8": {
            "PPO_1": (full, [2.0] * 20),
            "PPO_2": (full, [2.0] * 20),
            "PPO_3": (full, [2.0] * 20),  # all 3 usable, unaffected by embed_dim4
        },
    }
    _make_config(logs_root, config, variants)

    result = ll.load_config_curves(config, logs_dir=logs_root, final_iteration=20, min_runs=2)
    assert result.curves["embed_dim4"].shape == (2, 20)
    assert result.curves["embed_dim8"].shape == (3, 20)
    assert result.meta["n_runs"] == {"embed_dim4": 2, "embed_dim8": 3}


def test_load_config_curves_drops_variant_below_min_runs(tmp_path) -> None:
    logs_root = tmp_path / "logs"
    config = "embedding_scaling_rendezvous_100agents_ppo"
    full = list(range(1, 21))
    variants = {
        "embed_dim4": {"PPO_1": (full, [1.0] * 20), "PPO_2": (full, [1.0] * 20)},
        "embed_dim8": {"PPO_1": (full, [2.0] * 20)},  # only 1 usable run
    }
    _make_config(logs_root, config, variants)

    result = ll.load_config_curves(config, logs_dir=logs_root, final_iteration=20, min_runs=2)
    assert set(result.curves) == {"embed_dim4"}
    assert any(a.kind == "dropped_variant" and a.variant == "embed_dim8" for a in result.anomalies)


def test_load_config_curves_restart_is_usable_and_flagged(tmp_path) -> None:
    logs_root = tmp_path / "logs"
    config = "embedding_scaling_rendezvous_100agents_ppo"
    full = list(range(1, 21))
    run_dir = logs_root / config / "embed_dim4" / "PPO_1"
    _write_series(run_dir, ll.BY_ITER_SCORE_TAG, full, [1.0] * 20)  # crashed early attempt
    _write_series(run_dir, ll.BY_ITER_SCORE_TAG, full, [3.0] * 20)  # full restart, wins
    _write_series(logs_root / config / "embed_dim4" / "PPO_2", ll.BY_ITER_SCORE_TAG, full, [1.0] * 20)

    result = ll.load_config_curves(config, logs_dir=logs_root, final_iteration=20, min_runs=2)
    matrix = result.curves["embed_dim4"]
    assert 3.0 in matrix[:, 0]
    assert any(a.kind == "restarted" for a in result.anomalies)


def test_load_config_curves_raises_when_no_variant_qualifies(tmp_path) -> None:
    logs_root = tmp_path / "logs"
    config = "embedding_scaling_rendezvous_100agents_ppo"
    variants = {"embed_dim4": {"PPO_1": (list(range(1, 21)), [1.0] * 20)}}
    _make_config(logs_root, config, variants)
    with pytest.raises(ValueError):
        ll.load_config_curves(config, logs_dir=logs_root, final_iteration=20, min_runs=2)


# --------------------------------------------------------------------------- #
# _method_label
# --------------------------------------------------------------------------- #


def test_method_label_strips_seed_token() -> None:
    assert ll._method_label("embed_dim16_seed0") == "embed_dim16"


def test_method_label_strips_leading_activation_token() -> None:
    assert (
        ll._method_label("activationrelu_phi_layers1_phi_hidden_width32_seed0")
        == "phi_layers1_phi_hidden_width32"
    )


def test_method_label_strips_bare_activation_token_without_seed() -> None:
    assert (
        ll._method_label("activationtanh_phi_layers2_phi_hidden_width64")
        == "phi_layers2_phi_hidden_width64"
    )


def test_method_label_matches_across_pursuit_evasion_n4_and_n16_dir_styles() -> None:
    # Real-world case: N=4 architecture-scalability dirs carry a stray
    # "activationrelu_" prefix that N=16/50/100 dirs never had.
    n4_style = "activationrelu_phi_layers1_phi_hidden_width32_seed0"
    n16_style = "phi_layers1_phi_hidden_width32_seed0"
    assert ll._method_label(n4_style) == ll._method_label(n16_style)


def test_method_label_unaffected_when_no_activation_token() -> None:
    # Rendezvous architecture dirs never had the token; behavior must be unchanged.
    assert ll._method_label("phi_layers4_phi_hidden_width128") == "phi_layers4_phi_hidden_width128"


def test_method_label_does_not_strip_non_activation_tokens_named_similarly() -> None:
    # Only tokens matching activation<value> are stripped; unrelated tokens survive.
    assert ll._method_label("embed_dim16_activation") == "embed_dim16_activation"


# --------------------------------------------------------------------------- #
# scan_config_completeness
# --------------------------------------------------------------------------- #


def test_scan_config_completeness_reports_every_run(tmp_path) -> None:
    logs_root = tmp_path / "logs"
    config = "embedding_scaling_rendezvous_100agents_ppo"
    variants = {
        "embed_dim4": {
            "PPO_1": (list(range(1, 21)), [1.0] * 20),
            "PPO_2": (list(range(1, 11)), [1.0] * 10),  # truncated
        },
    }
    _make_config(logs_root, config, variants)

    frame = ll.scan_config_completeness(config, logs_dir=logs_root, final_iteration=20)
    assert len(frame) == 2
    row1 = frame[frame.run == "PPO_1"].iloc[0]
    row2 = frame[frame.run == "PPO_2"].iloc[0]
    assert bool(row1.usable) is True
    assert bool(row2.usable) is False


# --------------------------------------------------------------------------- #
# load_config_scores(require_complete=True)
# --------------------------------------------------------------------------- #


def test_load_config_scores_require_complete_excludes_truncated_run(tmp_path) -> None:
    logs_root = tmp_path / "logs"
    config = "embedding_scaling_rendezvous_100agents_ppo"
    full = list(range(1, 21))
    variants = {
        "embed_dim4": {
            "PPO_1": (full, [10.0] * 20),
            "PPO_2": (full, [10.0] * 20),
            "PPO_3": (list(range(1, 11)), [999.0] * 10),  # truncated, extreme value
        },
    }
    _make_config(logs_root, config, variants)

    scores, _, meta = ll.load_config_scores(
        config,
        logs_dir=logs_root,
        min_runs=2,
        score_tag=ll.BY_ITER_SCORE_TAG,
        require_complete=True,
        final_iteration=20,
    )
    assert meta["n_runs"] == {"embed_dim4": 2}
    assert np.all(scores["embed_dim4"] == pytest.approx(10.0))


def test_load_config_scores_keeps_independent_run_counts_per_variant(tmp_path) -> None:
    logs_root = tmp_path / "logs"
    config = "embedding_scaling_rendezvous_100agents_ppo"
    full = list(range(1, 21))
    variants = {
        "embed_dim4": {
            "PPO_1": (full, [10.0] * 20),
            "PPO_2": (full, [10.0] * 20),
            "PPO_3": (list(range(1, 11)), [999.0] * 10),  # truncated -> only 2 usable
        },
        "embed_dim8": {
            "PPO_1": (full, [20.0] * 20),
            "PPO_2": (full, [20.0] * 20),
            "PPO_3": (full, [20.0] * 20),  # all 3 usable, unaffected by embed_dim4
        },
    }
    _make_config(logs_root, config, variants)

    scores, _, meta = ll.load_config_scores(
        config,
        logs_dir=logs_root,
        min_runs=2,
        score_tag=ll.BY_ITER_SCORE_TAG,
        require_complete=True,
        final_iteration=20,
    )
    assert scores["embed_dim4"].shape == (2, 1)
    assert scores["embed_dim8"].shape == (3, 1)
    assert meta["n_runs"] == {"embed_dim4": 2, "embed_dim8": 3}
