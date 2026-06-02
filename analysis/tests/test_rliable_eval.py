"""Tests for the rliable-based statistical evaluation module."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from analysis import rliable_eval


@pytest.fixture
def synthetic_scores() -> dict:
    """Three configs with distinct, overlapping score ranges over six tasks."""
    rng = np.random.RandomState(0)
    return {
        "embed_dim16": rng.uniform(0.20, 0.80, size=(8, 6)),
        "embed_dim32": rng.uniform(0.30, 0.90, size=(8, 6)),
        "embed_dim64": rng.uniform(0.40, 1.00, size=(10, 6)),
    }


def test_iqm_within_task_extrema_and_ci_contains_point(synthetic_scores: dict) -> None:
    np.random.seed(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        point_estimates, interval_estimates = rliable_eval.aggregate_iqm(
            synthetic_scores, reps=2000, random_state=np.random.RandomState(123)
        )

    iqm_index = rliable_eval.AGGREGATE_METRIC_NAMES.index("IQM")
    for config, matrix in synthetic_scores.items():
        iqm_value = point_estimates[config][iqm_index]
        per_task_min = matrix.min(axis=0)
        per_task_max = matrix.max(axis=0)
        assert per_task_min.min() <= iqm_value <= per_task_max.max()

        lower, upper = interval_estimates[config][:, iqm_index]
        assert lower <= iqm_value + 1e-8
        assert iqm_value - 1e-8 <= upper
        assert lower <= upper


def test_all_aggregate_cis_contain_their_point(synthetic_scores: dict) -> None:
    np.random.seed(1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        point_estimates, interval_estimates = rliable_eval.aggregate_iqm(
            synthetic_scores, reps=2000, random_state=np.random.RandomState(7)
        )

    for config, point in point_estimates.items():
        lower = interval_estimates[config][0]
        upper = interval_estimates[config][1]
        assert np.all(lower <= point + 1e-8)
        assert np.all(point - 1e-8 <= upper)


def test_min_max_normalization_maps_each_task_into_unit_interval(synthetic_scores: dict) -> None:
    normalized = rliable_eval.normalize_scores(synthetic_scores, method="min_max")
    stacked = np.concatenate([normalized[config] for config in normalized], axis=0)
    assert stacked.min() >= -1e-9
    assert stacked.max() <= 1 + 1e-9
    assert np.allclose(stacked.min(axis=0), 0.0)
    assert np.allclose(stacked.max(axis=0), 1.0)


def test_probability_of_improvement_detects_dominant_config() -> None:
    tasks = 5
    scores = {
        "strong": np.full((6, tasks), 1.0),
        "weak": np.full((6, tasks), 0.0),
    }
    np.random.seed(2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        point_estimates, interval_estimates = rliable_eval.probability_of_improvement(
            scores, [("strong", "weak")], reps=500
        )
    key = "strong,weak"
    assert point_estimates[key][0] == pytest.approx(1.0)
    lower, upper = interval_estimates[key][:, 0]
    assert 0.0 <= lower <= upper <= 1.0
