"""Tests for the learning-curve numeric pipeline (subsampling, normalization, pointwise IQM)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np

from analysis import run_learning_curves as rlc


def test_subsample_indices_keeps_every_tenth_and_last() -> None:
    idx = rlc.subsample_indices(95, step=10)
    assert idx.tolist() == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 94]


def test_subsample_indices_no_extra_point_when_last_already_hit() -> None:
    idx = rlc.subsample_indices(91, step=10)
    assert idx.tolist() == [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]


def test_cellwide_minmax_normalize_uses_single_pooled_scale() -> None:
    curves = {
        "a": np.array([[0.0, 10.0], [0.0, 10.0]]),
        "b": np.array([[-10.0, 0.0]]),
    }
    normalized = rlc.cellwide_minmax_normalize(curves)
    np.testing.assert_allclose(normalized["a"], [[0.5, 1.0], [0.5, 1.0]])
    np.testing.assert_allclose(normalized["b"], [[0.0, 0.5]])


def test_cellwide_minmax_normalize_constant_values_no_division_by_zero() -> None:
    curves = {"a": np.full((2, 3), 5.0)}
    normalized = rlc.cellwide_minmax_normalize(curves)
    assert np.all(normalized["a"] == 0.0)


def test_pointwise_iqm_shapes_and_ci_contains_point() -> None:
    rng = np.random.RandomState(0)
    curves = {
        "a": rng.uniform(0.0, 1.0, size=(6, 50)),
        "b": rng.uniform(0.0, 1.0, size=(8, 50)),
    }
    iterations = np.arange(1, 51)
    point, interval, sub_iterations = rlc.pointwise_iqm(curves, iterations, reps=200)

    expected_idx = rlc.subsample_indices(50)
    assert sub_iterations.tolist() == (iterations[expected_idx]).tolist()
    for variant in curves:
        assert point[variant].shape == (len(expected_idx),)
        assert interval[variant].shape == (2, len(expected_idx))
        lower, upper = interval[variant]
        assert np.all(lower <= point[variant] + 1e-8)
        assert np.all(point[variant] - 1e-8 <= upper)


def test_pointwise_iqm_allows_different_run_counts_per_variant() -> None:
    rng = np.random.RandomState(1)
    curves = {
        "few_runs": rng.uniform(size=(2, 30)),
        "many_runs": rng.uniform(size=(5, 30)),
    }
    iterations = np.arange(1, 31)
    point, interval, sub_iterations = rlc.pointwise_iqm(curves, iterations, reps=200)
    assert point["few_runs"].shape == point["many_runs"].shape


def _fake_summary(config: str, environment: str, variants: list, n_points: int = 5) -> dict:
    rng = np.random.RandomState(hash(config) % (2**32))
    point = {v: rng.uniform(0.0, 1.0, size=n_points) for v in variants}
    interval = {v: np.sort(rng.uniform(0.0, 1.0, size=(2, n_points)), axis=0) for v in variants}
    return {
        "config": config,
        "meta": {"environment": environment},
        "anomalies": [],
        "point": point,
        "interval": interval,
        "iterations": np.arange(1, n_points + 1) * 10,
    }


def test_plot_embedding_grid_writes_png_and_pdf(tmp_path) -> None:
    variants = [f"embed_dim{k}" for k in rlc.EMBED_DIMS]
    summaries = {
        (task, size): _fake_summary(rlc.embedding_config(task, size), task, variants)
        for task in rlc.TASKS
        for size in rlc.SIZES
    }
    paths = rlc.plot_embedding_grid(summaries, tmp_path / "fig_6_1")
    assert len(paths) == 2
    for p in paths:
        assert p.exists() and p.stat().st_size > 0
    assert {p.suffix for p in paths} == {".png", ".pdf"}


def test_plot_architecture_grid_writes_png_and_pdf(tmp_path) -> None:
    variants = [f"phi_layers{d}_phi_hidden_width{w}" for d in rlc.DEPTHS for w in rlc.WIDTHS]
    summaries = {
        size: _fake_summary(rlc.architecture_config("rendezvous", size), "rendezvous", variants)
        for size in rlc.ARCH_SIZES
    }
    paths = rlc.plot_architecture_grid(summaries, tmp_path / "fig_6_2")
    assert len(paths) == 2
    for p in paths:
        assert p.exists() and p.stat().st_size > 0


def test_build_curves_csv_shape_and_columns() -> None:
    embed_variants = [f"embed_dim{k}" for k in rlc.EMBED_DIMS]
    embedding_summaries = {
        (task, size): _fake_summary(rlc.embedding_config(task, size), task, embed_variants, n_points=3)
        for task in rlc.TASKS
        for size in rlc.SIZES
    }
    arch_variants = [f"phi_layers{d}_phi_hidden_width{w}" for d in rlc.DEPTHS for w in rlc.WIDTHS]
    architecture_summaries = {
        (task, size): _fake_summary(rlc.architecture_config(task, size), task, arch_variants, n_points=3)
        for task in rlc.ARCH_TASKS
        for size in rlc.ARCH_SIZES
    }
    frame = rlc.build_curves_csv(embedding_summaries, architecture_summaries, reps=1234)

    expected_rows = (
        len(embedding_summaries) * len(embed_variants) * 3 + len(architecture_summaries) * len(arch_variants) * 3
    )
    assert len(frame) == expected_rows
    assert set(frame.columns) == {
        "figure",
        "config",
        "environment",
        "size",
        "variant",
        "iteration",
        "iqm",
        "ci_low",
        "ci_high",
        "reps",
    }
    assert (frame["reps"] == 1234).all()
    assert set(frame["figure"]) == {"fig_6_1_embedding", "fig_6_2_architecture"}
