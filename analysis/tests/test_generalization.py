"""Tests for the zero-shot cross-size generalization pipeline.

Deliberately model-free: the resolver and loader are exercised with fake file
trees / synthetic frames, and the rollout engine with a real (tiny) env driven
by a stub policy. No trained ``.zip`` is loaded, so the suite runs fast and does
not depend on the presence of trained models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis import generalization_loading as gl
from analysis import generalization_resolver as gr
from analysis.generalization_eval import (
    EpisodeResult,
    build_eval_env,
    rollout,
    write_raw_csv,
)
from analysis.generalization_resolver import load_config_spec


# --------------------------------------------------------------------------- #
# Resolver
# --------------------------------------------------------------------------- #

def test_model_prefix_strips_agents_and_fixes_typo() -> None:
    assert (
        gr.model_prefix_for_config("embedding_scaling_rendezvous_16agents_ppo")
        == "embedding_scaling_rendezvous_16_ppo"
    )
    assert (
        gr.model_prefix_for_config("architecture_scalability_rendezvous_4agents")
        == "architecture_schaling_rendezvous_4"
    )


def _make_model_tree(root, prefix, runs, dims):
    for run in runs:
        d = root / f"{prefix}_{run}"
        d.mkdir(parents=True)
        for dim in dims:
            (d / f"embed_dim{dim}.zip").write_text("stub")
            (d / f"embed_dim{dim}_checkpoints").mkdir()  # should be ignored


def test_resolve_run_dirs_and_variants(tmp_path) -> None:
    _make_model_tree(
        tmp_path, "embedding_scaling_rendezvous_16_ppo", runs=[1, 2, 3], dims=[4, 16, 64]
    )
    stem = "embedding_scaling_rendezvous_16agents_ppo"
    runs = gr.resolve_run_dirs(stem, model_root=tmp_path)
    assert sorted(runs) == [1, 2, 3]
    assert gr.discover_variants(runs[1]) == [4, 16, 64]


def test_resolve_run_dirs_ppo_fallback(tmp_path) -> None:
    # Config stem lacks the algo suffix the dirs carry (architecture sweep).
    _make_model_tree(
        tmp_path, "architecture_schaling_rendezvous_50_ppo", runs=[1, 2], dims=[]
    )
    runs = gr.resolve_run_dirs("architecture_scalability_rendezvous_50agents", model_root=tmp_path)
    assert sorted(runs) == [1, 2]


def test_resolve_models_skips_missing_zip(tmp_path) -> None:
    _make_model_tree(tmp_path, "embedding_scaling_rendezvous_4_ppo", runs=[1], dims=[16])
    stem = "embedding_scaling_rendezvous_4agents_ppo"
    models = gr.resolve_models(stem, model_root=tmp_path, variants=[16, 999])
    assert [m.embed_dim for m in models] == [16]  # 999 does not exist -> skipped
    assert models[0].run == 1
    assert models[0].zip_path.exists()


def test_load_config_spec_reads_real_config() -> None:
    spec = load_config_spec("embedding_scaling_rendezvous_16agents_ppo")
    assert spec.train_size == 16
    assert spec.variants == [4, 8, 16, 32, 64, 128]
    assert spec.env_config["max_agents"] == 100


# --------------------------------------------------------------------------- #
# Loader (aggregation + pivot)
# --------------------------------------------------------------------------- #

def _synthetic_raw() -> pd.DataFrame:
    rows = []
    for variant in ("embed_dim16", "embed_dim64"):
        for run in (1, 2):
            for test_size in (4, 16):
                for ep in range(3):  # 3 episodes to be meaned away
                    rows.append(
                        dict(
                            config="c", variant=variant, run=run, train_size=16,
                            test_size=test_size, seed=0, episode=ep,
                            ep_reward=float(run + ep + test_size),
                            max_pairwise_distance=2.0, distance_to_com=1.0,
                            ep_length=10, converged=True,
                        )
                    )
    return pd.DataFrame(rows)


def test_aggregate_means_over_episodes() -> None:
    agg = gl.aggregate(_synthetic_raw())
    # one row per (variant, run, test_size): 2 * 2 * 2 = 8
    assert len(agg) == 8
    cell = agg[(agg.variant == "embed_dim16") & (agg.run == 1) & (agg.test_size == 4)]
    # ep_reward = mean(run+ep+test) over ep in {0,1,2} = 1+ (0+1+2)/3 +4 = 6.0
    assert float(cell["ep_reward"].iloc[0]) == pytest.approx(6.0)


def test_to_score_dict_shape_and_order() -> None:
    agg = gl.aggregate(_synthetic_raw())
    score_dict, sizes = gl.to_score_dict(agg, metric="ep_reward")
    assert sizes == [4, 16]
    assert list(score_dict) == ["embed_dim16", "embed_dim64"]  # ascending embed_dim
    for mat in score_dict.values():
        assert mat.shape == (2, 4 // 4 + 1)  # (n_runs=2, n_test_sizes=2)


def test_to_score_dict_marks_missing_cells_nan() -> None:
    df = _synthetic_raw()
    df = df[~((df.variant == "embed_dim16") & (df.run == 2) & (df.test_size == 16))]
    score_dict, sizes = gl.to_score_dict(gl.aggregate(df), metric="ep_reward")
    mat = score_dict["embed_dim16"]  # run index 1 (run=2), size index 1 (16) -> NaN
    assert np.isnan(mat[1, 1])


# --------------------------------------------------------------------------- #
# Rollout engine (real tiny env, stub policy)
# --------------------------------------------------------------------------- #

@pytest.fixture
def tiny_env_config():
    spec = load_config_spec("embedding_scaling_rendezvous_4agents_ppo")
    cfg = dict(spec.env_config)
    cfg["max_steps"] = 12  # keep the rollout short
    return cfg


def test_build_eval_env_overrides_size_and_caps(tiny_env_config) -> None:
    env = build_eval_env(tiny_env_config, test_size=8, max_agents=100)
    assert env.agent_handler.num_agents == 8
    assert env.max_agents == 100  # NOT overridden to test size
    with pytest.raises(ValueError):
        build_eval_env(tiny_env_config, test_size=200, max_agents=100)


class _StubPolicy:
    """Random-action stand-in for a trained PPO (no learning, no zip)."""

    def __init__(self, action_space, seed: int = 0) -> None:
        self.action_space = action_space
        self.action_space.seed(seed)

    def predict(self, obs, deterministic: bool = True):
        actions = np.stack([self.action_space.sample() for _ in range(obs.shape[0])])
        return actions, None


def test_rollout_records_fields_and_truncation(tiny_env_config) -> None:
    from training.common_train_utils import wrap_env_for_sb3
    from analysis.generalization_eval import RENDEZVOUS_MONITOR_KEYS

    test_size = 4
    env = build_eval_env(tiny_env_config, test_size)
    vec = wrap_env_for_sb3(env, n_envs=1, monitor_keywords=RENDEZVOUS_MONITOR_KEYS)
    try:
        model = _StubPolicy(vec.action_space)
        episodes = rollout(model, vec, n_agents=test_size, n_episodes=2, max_steps=12)
    finally:
        vec.close()

    assert len(episodes) == 2
    for ep in episodes:
        assert set(ep) == {
            "ep_reward", "max_pairwise_distance", "distance_to_com", "ep_length", "converged"
        }
        # A random policy will not rendezvous within 12 steps -> truncation.
        assert ep["ep_length"] == 12
        assert ep["converged"] is False
        assert np.isfinite(ep["max_pairwise_distance"])  # near-terminal proxy, not NaN


def test_write_raw_csv_roundtrip(tmp_path) -> None:
    rec = EpisodeResult(
        config="c", variant="embed_dim16", run=1, train_size=16, test_size=4, seed=0,
        episode=0, ep_reward=-5.0, max_pairwise_distance=2.3, distance_to_com=1.1,
        ep_length=20, converged=True,
    )
    path = write_raw_csv([rec], tmp_path / "raw.csv")
    df = pd.read_csv(path)
    assert len(df) == 1
    assert df.loc[0, "variant"] == "embed_dim16"
    assert float(df.loc[0, "ep_reward"]) == pytest.approx(-5.0)
