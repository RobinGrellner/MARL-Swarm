"""Tests for the zero-shot cross-size generalization pipeline.

Mostly model-free: the resolver and loader are exercised with fake file trees
/ synthetic frames, and the rollout engine with a real (tiny) env driven by a
stub policy. The one exception is the parallel-vs-sequential equivalence test,
which saves a real (untrained) PPO checkpoint to prove the multiprocessing
path doesn't lose or corrupt results.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from stable_baselines3 import PPO

from analysis import generalization_loading as gl
from analysis import generalization_resolver as gr
from analysis import run_generalization as rg
from analysis.generalization_eval import (
    RENDEZVOUS_MONITOR_KEYS,
    PURSUIT_MONITOR_KEYS,
    EpisodeResult,
    build_eval_env,
    rollout,
    write_raw_csv,
)
from analysis.generalization_resolver import ConfigSpec, load_config_spec
from training.common_train_utils import wrap_env_for_sb3

# --------------------------------------------------------------------------- #
# Resolver
# --------------------------------------------------------------------------- #

def test_model_prefix_only_strips_agents_suffix() -> None:
    assert (
        gr.model_prefix_for_config("embedding_scaling_rendezvous_16agents_ppo")
        == "embedding_scaling_rendezvous_16_ppo"
    )
    assert (
        gr.model_prefix_for_config("architecture_scalability_rendezvous_4agents")
        == "architecture_scalability_rendezvous_4"
    )


def _make_model_tree(root, prefix, runs, filenames):
    for run in runs:
        d = root / f"{prefix}_{run}"
        d.mkdir(parents=True)
        for name in filenames:
            (d / f"{name}.zip").write_text("stub")
            (d / f"{name}_checkpoints").mkdir()  # should be ignored


def test_resolve_run_dirs_and_variants(tmp_path) -> None:
    _make_model_tree(
        tmp_path, "embedding_scaling_rendezvous_16_ppo", runs=[1, 2, 3],
        filenames=["embed_dim4", "embed_dim16", "embed_dim64"],
    )
    stem = "embedding_scaling_rendezvous_16agents_ppo"
    runs = gr.resolve_run_dirs(stem, model_root=tmp_path)
    assert sorted(runs) == [1, 2, 3]
    assert gr.discover_variants(runs[1]) == ["embed_dim4", "embed_dim16", "embed_dim64"]


def test_resolve_run_dirs_ppo_fallback_and_rendezvous_typo(tmp_path) -> None:
    # architecture_scalability's rendezvous dirs: typo'd "schaling", no algo suffix in the config stem.
    _make_model_tree(tmp_path, "architecture_schaling_rendezvous_50_ppo", runs=[1, 2], filenames=[])
    runs = gr.resolve_run_dirs("architecture_scalability_rendezvous_50agents", model_root=tmp_path)
    assert sorted(runs) == [1, 2]


def test_resolve_run_dirs_pursuit_evasion_typo(tmp_path) -> None:
    # architecture_scalability's pursuit_evasion dirs: reworded "scaling" + typo'd "pusuit_evasion".
    _make_model_tree(tmp_path, "architecture_scaling_pusuit_evasion_4_ppo", runs=[1], filenames=[])
    runs = gr.resolve_run_dirs("architecture_scalability_pursuit_evasion_4agents", model_root=tmp_path)
    assert sorted(runs) == [1]


def test_resolve_run_dirs_embedding_scaling_pursuit_evasion_not_mistyped(tmp_path) -> None:
    # embedding_scaling's pursuit_evasion dirs are spelled correctly; the
    # architecture_scaling typo scheme must not be applied here.
    _make_model_tree(tmp_path, "embedding_scaling_pursuit_evasion_16_ppo", runs=[1], filenames=[])
    runs = gr.resolve_run_dirs("embedding_scaling_pursuit_evasion_16agents_ppo", model_root=tmp_path)
    assert sorted(runs) == [1]


def test_discover_variants_phi_naming_tolerates_prefix_and_suffix_noise(tmp_path) -> None:
    _make_model_tree(
        tmp_path, "architecture_scaling_pusuit_evasion_4_ppo", runs=[1],
        filenames=[
            "activationrelu_phi_layers1_phi_hidden_width32_seed0",
            "phi_layers1_phi_hidden_width64_seed0",
            "phi_layers2_phi_hidden_width32",
        ],
    )
    run_dir = gr.resolve_run_dirs("architecture_scaling_pusuit_evasion_4_ppo", model_root=tmp_path)[1]
    assert gr.discover_variants(run_dir) == ["phi1_w32", "phi1_w64", "phi2_w32"]


def test_resolve_models_skips_missing_zip(tmp_path) -> None:
    _make_model_tree(tmp_path, "embedding_scaling_rendezvous_4_ppo", runs=[1], filenames=["embed_dim16"])
    stem = "embedding_scaling_rendezvous_4agents_ppo"
    models = gr.resolve_models(stem, model_root=tmp_path, variants=["embed_dim16", "embed_dim999"])
    assert [m.variant for m in models] == ["embed_dim16"]  # embed_dim999 does not exist -> skipped
    assert models[0].run == 1
    assert models[0].zip_path.exists()


def test_variant_sort_key() -> None:
    assert gr.variant_sort_key("embed_dim16") == (16,)
    assert gr.variant_sort_key("phi2_w64") == (2, 64)
    with pytest.raises(ValueError):
        gr.variant_sort_key("not_a_variant")


def test_load_config_spec_reads_real_config() -> None:
    spec = load_config_spec("embedding_scaling_rendezvous_16agents_ppo")
    assert spec.train_size == 16
    assert spec.variants == ["embed_dim4", "embed_dim8", "embed_dim16", "embed_dim32", "embed_dim64", "embed_dim128"]
    assert spec.env_config["max_agents"] == 100
    assert spec.task == "rendezvous"
    assert spec.max_size == 100


def test_load_config_spec_reads_pursuit_config() -> None:
    spec = load_config_spec("embedding_scaling_pursuit_evasion_16agents_ppo")
    assert spec.task == "pursuit_evasion"
    assert spec.train_size == 16
    assert spec.max_size == 100
    assert spec.variants == ["embed_dim4", "embed_dim8", "embed_dim16", "embed_dim32", "embed_dim64", "embed_dim128"]


def test_load_config_spec_reads_architecture_scaling_config() -> None:
    spec = load_config_spec("architecture_scalability_rendezvous_4agents")
    assert spec.task == "rendezvous"
    assert spec.train_size == 4
    assert spec.variants == [
        "phi1_w32", "phi1_w64", "phi1_w128",
        "phi2_w32", "phi2_w64", "phi2_w128",
        "phi4_w32", "phi4_w64", "phi4_w128",
    ]


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


@pytest.fixture
def tiny_pursuit_env_config():
    spec = load_config_spec("embedding_scaling_pursuit_evasion_4agents_ppo")
    cfg = dict(spec.env_config)
    cfg["max_steps"] = 12  # keep the rollout short
    return cfg


def test_build_eval_env_overrides_size_and_caps(tiny_env_config) -> None:
    env = build_eval_env(tiny_env_config, test_size=8, max_agents=100)
    assert env.agent_handler.num_agents == 8
    assert env.max_agents == 100  # NOT overridden to test size
    with pytest.raises(ValueError):
        build_eval_env(tiny_env_config, test_size=200, max_agents=100)


def test_build_eval_env_pursuit_dispatch(tiny_pursuit_env_config) -> None:
    env = build_eval_env(tiny_pursuit_env_config, test_size=8, max_agents=100)
    assert env.num_pursuers == 8
    assert env._max_pursuers == 100  # NOT overridden to test size
    with pytest.raises(ValueError):
        build_eval_env(tiny_pursuit_env_config, test_size=200, max_agents=100)


class _StubPolicy:
    """Random-action stand-in for a trained PPO (no learning, no zip)."""

    def __init__(self, action_space, seed: int = 0) -> None:
        self.action_space = action_space
        self.action_space.seed(seed)

    def predict(self, obs, deterministic: bool = True):
        actions = np.stack([self.action_space.sample() for _ in range(obs.shape[0])])
        return actions, None


def test_rollout_records_fields_and_truncation(tiny_env_config) -> None:
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
            "ep_reward", "mean_pairwise_distance", "max_pairwise_distance",
            "distance_to_com", "ep_length", "converged"
        }
        # A random policy will not rendezvous within 12 steps -> truncation.
        assert ep["ep_length"] == 12
        assert ep["converged"] is False
        assert np.isfinite(ep["max_pairwise_distance"])  # near-terminal proxy, not NaN
        assert np.isfinite(ep["mean_pairwise_distance"])
        assert ep["mean_pairwise_distance"] <= ep["max_pairwise_distance"]


def test_rollout_pursuit_records_fields(tiny_pursuit_env_config) -> None:
    test_size = 4
    env = build_eval_env(tiny_pursuit_env_config, test_size)
    vec = wrap_env_for_sb3(env, n_envs=1, monitor_keywords=PURSUIT_MONITOR_KEYS)
    try:
        model = _StubPolicy(vec.action_space)
        episodes = rollout(
            model, vec, n_agents=test_size, n_episodes=2, max_steps=12,
            monitor_keys=PURSUIT_MONITOR_KEYS,
        )
    finally:
        vec.close()

    assert len(episodes) == 2
    for ep in episodes:
        assert set(ep) == {"ep_reward", "min_distance_to_evader", "ep_length", "converged"}
        assert ep["ep_length"] <= 12
        # converged == terminated early == evader captured
        assert ep["converged"] == (ep["ep_length"] < 12)


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
    # Task-specific fields not supplied stay NaN (shared CSV schema).
    assert np.isnan(df.loc[0, "capture_time"])


# --------------------------------------------------------------------------- #
# Delta analysis (Gl. 56/57)
# --------------------------------------------------------------------------- #

def test_delta_analysis_diagonal_is_zero(tmp_path) -> None:
    rows = []
    for train_size in (4, 16):
        for run in (1, 2):
            for test_size in (4, 16):
                # Native policies score best at their own size; transfer degrades.
                reward = -1.0 if test_size == train_size else -3.0 - run
                rows.append(
                    dict(variant="embed_dim16", run=run, train_size=train_size,
                         test_size=test_size, ep_reward=reward)
                )
    cube = pd.DataFrame(rows)

    delta_df, figures = rg._delta_analysis(
        cube, tmp_path, tmp_path, reps=200, confidence=0.95
    )

    assert (tmp_path / "delta_iqm.csv").exists()
    assert len(figures) == 1
    diag = delta_df[delta_df.train_size == delta_df.test_size]
    assert np.allclose(diag["delta_iqm"], 0.0)
    off = delta_df[delta_df.train_size != delta_df.test_size]
    # Transfer is worse than native here, so the gap is positive off-diagonal.
    assert (off["delta_iqm"] > 0).all()


# --------------------------------------------------------------------------- #
# Cube task-mixing guard
# --------------------------------------------------------------------------- #

def test_analyze_cube_rejects_mixed_tasks(tmp_path, monkeypatch) -> None:
    specs = {
        "cfg_a": ConfigSpec(
            stem="cfg_a", env_config={}, train_size=4,
            variants=["embed_dim16"], task="rendezvous", max_size=100,
        ),
        "cfg_b": ConfigSpec(
            stem="cfg_b", env_config={}, train_size=4,
            variants=["embed_dim16"], task="pursuit_evasion", max_size=100,
        ),
    }

    def fake_analyze_generalization(config, **_kwargs):
        agg = pd.DataFrame(
            {"variant": ["embed_dim16"], "run": [1], "test_size": [4], "ep_reward": [-1.0]}
        )
        return {"spec": specs[config], "aggregated": agg}

    monkeypatch.setattr(rg, "analyze_generalization", fake_analyze_generalization)

    with pytest.raises(ValueError, match="mixed tasks"):
        rg.analyze_cube(["cfg_a", "cfg_b"], output_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Parallel vs. sequential equivalence
# --------------------------------------------------------------------------- #

def test_run_or_load_raw_parallel_matches_sequential(tmp_path) -> None:
    spec = load_config_spec("embedding_scaling_rendezvous_4agents_ppo")
    spec = replace(spec, env_config={**spec.env_config, "max_steps": 12})

    env = build_eval_env(spec.env_config, test_size=4, max_agents=spec.max_size)
    vec = wrap_env_for_sb3(env, n_envs=1)
    try:
        model_dir = tmp_path / "model" / "embedding_scaling_rendezvous_4_ppo_1"
        model_dir.mkdir(parents=True)
        PPO("MlpPolicy", vec, n_steps=8, batch_size=8, device="cpu").save(
            str(model_dir / "embed_dim16.zip")
        )
    finally:
        vec.close()

    kwargs = dict(
        spec=spec, variants=["embed_dim16"], test_sizes=[4], n_episodes=2,
        eval_seeds=[0], model_root=tmp_path / "model", device="cpu", verbose=False,
    )
    df_seq = gl.run_or_load_raw(cache_path=tmp_path / "seq.csv", workers=1, **kwargs)
    df_par = gl.run_or_load_raw(cache_path=tmp_path / "par.csv", workers=2, **kwargs)

    key_cols = ["config", "variant", "run", "test_size", "seed", "episode"]
    assert sorted(df_seq.columns) == sorted(df_par.columns)
    assert sorted(map(tuple, df_seq[key_cols].to_numpy().tolist())) == sorted(
        map(tuple, df_par[key_cols].to_numpy().tolist())
    )
