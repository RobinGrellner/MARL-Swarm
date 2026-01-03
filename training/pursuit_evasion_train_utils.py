"""
Training utilities for Pursuit-Evasion environment.

This module provides functions to train pursuer agents to capture an evader
using PPO or TRPO with parameter sharing and mean embedding feature extraction.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import warnings

from environments.pursuit.pursuit_evasion_env import PursuitEvasionEnv
from supersuit import flatten_v0 as supersuit_flatten
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.base_class import BaseAlgorithm
from sb3_contrib import TRPO
from policies.mean_embedding_extractor import MeanEmbeddingExtractor

import torch

# Suppress render_mode warning from VecEnv
warnings.filterwarnings("ignore", message=".*render_mode.*", category=UserWarning)


def wrap_env_for_sb3(
    env: PursuitEvasionEnv, *, n_envs: int = 1, monitor_keywords: Optional[Tuple[str, ...]] = None
) -> VecMonitor:
    """Wrap a PettingZoo Pursuit-Evasion environment for Stable-Baselines3 training.

    The wrapping sequence is:
    1. Flatten per-agent observations into single vectors
    2. Convert PettingZoo env to vectorized SB3 environment
    3. Concatenate n_envs copies for parallel training
    4. Apply VecMonitor for episode statistics

    Args:
        env: PursuitEvasionEnv instance
        n_envs: Number of parallel environments
        monitor_keywords: Optional keys to monitor from info dict

    Returns:
        Wrapped vectorized environment ready for SB3
    """
    env_flat = supersuit_flatten(env)
    vec_env = pettingzoo_env_to_vec_env_v1(env_flat)
    vec_env = concat_vec_envs_v1(vec_env, num_vec_envs=n_envs, num_cpus=0, base_class="stable_baselines3")

    if monitor_keywords is not None:
        vec_env = VecMonitor(vec_env, info_keywords=monitor_keywords)
    else:
        vec_env = VecMonitor(vec_env)

    return vec_env


def make_policy_kwargs(layout: Dict[str, int], *, embed_dim: int = 64, phi_layers: int = 1) -> Dict[str, Any]:
    """Build policy kwargs with mean embedding feature extractor.

    Args:
        layout: Observation layout dict with local_dim, neigh_dim, max_neighbours
        embed_dim: Embedding dimension
        phi_layers: Number of hidden layers in φ network

    Returns:
        Policy kwargs for SB3 algorithm
    """
    local_dim = layout["local_dim"]
    neigh_dim = layout["neigh_dim"]
    max_neigh = layout["max_neighbours"]

    phi_hidden = [64] * max(phi_layers, 1)

    return {
        "features_extractor_class": MeanEmbeddingExtractor,
        "features_extractor_kwargs": {
            "local_dim": local_dim,
            "neigh_dim": neigh_dim,
            "max_neigh": max_neigh,
            "embed_dim": embed_dim,
            "phi_hidden": phi_hidden,
        },
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
    }


def setup_model(
    vec_env: VecMonitor, policy_kwargs: Dict[str, Any], algo_params: Dict[str, Any], algorithm: str = "ppo"
) -> BaseAlgorithm:
    """Setup RL model (PPO or TRPO).

    Args:
        vec_env: Vectorized environment
        policy_kwargs: Policy network configuration
        algo_params: Algorithm hyperparameters
        algorithm: "ppo" or "trpo"

    Returns:
        Configured RL model
    """
    algo_params = algo_params.copy()

    if "device" not in algo_params:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        algo_params["device"] = device
    else:
        device = algo_params["device"]

    print(f"\n{'=' * 60}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Selected Device: {device}")
    print(f"{'=' * 60}\n")

    algorithm = algorithm.lower()
    if algorithm == "ppo":
        return PPO(
            policy="MlpPolicy",
            env=vec_env,
            policy_kwargs=policy_kwargs,
            **algo_params,
        )
    elif algorithm == "trpo":
        return TRPO(
            policy="MlpPolicy",
            env=vec_env,
            policy_kwargs=policy_kwargs,
            **algo_params,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'ppo' or 'trpo'.")


def run_training_pursuit_evasion(
    env: PursuitEvasionEnv,
    embed_config: Dict[str, Any],
    algo_params: Dict[str, Any],
    *,
    algorithm: str = "ppo",
    total_timesteps: int = 200_000,
    log_info_keys: Optional[Tuple[str, ...]] = None,
    n_envs: int = 1,
    save_path: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> Tuple[BaseAlgorithm, Dict[str, Any]]:
    """Train RL model (PPO or TRPO) on Pursuit-Evasion environment.

    Args:
        env: PursuitEvasionEnv instance
        embed_config: Configuration for mean embedding extractor
        algo_params: Algorithm hyperparameters
        algorithm: "ppo" or "trpo"
        total_timesteps: Total training timesteps
        log_info_keys: Keys to log from env info dict
        n_envs: Number of parallel environments
        save_path: Path to save model (optional)
        resume_from: Path to a saved model to resume from (optional)

    Returns:
        Tuple of (trained model, info dict with vec_env)
    """
    # 1. Extract layout before wrapping
    layout = env.obs_layout
    # 2. Wrap environment
    vec_env = wrap_env_for_sb3(env, n_envs=n_envs, monitor_keywords=log_info_keys)

    # 3. Handle resume vs fresh training
    if resume_from:
        print(f"\n{'=' * 60}")
        print(f"Resuming training from: {resume_from}")
        print(f"{'=' * 60}\n")

        resume_algorithm = algorithm.lower()
        if resume_algorithm == "ppo":
            model = PPO.load(resume_from, env=vec_env, device=algo_params.get("device", "cpu"))
        elif resume_algorithm == "trpo":
            model = TRPO.load(resume_from, env=vec_env, device=algo_params.get("device", "cpu"))
        else:
            raise ValueError(f"Unknown algorithm: {resume_algorithm}")

        if "learning_rate" in algo_params:
            model.learning_rate = algo_params["learning_rate"]
    else:
        # Create new model
        policy_kwargs = make_policy_kwargs(
            layout, embed_dim=embed_config.get("embed_dim", 64), phi_layers=embed_config.get("phi_layers", 1)
        )

        algorithm = algorithm.lower()
        if algorithm == "ppo":
            default_params = {
                "learning_rate": 3e-4,
                "n_steps": 1024,
                "batch_size": 512,
                "n_epochs": 5,
                "gamma": 0.99,
                "gae_lambda": 0.98,
                "clip_range": 0.2,
                "target_kl": 0.015,
                "verbose": 1,
            }
        elif algorithm == "trpo":
            default_params = {
                "learning_rate": 1e-3,
                "n_steps": 2048,
                "batch_size": 128,
                "gamma": 0.99,
                "gae_lambda": 0.98,
                "n_critic_updates": 5,
                "cg_max_steps": 10,
                "cg_damping": 0.1,
                "target_kl": 0.01,
                "verbose": 1,
            }
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        default_params.update(algo_params)
        model = setup_model(vec_env, policy_kwargs, default_params, algorithm=algorithm)

    # 4. Train model
    model.learn(total_timesteps=total_timesteps)

    # 5. Save model if path provided
    if save_path:
        model.save(save_path)
        print(f"Model saved to {save_path}")

    info = {
        "layout": layout,
        "embed_config": embed_config,
        "algo_params": algo_params,
        "algorithm": algorithm,
        "vec_env": vec_env,
    }
    return model, info
