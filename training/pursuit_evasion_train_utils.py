"""
Training utilities for Pursuit-Evasion environment.

This module provides functions to train pursuer agents to capture an evader
using PPO or TRPO with parameter sharing and mean embedding feature extraction.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from stable_baselines3.common.base_class import BaseAlgorithm

from environments.pursuit.pursuit_evasion_env import PursuitEvasionEnv

# Import the generic training function
from training.common_train_utils import run_training


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

    Backward-compatible wrapper around the generic training function.

    Args:
        env: PursuitEvasionEnv instance
        embed_config: Configuration for mean embedding extractor
        algo_params: Algorithm hyperparameters
        algorithm: "ppo" or "trpo"
        total_timesteps: Total training timesteps
        log_info_keys: Keys to log from env info dict (default: capture success metrics)
        n_envs: Number of parallel environments
        save_path: Path to save model (optional)
        resume_from: Path to a saved model to resume from (optional)

    Returns:
        Tuple of (trained model, info dict with vec_env)
    """
    # Default logging keywords for Pursuit-Evasion: capture success metrics
    if log_info_keys is None:
        log_info_keys = ("evader_captured", "min_distance_to_evader")

    return run_training(
        env,
        embed_config,
        algo_params,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        log_info_keys=log_info_keys,
        n_envs=n_envs,
        save_path=save_path,
        resume_from=resume_from,
    )
