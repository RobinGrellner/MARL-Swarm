from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np

from environments.rendezvous.rendezvous_env import RendezvousEnv

from gymnasium import spaces
from supersuit import flatten_v0 as supersuit_flatten
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from policies.meanEmbeddingExtractor import MeanEmbeddingExtractor

import torch
import torch.nn as nn

# Suppress render_mode warning from VecEnv
warnings.filterwarnings("ignore", message=".*render_mode.*", category=UserWarning)

def wrap_env_for_sb3(env: RendezvousEnv, *, n_envs: int = 1, monitor_keywords: Optional[Tuple[str, ...]] = None, normalize: bool = True) -> VecMonitor:
    """Wrap a PettingZoo environment for Stable-Baselines3 training.

    The wrapping sequence is as follows:

    1. Per Agent obs into single vector
    2. Convert the flattened PettingZoo env into a vectorised environment with pettingzoo_env_to_vec_env_v1.
    3. Concatenate n_envs copies of the vector environment into a single SB3 VecEnv with concat_vec_envs_v1.
    4. Apply a VecMonitor to record episode statistics and optionally monitor additional keys from the environment's info.
    5. Optionally normalise observations and rewards via VecNormalize.

    :param env: the PettingZoo environment to wrap
    :param n_envs: number of vectorised copies (for parallel workers)
    :param monitor_keywords: optional tuple of keys from env.info to record
    :param normalize: whether to apply observation and reward normalisation
    :return: a wrapped ``VecEnv`` ready for SB3
    """
    # Flatten PettingZoo observations to 1D per agent
    env_flat = supersuit_flatten(env)
    # Convert to a SuperSuit vector env
    vec_env = pettingzoo_env_to_vec_env_v1(env_flat)
    # Concatenate into an SB3 VecEnv; base_class ensures correct wrappers
    vec_env = concat_vec_envs_v1(vec_env, num_vec_envs=n_envs, num_cpus=0, base_class="stable_baselines3")
    # Apply VecMonitor to log episode statistics
    if monitor_keywords is not None:
        vec_env = VecMonitor(vec_env, info_keywords=monitor_keywords)
    else:
        vec_env = VecMonitor(vec_env)
    # Optionally apply normalisation
    if normalize:
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    return vec_env


#TODO ab hier nachprüfen
def make_policy_kwargs(layout: Dict[str, int], *, embed_dim: int = 64, phi_layers: int = 1) -> Dict[str, Any]:
    local_dim = layout["local_dim"]
    neigh_dim = layout["neigh_dim"]
    max_neigh = layout["max_neighbours"]
    # Determine hidden layer configuration: repeat 64 for the specified depth
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
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),  # Fixed: removed list wrapper
    }


def setup_ppo_model(vec_env: VecMonitor, policy_kwargs: Dict[str, Any], ppo_params: Dict[str, Any]) -> PPO:
    # Force GPU usage if available (unless explicitly set in ppo_params)
    ppo_params = ppo_params.copy()  # Don't modify original dict

    if "device" not in ppo_params:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        ppo_params["device"] = device
        print(f"\n{'='*60}")
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
        print(f"Selected Device: {device}")
        print(f"{'='*60}\n")

    return PPO(
        policy="MlpPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        **ppo_params,
    )


def run_training_rendezvous(
    env: RendezvousEnv,
    embed_config: Dict[str, Any],
    ppo_params: Dict[str, Any],
    *,
    total_timesteps: int = 200_000,
    log_info_keys: Optional[Tuple[str, ...]] = None,
    n_envs: int = 1,
    normalize: bool = True,
    save_path: Optional[str] = None,
) -> Tuple[PPO, Dict[str, Any]]:
    """Train a PPO model on RendezvousEnv with mean embedding.

    Args:
        env: RendezvousEnv instance
        embed_config: Configuration for mean embedding extractor
        ppo_params: PPO hyperparameters
        total_timesteps: Total training timesteps
        log_info_keys: Keys to log from env info dict
        n_envs: Number of parallel environments
        normalize: Whether to apply VecNormalize
        save_path: Path to save model and VecNormalize stats (optional)

    Returns:
        Tuple of (trained model, info dict with vec_env if normalize=True)
    """
    # 1. Extract layout before wrapping (avoid wrapper hiding attributes)
    layout = env.obs_layout
    # 2. Wrap environment into vector form
    vec_env = wrap_env_for_sb3(env, n_envs=n_envs, monitor_keywords=log_info_keys, normalize=normalize)
    # 3. Build policy kwargs from layout and embedding configuration
    policy_kwargs = make_policy_kwargs(layout, embed_dim=embed_config.get("embed_dim", 64), phi_layers=embed_config.get("phi_layers", 1))
    # 4. Merge default PPO parameters with user provided values
    default_ppo_params = {
        "learning_rate": 3e-4,
        "n_steps": 128,
        "batch_size": 256,
        "n_epochs": 10,
        "ent_coef": 0.01,
        "vf_coef": 0.7,
        "clip_range": 0.2,
        "target_kl": 0.02,
        # "device" is auto-selected in setup_ppo_model (cuda if available, else cpu)
        "verbose": 1,
    }
    # Update defaults with supplied parameters
    default_ppo_params.update(ppo_params)
    # 5. Create model
    model = setup_ppo_model(vec_env, policy_kwargs, default_ppo_params)
    # 6. Train model
    model.learn(total_timesteps=total_timesteps)

    # 7. Save model and VecNormalize stats if path provided
    if save_path:
        model.save(save_path)
        print(f"Model saved to {save_path}")

        # Save VecNormalize statistics for proper evaluation
        if normalize:
            vecnorm_path = save_path.replace(".zip", "_vecnormalize.pkl")
            vec_env.save(vecnorm_path)
            print(f"VecNormalize stats saved to {vecnorm_path}")

    # Collect diagnostic info
    info = {
        "layout": layout,
        "embed_config": embed_config,
        "ppo_params": default_ppo_params,
        "vec_env": vec_env if normalize else None,  # Include vec_env for manual saving
    }
    return model, info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a configurable PPO training on RendezvousEnv")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of agents")
    parser.add_argument("--world-size", type=float, default=10.0, help="Side length of the world")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--obs-model", type=str, default="local_basic", choices=[
        "global_basic", "global_extended", "local_basic", "local_extended", "local_comm", "classic"
    ], help="Observation model to use")
    parser.add_argument("--comm-radius", type=float, default=None, help="Communication radius (None defaults to world size)")
    parser.add_argument("--kinematics", type=str, default="single", choices=["single", "double"], help="Kinematic model")
    parser.add_argument("--max-agents", type=int, default=None, help="Maximum agents for scale‑invariant observation size")
    parser.add_argument("--embed-dim", type=int, default=64, help="Dimensionality of the mean embedding")
    parser.add_argument("--phi-layers", type=int, default=1, help="Number of hidden layers in the φ network")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for PPO")
    parser.add_argument("--n-steps", type=int, default=128, help="Rollout length per environment")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs per update")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on (cpu/cuda)")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel vector environments")
    parser.add_argument("--no-normalize", action="store_true", help="Disable observation and reward normalisation")
    parser.add_argument("--log-info", nargs="*", default=None, help="Info keys to log via VecMonitor")
    parser.add_argument("--model-path", type=str, default="models/rendezvous_model.zip", help="Path to save the trained model")
    args = parser.parse_args()

    # Build environment configuration
    env_params = {
        "num_agents": args.num_agents,
        "world_size": args.world_size,
        "max_steps": args.max_steps,
        "obs_model": args.obs_model,
        "comm_radius": args.comm_radius,
        "kinematics": args.kinematics,
    }
    embed_config = {
        "embed_dim": args.embed_dim,
        "phi_layers": args.phi_layers,
    }
    ppo_params = {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "n_epochs": args.n_epochs,
        "device": args.device,
    }
    
    env = RendezvousEnv(**env_params)

    # Run training
    model, info = run_training_rendezvous(
        env,
        embed_config,
        ppo_params,
        total_timesteps=args.total_timesteps,
        log_info_keys=tuple(args.log_info) if args.log_info else None,
        n_envs=args.n_envs,
        normalize=not args.no_normalize,
        save_path=args.model_path,
    )

    # Output basic summary
    print("\nTraining completed!")
    print(f"Model saved to: {args.model_path}")
    if not args.no_normalize:
        vecnorm_path = args.model_path.replace(".zip", "_vecnormalize.pkl")
        print(f"VecNormalize stats saved to: {vecnorm_path}")
