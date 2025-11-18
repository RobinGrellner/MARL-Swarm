"""
Evaluation utilities for scale-invariant multi-agent RL.

This module provides functions for:
- Loading trained models with proper VecNormalize handling
- Evaluating policies on different swarm sizes
- Computing evaluation metrics
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from supersuit import concat_vec_envs_v1, flatten_v0, pettingzoo_env_to_vec_env_v1

from environments.rendezvous.rendezvous_env import RendezvousEnv


def load_model_with_normalization(
    model_path: str,
    vecnormalize_path: Optional[str] = None,
    verbose: bool = True,
) -> Tuple[PPO, Optional[Path]]:
    """Load a trained PPO model and its VecNormalize statistics.

    Args:
        model_path: Path to the trained model (.zip file)
        vecnormalize_path: Path to VecNormalize stats (.pkl). If None, infers from model_path
        verbose: Whether to print loading information

    Returns:
        Tuple of (loaded PPO model, Path to VecNormalize stats if exists, else None)
    """
    model_path = Path(model_path)

    # Infer VecNormalize path if not provided
    if vecnormalize_path is None:
        vecnormalize_path = str(model_path).replace(".zip", "_vecnormalize.pkl")
    vecnormalize_path = Path(vecnormalize_path)

    # Load model
    if verbose:
        print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    if verbose:
        print("✓ Model loaded successfully")

    # Check for VecNormalize stats
    if vecnormalize_path.exists():
        if verbose:
            print(f"✓ VecNormalize stats found: {vecnormalize_path}")
        return model, vecnormalize_path
    else:
        if verbose:
            print(f"⚠ VecNormalize stats NOT found: {vecnormalize_path}")
            print("  Evaluation will proceed WITHOUT normalization")
        return model, None


def create_eval_env_with_normalization(
    env: RendezvousEnv,
    vecnormalize_path: Optional[Path] = None,
    verbose: bool = True,
) -> Tuple[VecNormalize, RendezvousEnv]:
    """Wrap environment with VecNormalize using frozen training statistics.

    Args:
        env: RendezvousEnv instance
        vecnormalize_path: Path to VecNormalize stats. If None, returns env without normalization
        verbose: Whether to print information

    Returns:
        Tuple of (VecNormalize wrapper with frozen stats, original env)
    """
    if vecnormalize_path is None:
        if verbose:
            print("  Evaluating without VecNormalize")
        return None, env

    # Wrap environment for SB3
    env_flat = flatten_v0(env)
    vec_env = pettingzoo_env_to_vec_env_v1(env_flat)
    vec_env = concat_vec_envs_v1(vec_env, num_vec_envs=1, num_cpus=0, base_class="stable_baselines3")
    vec_env = VecMonitor(vec_env)

    # Load VecNormalize with frozen statistics
    vec_env = VecNormalize.load(vecnormalize_path, vec_env)
    vec_env.training = False  # CRITICAL: Freeze statistics from training
    vec_env.norm_reward = False  # Don't normalize rewards during evaluation

    if verbose:
        print("  ✓ VecNormalize loaded with FROZEN training statistics")

    return vec_env, env


def evaluate_policy(
    model: PPO,
    env: RendezvousEnv,
    vec_env: Optional[VecNormalize] = None,
    n_episodes: int = 100,
    deterministic: bool = True,
    render: bool = False,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate a trained policy on an environment.

    Args:
        model: Trained PPO model
        env: Environment to evaluate on (unwrapped)
        vec_env: VecNormalize wrapper with frozen statistics (if used during training)
        n_episodes: Number of episodes to evaluate
        deterministic: Use deterministic actions (policy mode) vs stochastic (sampling)
        render: Whether to render the environment
        verbose: Whether to print progress

    Returns:
        Dictionary with evaluation metrics:
            - mean_reward: Average episode reward
            - std_reward: Standard deviation of episode rewards
            - mean_length: Average episode length
            - std_length: Standard deviation of episode lengths
            - mean_final_max_dist: Average final maximum pairwise distance
            - std_final_max_dist: Std of final maximum pairwise distance
            - mean_final_mean_dist: Average final distance to center of mass
            - success_rate: Fraction of episodes that terminated successfully
            - n_episodes: Number of episodes evaluated
    """
    episode_rewards = []
    episode_lengths = []
    final_max_distances = []
    final_mean_distances = []
    success_count = 0

    for episode in range(n_episodes):
        # Reset environment
        if vec_env is not None:
            obs = vec_env.reset()
        else:
            obs, _ = env.reset()

        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            # Get action from policy
            action, _ = model.predict(obs, deterministic=deterministic)

            # Step environment
            if vec_env is not None:
                obs, reward, done_array, info = vec_env.step(action)
                done = done_array[0]  # VecEnv returns array, get first element
                episode_reward += reward[0]  # VecEnv returns array
            else:
                # For unwrapped env, we need to handle dict observations
                # This branch is for when VecNormalize is not used
                actions = {}
                for i, agent_name in enumerate(env.agents):
                    agent_obs = obs[agent_name]
                    agent_action, _ = model.predict(agent_obs[np.newaxis, :], deterministic=deterministic)
                    actions[agent_name] = agent_action[0]

                obs, rewards, terminations, truncations, infos = env.step(actions)
                episode_reward += sum(rewards.values()) / len(rewards)  # Average reward
                done = any(terminations.values()) or any(truncations.values())

            episode_length += 1

            if render:
                env.render()

        # Record episode statistics
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        # Get final state metrics from info
        if vec_env is not None:
            # VecEnv returns list of info dicts
            # After wrapping, info might be stored differently
            try:
                if len(info) > 0:
                    env_info = info[0]
                    # Check if this is a dict with agent keys or episode info
                    if isinstance(env_info, dict):
                        # Try to extract metrics directly or from nested structure
                        if "max_pairwise_distance" in env_info:
                            final_max_distances.append(env_info["max_pairwise_distance"])
                            final_mean_distances.append(env_info.get("distance_to_com", np.nan))
                        elif len(env_info) > 0:
                            # Might be agent-keyed, get first agent's info
                            first_value = list(env_info.values())[0]
                            if isinstance(first_value, dict):
                                final_max_distances.append(first_value.get("max_pairwise_distance", np.nan))
                                final_mean_distances.append(first_value.get("distance_to_com", np.nan))
                            else:
                                # No valid metrics available
                                final_max_distances.append(np.nan)
                                final_mean_distances.append(np.nan)
                        else:
                            final_max_distances.append(np.nan)
                            final_mean_distances.append(np.nan)
                    else:
                        final_max_distances.append(np.nan)
                        final_mean_distances.append(np.nan)
                else:
                    final_max_distances.append(np.nan)
                    final_mean_distances.append(np.nan)
            except (AttributeError, KeyError, IndexError) as e:
                # Fallback: compute metrics from environment state
                try:
                    # Get the unwrapped environment from VecNormalize
                    unwrapped_env = vec_env.venv.envs[0].par_env
                    if hasattr(unwrapped_env, 'agent_handler'):
                        # Compute metrics directly
                        positions = unwrapped_env.agent_handler.positions
                        max_dist = np.max(np.linalg.norm(positions[:, None] - positions[None, :], axis=2))
                        com = np.mean(positions, axis=0)
                        mean_dist = np.mean(np.linalg.norm(positions - com, axis=1))
                        final_max_distances.append(float(max_dist))
                        final_mean_distances.append(float(mean_dist))
                    else:
                        final_max_distances.append(np.nan)
                        final_mean_distances.append(np.nan)
                except:
                    final_max_distances.append(np.nan)
                    final_mean_distances.append(np.nan)
        else:
            # Unwrapped env
            if infos:
                first_agent_info = list(infos.values())[0]
                final_max_distances.append(first_agent_info.get("max_pairwise_distance", np.nan))
                final_mean_distances.append(first_agent_info.get("distance_to_com", np.nan))
            else:
                final_max_distances.append(np.nan)
                final_mean_distances.append(np.nan)

        # Check success
        if vec_env is not None:
            # Check if episode terminated successfully
            if done and episode_length < env.max_steps:
                success_count += 1
        else:
            if any(terminations.values()):
                success_count += 1

        if verbose and (episode + 1) % 10 == 0:
            print(f"    Episode {episode + 1}/{n_episodes} completed")

    # Compute statistics
    results = {
        "mean_reward": float(np.mean(episode_rewards)),
        "std_reward": float(np.std(episode_rewards)),
        "mean_length": float(np.mean(episode_lengths)),
        "std_length": float(np.std(episode_lengths)),
        "mean_final_max_dist": float(np.nanmean(final_max_distances)),
        "std_final_max_dist": float(np.nanstd(final_max_distances)),
        "mean_final_mean_dist": float(np.nanmean(final_mean_distances)),
        "std_final_mean_dist": float(np.nanstd(final_mean_distances)),
        "success_rate": float(success_count / n_episodes),
        "n_episodes": n_episodes,
    }

    return results


def evaluate_on_multiple_sizes(
    model: PPO,
    test_sizes: List[int],
    env_config: Dict,
    vecnormalize_path: Optional[Path] = None,
    n_episodes: int = 100,
    deterministic: bool = True,
    verbose: bool = True,
) -> Dict[int, Dict[str, float]]:
    """Evaluate a policy on multiple swarm sizes.

    Args:
        model: Trained PPO model
        test_sizes: List of swarm sizes to test (e.g., [4, 8, 16, 32, 50])
        env_config: Environment configuration dict (world_size, obs_model, etc.)
        vecnormalize_path: Path to VecNormalize stats
        n_episodes: Number of episodes per swarm size
        deterministic: Use deterministic policy
        verbose: Print progress

    Returns:
        Dictionary mapping swarm size to evaluation results
    """
    all_results = {}

    for num_agents in test_sizes:
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Evaluating on {num_agents} agents...")
            print(f"{'=' * 60}")

        # Create environment with this swarm size
        env = RendezvousEnv(num_agents=num_agents, **env_config)

        # Wrap with VecNormalize if stats are available
        vec_env, env = create_eval_env_with_normalization(env, vecnormalize_path, verbose=verbose)

        # Evaluate
        results = evaluate_policy(
            model=model,
            env=env,
            vec_env=vec_env,
            n_episodes=n_episodes,
            deterministic=deterministic,
            render=False,
            verbose=verbose,
        )

        # Store results
        all_results[num_agents] = results

        # Print summary
        if verbose:
            print(f"\n  Results for {num_agents} agents:")
            print(f"    Mean reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
            print(f"    Mean length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
            print(f"    Final max dist: {results['mean_final_max_dist']:.3f} ± {results['std_final_max_dist']:.3f}")
            print(f"    Success rate: {results['success_rate']:.1%}")

        # Clean up
        env.close()
        if vec_env is not None:
            vec_env.close()

    return all_results


def print_scalability_summary(results: Dict[int, Dict[str, float]]) -> None:
    """Print a formatted summary table of scalability results.

    Args:
        results: Dictionary mapping swarm size to evaluation metrics
    """
    print(f"\n{'=' * 80}")
    print("SCALABILITY SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Agents':<10} {'Mean Reward':<20} {'Success Rate':<15} {'Final Max Dist':<20}")
    print("-" * 80)

    for num_agents in sorted(results.keys()):
        r = results[num_agents]
        reward_str = f"{r['mean_reward']:>7.3f} ± {r['std_reward']:<6.3f}"
        success_str = f"{r['success_rate']:>6.1%}"
        dist_str = f"{r['mean_final_max_dist']:>7.3f} ± {r['std_final_max_dist']:<6.3f}"
        print(f"{num_agents:<10} {reward_str:<20} {success_str:<15} {dist_str:<20}")
