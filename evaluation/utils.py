"""
Evaluation utilities for scale-invariant multi-agent RL.

This module provides functions for:
- Loading trained models
- Evaluating policies on different swarm sizes
- Computing evaluation metrics
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from stable_baselines3 import PPO

from environments.rendezvous.rendezvous_env import RendezvousEnv


def load_model(
    model_path: str,
    verbose: bool = True,
) -> PPO:
    """Load a trained PPO model.

    Args:
        model_path: Path to the trained model (.zip file)
        verbose: Whether to print loading information

    Returns:
        Loaded PPO model
    """
    model_path = Path(model_path)

    if verbose:
        print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    if verbose:
        print("✓ Model loaded successfully")

    return model


def evaluate_policy(
    model: PPO,
    env: RendezvousEnv,
    n_episodes: int = 100,
    deterministic: bool = True,
    render: bool = False,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate a trained policy on an environment.

    Args:
        model: Trained PPO model
        env: Environment to evaluate on
        n_episodes: Number of episodes to evaluate
        deterministic: Use deterministic actions
        render: Whether to render the environment
        verbose: Whether to print progress

    Returns:
        Dictionary with evaluation metrics
    """
    episode_rewards = []
    episode_lengths = []
    final_max_distances = []
    final_mean_distances = []
    success_count = 0

    for episode in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        done = False

        while not done:
            actions = {}
            for agent_name in env.agents:
                agent_obs = obs[agent_name]
                agent_action, _ = model.predict(agent_obs[np.newaxis, :], deterministic=deterministic)
                actions[agent_name] = agent_action[0]

            obs, rewards, terminations, truncations, infos = env.step(actions)
            episode_reward += sum(rewards.values()) / len(rewards)
            done = any(terminations.values()) or any(truncations.values())
            episode_length += 1

            if render:
                env.render()

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)

        if infos:
            first_agent_info = list(infos.values())[0]
            final_max_distances.append(first_agent_info.get("max_pairwise_distance", np.nan))
            final_mean_distances.append(first_agent_info.get("distance_to_com", np.nan))
        else:
            final_max_distances.append(np.nan)
            final_mean_distances.append(np.nan)

        if any(terminations.values()):
            success_count += 1

        if verbose and (episode + 1) % 10 == 0:
            print(f"    Episode {episode + 1}/{n_episodes} completed")

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
    n_episodes: int = 100,
    deterministic: bool = True,
    verbose: bool = True,
) -> Dict[int, Dict[str, float]]:
    """Evaluate a policy on multiple swarm sizes.

    Args:
        model: Trained PPO model
        test_sizes: List of swarm sizes to test
        env_config: Environment configuration dict
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

        env = RendezvousEnv(num_agents=num_agents, **env_config)

        results = evaluate_policy(
            model=model,
            env=env,
            n_episodes=n_episodes,
            deterministic=deterministic,
            render=False,
            verbose=verbose,
        )

        all_results[num_agents] = results

        if verbose:
            print(f"\n  Results for {num_agents} agents:")
            print(f"    Mean reward: {results['mean_reward']:.3f} ± {results['std_reward']:.3f}")
            print(f"    Mean length: {results['mean_length']:.1f} ± {results['std_length']:.1f}")
            print(f"    Final max dist: {results['mean_final_max_dist']:.3f} ± {results['std_final_max_dist']:.3f}")
            print(f"    Success rate: {results['success_rate']:.1%}")

        env.close()

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
