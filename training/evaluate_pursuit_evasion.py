"""
Evaluation script for trained Pursuit-Evasion models.

Supports rendering, multi-episode evaluation, and metrics collection.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from environments.pursuit.pursuit_evasion_env import PursuitEvasionEnv


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate a trained Pursuit-Evasion model")

    # Model parameters
    parser.add_argument("--model-path", type=str, required=True, help="Path to the trained model (.zip file)")

    # Environment parameters
    parser.add_argument("--num-pursuers", type=int, default=10, help="Number of pursuer agents")
    parser.add_argument("--world-size", type=float, default=10.0, help="Side length of the square world")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument("--capture-radius", type=float, default=0.5, help="Capture distance threshold")
    parser.add_argument("--evader-speed", type=float, default=1.0, help="Evader speed")
    parser.add_argument(
        "--obs-model",
        type=str,
        default="global_basic",
        choices=["global_basic", "global_extended", "local_basic", "local_extended"],
        help="Observation model",
    )
    parser.add_argument("--comm-radius", type=float, default=None, help="Communication radius for local observations")
    parser.add_argument(
        "--evader-strategy",
        type=str,
        default="voronoi_center",
        choices=["simple", "max_min_distance", "weighted_escape", "voronoi_center"],
        help="Evader evasion strategy",
    )
    parser.add_argument("--kinematics", type=str, default="single", choices=["single", "double"], help="Agent kinematics")

    # Evaluation parameters
    parser.add_argument("--n-episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic policy (no exploration)")
    parser.add_argument(
        "--render-mode",
        type=str,
        default="human",
        choices=["human", "rgb_array", None],
        help="Rendering mode",
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for rendering")

    return parser.parse_args()


def load_model(model_path: str) -> PPO:
    """Load a trained model."""
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"Loading model from {model_path}...")
    model = PPO.load(model_path)
    print("✓ Model loaded successfully\n")
    return model


def evaluate_episode(
    model: PPO,
    env: PursuitEvasionEnv,
    episode_num: int,
    deterministic: bool = True,
    render: bool = True,
) -> dict:
    """Run a single evaluation episode."""
    obs, info = env.reset()
    episode_reward = 0.0
    episode_length = 0
    capture_occurred = False
    min_distances = []

    while True:
        # Get actions from model
        actions = {}
        for agent in env.agents:
            agent_obs = obs[agent]
            action, _ = model.predict(agent_obs[np.newaxis, :], deterministic=deterministic)
            actions[agent] = action[0]

        # Step environment
        obs, rewards, terminations, truncations, infos = env.step(actions)
        episode_reward += sum(rewards.values()) / len(rewards)
        episode_length += 1

        # Track metrics
        if infos:
            first_info = list(infos.values())[0]
            min_dist = first_info.get("min_distance_to_evader", np.nan)
            min_distances.append(min_dist)
            if first_info.get("evader_captured", False):
                capture_occurred = True

        # Render if requested
        if render:
            env.render()

        # Check termination
        if any(terminations.values()) or any(truncations.values()):
            break

    env.close()

    return {
        "episode": episode_num,
        "episode_reward": episode_reward,
        "episode_length": episode_length,
        "capture_occurred": capture_occurred,
        "min_distance_mean": float(np.nanmean(min_distances)) if min_distances else np.nan,
        "min_distance_min": float(np.nanmin(min_distances)) if min_distances else np.nan,
    }


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Load model
    model = load_model(args.model_path)

    print(f"{'=' * 70}")
    print(f"Evaluating Pursuit-Evasion Policy")
    print(f"{'=' * 70}")
    print(f"Model: {args.model_path}")
    print(f"Pursuers: {args.num_pursuers}")
    print(f"World Size: {args.world_size}")
    print(f"Observation Model: {args.obs_model}")
    print(f"Evader Strategy: {args.evader_strategy}")
    print(f"Episodes: {args.n_episodes}")
    print(f"Deterministic: {args.deterministic}")
    print(f"Render Mode: {args.render_mode}")
    print(f"{'=' * 70}\n")

    # Evaluate multiple episodes
    results = []
    for episode in range(args.n_episodes):
        print(f"Running episode {episode + 1}/{args.n_episodes}...")

        env = PursuitEvasionEnv(
            num_pursuers=args.num_pursuers,
            world_size=args.world_size,
            max_steps=args.max_steps,
            capture_radius=args.capture_radius,
            evader_speed=args.evader_speed,
            obs_model=args.obs_model,
            comm_radius=args.comm_radius,
            evader_strategy=args.evader_strategy,
            kinematics=args.kinematics,
            render_mode=args.render_mode,
            fps=args.fps,
        )

        episode_result = evaluate_episode(
            model=model,
            env=env,
            episode_num=episode + 1,
            deterministic=args.deterministic,
            render=args.render_mode == "human",
        )

        results.append(episode_result)
        print(
            f"  Episode {episode_result['episode']}: "
            f"Reward={episode_result['episode_reward']:.3f}, "
            f"Length={episode_result['episode_length']}, "
            f"Captured={episode_result['capture_occurred']}"
        )

    # Print summary
    print(f"\n{'=' * 70}")
    print("EVALUATION SUMMARY")
    print(f"{'=' * 70}")

    episode_rewards = [r["episode_reward"] for r in results]
    episode_lengths = [r["episode_length"] for r in results]
    capture_rate = sum(r["capture_occurred"] for r in results) / len(results)
    min_distances = [r["min_distance_mean"] for r in results if not np.isnan(r["min_distance_mean"])]

    print(f"Mean Reward: {np.mean(episode_rewards):.3f} ± {np.std(episode_rewards):.3f}")
    print(f"Mean Length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Capture Rate: {capture_rate:.1%}")
    if min_distances:
        print(f"Mean Min Distance: {np.mean(min_distances):.3f} ± {np.std(min_distances):.3f}")

    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
