"""Evaluate trained rendezvous policy."""

import argparse
import numpy as np
from environments.rendezvous.rendezvous_env import RendezvousEnv
from stable_baselines3 import PPO


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent on the Rendezvous environment")
    parser.add_argument("--model-path", type=str, default="models/rendezvous_trained.zip", help="Path to trained model")
    parser.add_argument("--num-agents", type=int, default=6, help="Number of agents in the environment")
    parser.add_argument("--world-size", type=float, default=10.0, help="Side length of the square world")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument(
        "--obs-model",
        type=str,
        default="local_basic",
        choices=["global_basic", "global_extended", "local_basic", "local_extended", "local_comm", "classic"],
        help="Observation model to use",
    )
    parser.add_argument(
        "--comm-radius", type=float, default=8.0, help="Communication radius for local observation models"
    )
    parser.add_argument("--torus", action="store_true", help="Whether to wrap around the world boundaries")
    parser.add_argument(
        "--break-distance-threshold", type=float, default=None, help="Early termination threshold on pairwise distances"
    )
    parser.add_argument(
        "--kinematics", type=str, default="single", choices=["single", "double"], help="Agent kinematic model"
    )
    parser.add_argument("--v-max", type=float, default=1.0, help="Maximum linear velocity for agents")
    parser.add_argument("--omega-max", type=float, default=1.0, help="Maximum angular velocity for agents")
    parser.add_argument(
        "--max-agents",
        type=int,
        default=None,
        help="Maximum number of agents to size the observation space for scale invariance",
    )
    parser.add_argument("--n-episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument(
        "--render-mode", type=str, default=None, choices=["human", "rgb_array", None], help="Rendering mode"
    )
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for rendering")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    """Run evaluation episodes with trained policy."""
    env = RendezvousEnv(
        num_agents=args.num_agents,
        world_size=args.world_size,
        max_steps=args.max_steps,
        obs_model=args.obs_model,
        comm_radius=args.comm_radius,
        torus=args.torus,
        break_distance_threshold=args.break_distance_threshold,
        kinematics=args.kinematics,
        v_max=args.v_max,
        omega_max=args.omega_max,
        max_agents=args.max_agents,
        render_mode=args.render_mode,
        fps=args.fps,
    )

    print(f"Loading model from {args.model_path}...")
    model = PPO.load(args.model_path, device="cpu")
    print("Model loaded successfully!")

    episode_returns = []
    episode_lengths = []
    episode_mean_distances = []

    for ep in range(args.n_episodes):
        obs, infos = env.reset()
        ep_ret = 0.0
        step_rewards = []
        step_count = 0
        done = False

        if args.render_mode == "human":
            print(f"\n{'=' * 50}")
            print(f"Episode {ep + 1}/{args.n_episodes}")
            print(f"{'=' * 50}")

        while not done:
            actions = {a: model.predict(obs[a], deterministic=args.deterministic)[0] for a in env.agents}
            obs, rewards, term, trunc, infos = env.step(actions)
            done = all(term[a] or trunc[a] for a in env.agents)

            if args.render_mode:
                env.render()

            r_t = float(np.mean(list(rewards.values())))
            ep_ret += r_t
            step_rewards.append(r_t)
            step_count += 1

        step_mean = float(np.mean(step_rewards)) if step_rewards else 0.0
        d_c = getattr(env, "dc", env.world_size)
        mean_pairwise_est = -step_mean * d_c

        episode_returns.append(ep_ret)
        episode_lengths.append(step_count)
        episode_mean_distances.append(mean_pairwise_est)

        if args.render_mode == "human":
            print(f"\nEpisode {ep + 1} complete:")
            print(f"  Total steps: {step_count}")
            print(f"  Episode return: {ep_ret:.2f}")
            print(f"  Mean step reward: {step_mean:.3f}")
            print(f"  Est. mean pairwise distance: {mean_pairwise_est:.2f}")

    env.close()

    # Print summary statistics
    print(f"\n{'=' * 50}")
    print("Evaluation Summary")
    print(f"{'=' * 50}")
    print(f"Episodes evaluated: {args.n_episodes}")
    print(f"Mean return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Mean pairwise distance: {np.mean(episode_mean_distances):.2f} ± {np.std(episode_mean_distances):.2f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
