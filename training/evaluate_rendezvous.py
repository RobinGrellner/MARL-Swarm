"""Evaluate trained rendezvous policy."""
import argparse
import numpy as np
from environments.rendezvous.rendezvous_env import RendezvousEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize
from supersuit import flatten_v0, pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1
import os


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained PPO agent on the Rendezvous environment")
    parser.add_argument("--model-path", type=str, default="models/rendezvous_trained.zip", help="Path to trained model")
    parser.add_argument("--vecnormalize-path", type=str, default=None, help="Path to VecNormalize stats (auto-detected if not specified)")
    parser.add_argument("--num-agents", type=int, default=6, help="Number of agents in the environment")
    parser.add_argument("--world-size", type=float, default=10.0, help="Side length of the square world")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--obs-model", type=str, default="local_basic", choices=[
        "global_basic", "global_extended", "local_basic", "local_extended", "local_comm", "classic"
    ], help="Observation model to use")
    parser.add_argument("--comm-radius", type=float, default=8.0, help="Communication radius for local observation models")
    parser.add_argument("--torus", action="store_true", help="Whether to wrap around the world boundaries")
    parser.add_argument("--break-distance-threshold", type=float, default=None, help="Early termination threshold on pairwise distances")
    parser.add_argument("--kinematics", type=str, default="single", choices=["single", "double"], help="Agent kinematic model")
    parser.add_argument("--v-max", type=float, default=1.0, help="Maximum linear velocity for agents")
    parser.add_argument("--omega-max", type=float, default=1.0, help="Maximum angular velocity for agents")
    parser.add_argument("--max-agents", type=int, default=None, help="Maximum number of agents to size the observation space for scale invariance")
    parser.add_argument("--n-episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--render-mode", type=str, default=None, choices=["human", "rgb_array", None], help="Rendering mode")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second for rendering")
    parser.add_argument("--deterministic", action="store_true", default=True, help="Use deterministic actions")
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> None:
    """Run evaluation episodes with trained policy."""
    # Create environment matching training configuration
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

    # Auto-detect VecNormalize path if not provided
    vecnorm_path = args.vecnormalize_path
    if vecnorm_path is None:
        vecnorm_path = args.model_path.replace(".zip", "_vecnormalize.pkl")

    # Check if VecNormalize stats exist
    use_vecnormalize = os.path.exists(vecnorm_path)

    if use_vecnormalize:
        print(f"Loading VecNormalize stats from {vecnorm_path}...")
        # Need to wrap PettingZoo env same way as training
        env_wrapped = flatten_v0(env)
        vec_env_base = pettingzoo_env_to_vec_env_v1(env_wrapped)
        vec_env_base = concat_vec_envs_v1(vec_env_base, num_vec_envs=1, num_cpus=0, base_class="stable_baselines3")

        # Load VecNormalize stats
        vec_env = VecNormalize.load(vecnorm_path, vec_env_base)
        # IMPORTANT: Set training=False and norm_reward=False for evaluation
        vec_env.training = False
        vec_env.norm_reward = False
        print("VecNormalize loaded! Observations will be normalized.")
    else:
        print(f"⚠️  VecNormalize stats not found at {vecnorm_path}")
        print("⚠️  Evaluating WITHOUT normalization - performance may be poor!")
        vec_env = None

    # Load trained model
    print(f"Loading model from {args.model_path}...")
    model = PPO.load(args.model_path, device="cpu")
    print("Model loaded successfully!")

    episode_returns = []
    episode_lengths = []
    episode_mean_distances = []

    for ep in range(args.n_episodes):
        # Reset base environment for rendering
        env.reset()

        if use_vecnormalize:
            # Reset vec_env
            vec_obs = vec_env.reset()
        else:
            obs, infos = env.reset()

        ep_ret = 0.0
        step_rewards = []
        step_count = 0
        done = False

        if args.render_mode == "human":
            print(f"\n{'='*50}")
            print(f"Episode {ep+1}/{args.n_episodes}")
            print(f"{'='*50}")

        while not done:
            if use_vecnormalize:
                # Get actions from policy using normalized observations
                actions_vec, _ = model.predict(vec_obs, deterministic=args.deterministic)

                # Convert vectorized actions back to dict format for base env
                # vec_obs has shape (num_agents,) for our case
                actions = {}
                for i, agent_name in enumerate(env.agents):
                    actions[agent_name] = actions_vec[i]
            else:
                # Get actions from trained policy (without normalization)
                actions = {a: model.predict(obs[a], deterministic=args.deterministic)[0] for a in env.agents}

            # Step the base environment (for rendering and rewards)
            obs, rewards, term, trunc, infos = env.step(actions)

            if use_vecnormalize:
                # Also step vec_env to keep it synchronized
                vec_obs, vec_rewards, vec_dones, vec_infos = vec_env.step(actions_vec)
                done = vec_dones[0]
            else:
                done = all(term[a] or trunc[a] for a in env.agents)

            if args.render_mode:
                env.render()

            # Calculate episode statistics
            r_t = float(np.mean(list(rewards.values())))
            ep_ret += r_t
            step_rewards.append(r_t)
            step_count += 1

        step_mean = float(np.mean(step_rewards)) if step_rewards else 0.0
        # Estimate mean pairwise distance
        d_c = getattr(env, "dc", env.world_size)
        mean_pairwise_est = -step_mean * d_c

        episode_returns.append(ep_ret)
        episode_lengths.append(step_count)
        episode_mean_distances.append(mean_pairwise_est)

        if args.render_mode == "human":
            print(f"\nEpisode {ep+1} complete:")
            print(f"  Total steps: {step_count}")
            print(f"  Episode return: {ep_ret:.2f}")
            print(f"  Mean step reward: {step_mean:.3f}")
            print(f"  Est. mean pairwise distance: {mean_pairwise_est:.2f}")

    env.close()
    if use_vecnormalize:
        vec_env.close()

    # Print summary statistics
    print(f"\n{'='*50}")
    print("Evaluation Summary")
    print(f"{'='*50}")
    print(f"Episodes evaluated: {args.n_episodes}")
    print(f"Mean return: {np.mean(episode_returns):.2f} ± {np.std(episode_returns):.2f}")
    print(f"Mean episode length: {np.mean(episode_lengths):.1f} ± {np.std(episode_lengths):.1f}")
    print(f"Mean pairwise distance: {np.mean(episode_mean_distances):.2f} ± {np.std(episode_mean_distances):.2f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)
