from __future__ import annotations

import argparse
from datetime import datetime

from environments.rendezvous.rendezvous_env import RendezvousEnv
from training.rendezvous_train_utils import run_training_rendezvous
from training.common_train_utils import add_common_training_args, build_algo_params, build_embed_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train an RL agent on the Rendezvous environment")

    # Environment parameters
    parser.add_argument("--num-agents", type=int, default=4, help="Number of agents in the environment")
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
        "--comm-radius", type=float, default=None, help="Communication radius for local observation models"
    )
    parser.add_argument(
        "--torus", action="store_true", help="Whether to wrap around the world boundaries (toroidal world)"
    )
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

    # Common training and architecture parameters (shared across all scripts)
    add_common_training_args(parser)

    # Override default model path for rendezvous with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.set_defaults(model_path=f"models/rv_{timestamp}.zip")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Create environment
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
    )

    # Build algorithm parameters and embedding configuration from CLI args
    algo_params = build_algo_params(args, args.algorithm)
    embed_config = build_embed_config(args)

    print(f"\n{'=' * 60}")
    print(f"Training {args.algorithm.upper()} on Rendezvous Environment")
    print(f"{'=' * 60}")
    print(f"Agents: {args.num_agents}")
    print(f"World Size: {args.world_size}")
    print(f"Observation Model: {args.obs_model}")
    print(f"Total Timesteps: {args.total_timesteps:,}")
    print(f"Parallel Envs: {args.num_vec_envs}")
    print(f"{'=' * 60}")
    print(f"Architecture Configuration:")
    print(f"  Activation: {args.activation}")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  Policy Layers: {args.policy_layers}")
    print(f"  Embed Dim: {args.embed_dim}")
    print(f"  Phi Layers: {args.phi_layers}")
    print(f"{'=' * 60}\n")

    # Train model using utility function
    model, info = run_training_rendezvous(
        env=env,
        embed_config=embed_config,
        algo_params=algo_params,
        algorithm=args.algorithm,
        total_timesteps=args.total_timesteps,
        n_envs=args.num_vec_envs,
        save_path=args.model_path,
        resume_from=args.resume_from,
    )

    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"Model saved to: {args.model_path}")
    print(f"Total timesteps: {model.num_timesteps:,}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
