from __future__ import annotations

import argparse
import os

from environments.rendezvous.rendezvous_env import RendezvousEnv
from training.rendezvous_train_utils import run_training_rendezvous


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for the training script."""
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

    # Training parameters
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "trpo"], help="RL algorithm to use")
    parser.add_argument(
        "--total-timesteps", type=int, default=200_000, help="Total number of environment steps for training"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None, help="Learning rate (defaults: PPO=3e-4, TRPO=1e-3)"
    )
    parser.add_argument("--num-vec-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--n-steps", type=int, default=None, help="Rollout length (defaults: PPO=1024, TRPO=2048)")
    parser.add_argument("--batch-size", type=int, default=None, help="Minibatch size (defaults: PPO=512, TRPO=128)")
    parser.add_argument("--n-epochs", type=int, default=None, help="Number of epochs (PPO only, default=5)")
    parser.add_argument("--model-path", type=str, default="rendezvous_model.zip", help="File to save the trained model")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to a saved model to resume training from")
    parser.add_argument("--tensorboard-log", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA/GPU for training (default: CPU)")

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

    # Build algorithm parameters from CLI args
    algo_params = {"verbose": 1}
    if args.learning_rate is not None:
        algo_params["learning_rate"] = args.learning_rate
    if args.n_steps is not None:
        algo_params["n_steps"] = args.n_steps
    if args.batch_size is not None:
        algo_params["batch_size"] = args.batch_size
    if args.n_epochs is not None and args.algorithm == "ppo":
        algo_params["n_epochs"] = args.n_epochs
    if args.tensorboard_log is not None:
        algo_params["tensorboard_log"] = args.tensorboard_log

    # Set device (default to CPU, use CUDA only if explicitly requested)
    algo_params["device"] = "cuda" if args.use_cuda else "cpu"

    # Embedding configuration
    embed_config = {
        "embed_dim": 64,
        "phi_layers": 1,
    }

    print(f"\n{'=' * 60}")
    print(f"Training {args.algorithm.upper()} on Rendezvous Environment")
    print(f"{'=' * 60}")
    print(f"Agents: {args.num_agents}")
    print(f"World Size: {args.world_size}")
    print(f"Observation Model: {args.obs_model}")
    print(f"Total Timesteps: {args.total_timesteps:,}")
    print(f"Parallel Envs: {args.num_vec_envs}")
    print(f"{'=' * 60}\n")

    # Handle model resumption
    if args.resume_from:
        print(f"⚠️  Resume functionality not yet implemented in run_training_rendezvous")
        print(f"Starting fresh training instead...")

    # Train model using utility function
    model, info = run_training_rendezvous(
        env=env,
        embed_config=embed_config,
        algo_params=algo_params,
        algorithm=args.algorithm,
        total_timesteps=args.total_timesteps,
        n_envs=args.num_vec_envs,
        save_path=args.model_path,
    )

    print(f"\n{'=' * 60}")
    print(f"Training Complete!")
    print(f"Model saved to: {args.model_path}")
    print(f"Total timesteps: {model.num_timesteps:,}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
