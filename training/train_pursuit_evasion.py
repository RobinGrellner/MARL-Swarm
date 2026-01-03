"""
Training script for Pursuit-Evasion environment.

Trains multiple pursuer agents using PPO or TRPO with parameter sharing
to capture a scripted evader agent.
"""

from __future__ import annotations

import argparse

from environments.pursuit.pursuit_evasion_env import PursuitEvasionEnv
from training.pursuit_evasion_train_utils import run_training_pursuit_evasion


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train pursuers on the Pursuit-Evasion environment")

    # Environment parameters
    parser.add_argument("--num-pursuers", type=int, default=10, help="Number of pursuer agents")
    parser.add_argument("--world-size", type=float, default=10.0, help="Side length of the square world")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--capture-radius", type=float, default=0.5, help="Capture distance threshold")
    parser.add_argument("--evader-speed", type=float, default=1.0, help="Maximum speed of evader")
    parser.add_argument("--pursuer-speed", type=float, default=1.0, help="Maximum speed of pursuers")
    parser.add_argument(
        "--evader-strategy",
        type=str,
        default="voronoi_center",
        choices=["simple", "max_min_distance", "weighted_escape", "voronoi_center"],
        help="Evasion strategy (Hüttenrauch: voronoi_center)",
    )
    parser.add_argument(
        "--obs-model",
        type=str,
        default="global_basic",
        choices=["global_basic", "global_extended", "local_basic", "local_extended"],
        help="Observation model to use",
    )
    parser.add_argument("--comm-radius", type=float, default=None, help="Communication radius for local observation models")
    parser.add_argument(
        "--kinematics", type=str, default="single", choices=["single", "double"], help="Agent kinematic model"
    )
    parser.add_argument("--max-pursuers", type=int, default=None, help="Maximum number of pursuers for scale invariance")

    # Training parameters
    parser.add_argument("--algorithm", type=str, default="ppo", choices=["ppo", "trpo"], help="RL algorithm to use")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Total number of environment steps for training")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate")
    parser.add_argument("--num-vec-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--n-steps", type=int, default=None, help="Rollout length")
    parser.add_argument("--batch-size", type=int, default=None, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=None, help="Number of epochs (PPO only)")
    parser.add_argument("--model-path", type=str, default="pursuit_evasion_model.zip", help="File to save the trained model")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to a saved model to resume training from")
    parser.add_argument("--tensorboard-log", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA/GPU for training")

    return parser.parse_args()


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Create environment
    env = PursuitEvasionEnv(
        num_pursuers=args.num_pursuers,
        world_size=args.world_size,
        max_steps=args.max_steps,
        capture_radius=args.capture_radius,
        evader_speed=args.evader_speed,
        obs_model=args.obs_model,
        comm_radius=args.comm_radius,
        max_pursuers=args.max_pursuers if args.max_pursuers is not None else args.num_pursuers,
        kinematics=args.kinematics,
        evader_strategy=args.evader_strategy,
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

    algo_params["device"] = "cuda" if args.use_cuda else "cpu"

    # Embedding configuration
    embed_config = {
        "embed_dim": 64,
        "phi_layers": 1,
    }

    print(f"\n{'=' * 60}")
    print(f"Training {args.algorithm.upper()} on Pursuit-Evasion Environment")
    print(f"{'=' * 60}")
    print(f"Pursuers: {args.num_pursuers}")
    print(f"World Size: {args.world_size}")
    print(f"Observation Model: {args.obs_model}")
    print(f"Evader Strategy: {args.evader_strategy} (Hüttenrauch)")
    print(f"Total Timesteps: {args.total_timesteps:,}")
    print(f"Parallel Envs: {args.num_vec_envs}")
    print(f"{'=' * 60}\n")

    # Train model using utility function
    model, info = run_training_pursuit_evasion(
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
