"""
Training script for Pursuit-Evasion environment.

Trains multiple pursuer agents using PPO or TRPO with parameter sharing
to capture a scripted evader agent.
"""

from __future__ import annotations

import argparse
from datetime import datetime

from environments.pursuit.pursuit_evasion_env import PursuitEvasionEnv
from training.pursuit_evasion_train_utils import run_training_pursuit_evasion
from training.common_train_utils import add_common_training_args, build_algo_params, build_embed_config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train pursuers on the Pursuit-Evasion environment")

    # Environment parameters
    parser.add_argument("--num-pursuers", type=int, default=10, help="Number of pursuer agents")
    parser.add_argument("--world-size", type=float, default=10.0, help="Side length of the square world")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--capture-radius", type=float, default=0.5, help="Capture distance threshold")
    parser.add_argument("--evader-speed", type=float, default=1.0, help="Maximum speed of evader")
    parser.add_argument(
        "--evader-strategy",
        type=str,
        default="huttenrauch",
        choices=["simple", "max_min_distance", "weighted_escape", "voronoi_center", "huttenrauch"],
        help="Evasion strategy (Hüttenrauch: huttenrauch)",
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
    parser.add_argument("--v-max", type=float, default=1.0, help="Maximum linear velocity for pursuers")
    parser.add_argument("--omega-max", type=float, default=1.0, help="Maximum angular velocity for pursuers")
    parser.add_argument("--torus", action="store_true", help="Use toroidal world topology (wraparound)")

    # Common training and architecture parameters (shared across all scripts)
    add_common_training_args(parser)

    # Override default model path for pursuit-evasion with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    parser.set_defaults(model_path=f"models/pe_{timestamp}.zip")

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
        v_max=args.v_max,
        obs_model=args.obs_model,
        comm_radius=args.comm_radius,
        max_pursuers=args.max_pursuers if args.max_pursuers is not None else args.num_pursuers,
        kinematics=args.kinematics,
        omega_max=args.omega_max,
        evader_strategy=args.evader_strategy,
        torus=args.torus,
    )

    # Build algorithm parameters and embedding configuration from CLI args
    algo_params = build_algo_params(args, args.algorithm)
    embed_config = build_embed_config(args)

    print(f"\n{'=' * 60}")
    print(f"Training {args.algorithm.upper()} on Pursuit-Evasion Environment")
    print(f"{'=' * 60}")
    print(f"Pursuers: {args.num_pursuers}")
    print(f"World Size: {args.world_size}")
    print(f"Observation Model: {args.obs_model}")
    print(f"Evader Strategy: {args.evader_strategy} (Hüttenrauch)")
    print(f"Total Timesteps: {args.total_timesteps:,}")
    print(f"Parallel Envs: {args.num_vec_envs}")
    print(f"\nArchitecture:")
    print(f"  Activation: {args.activation}")
    print(f"  Aggregation: {args.aggregation}")
    print(f"  Embed Dim: {args.embed_dim}")
    print(f"  Phi Layers: {args.phi_layers}")
    print(f"  Policy Layers: {args.policy_layers}")
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
