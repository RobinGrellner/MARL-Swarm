from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
from stable_baselines3.common.base_class import BaseAlgorithm

from environments.rendezvous.rendezvous_env import RendezvousEnv

# Import the generic training function
from training.common_train_utils import run_training


def run_training_rendezvous(
    env: RendezvousEnv,
    embed_config: Dict[str, Any],
    algo_params: Dict[str, Any],
    *,
    algorithm: str = "ppo",
    total_timesteps: int = 200_000,
    log_info_keys: Optional[Tuple[str, ...]] = None,
    n_envs: int = 1,
    normalize: bool = True,
    save_path: Optional[str] = None,
    resume_from: Optional[str] = None,
) -> Tuple[BaseAlgorithm, Dict[str, Any]]:
    """Train an RL model (PPO or TRPO) on RendezvousEnv with mean embedding.

    Backward-compatible wrapper around the generic training function.

    Args:
        env: RendezvousEnv instance
        embed_config: Configuration for mean embedding extractor
        algo_params: Algorithm hyperparameters (PPO or TRPO)
        algorithm: "ppo" or "trpo"
        total_timesteps: Total training timesteps
        log_info_keys: Keys to log from env info dict (default: swarm convergence metrics)
        n_envs: Number of parallel environments
        normalize: Whether to apply VecNormalize (deprecated, kept for API compatibility)
        save_path: Path to save model (optional)
        resume_from: Path to a saved model to resume training from (optional)

    Returns:
        Tuple of (trained model, info dict with vec_env)
    """
    # Default logging keywords for Rendezvous: swarm convergence metrics
    if log_info_keys is None:
        log_info_keys = ("max_pairwise_distance", "distance_to_com")

    return run_training(
        env,
        embed_config,
        algo_params,
        algorithm=algorithm,
        total_timesteps=total_timesteps,
        log_info_keys=log_info_keys,
        n_envs=n_envs,
        normalize=normalize,
        save_path=save_path,
        resume_from=resume_from,
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run a configurable PPO training on RendezvousEnv")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of agents")
    parser.add_argument("--world-size", type=float, default=10.0, help="Side length of the world")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum steps per episode")
    parser.add_argument(
        "--obs-model",
        type=str,
        default="local_basic",
        choices=["global_basic", "global_extended", "local_basic", "local_extended", "local_comm", "classic"],
        help="Observation model to use",
    )
    parser.add_argument(
        "--comm-radius", type=float, default=None, help="Communication radius (None defaults to world size)"
    )
    parser.add_argument(
        "--kinematics", type=str, default="single", choices=["single", "double"], help="Kinematic model"
    )
    parser.add_argument(
        "--max-agents", type=int, default=None, help="Maximum agents for scale-invariant observation size"
    )
    parser.add_argument("--embed-dim", type=int, default=64, help="Dimensionality of the mean embedding")
    parser.add_argument("--phi-layers", type=int, default=1, help="Number of hidden layers in the phi network")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for PPO")
    parser.add_argument("--n-steps", type=int, default=128, help="Rollout length per environment")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs per update")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Total training timesteps")
    parser.add_argument("--device", type=str, default="cpu", help="Device to train on (cpu/cuda)")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel vector environments")
    parser.add_argument("--no-normalize", action="store_true", help="Disable observation and reward normalisation")
    parser.add_argument("--log-info", nargs="*", default=None, help="Info keys to log via VecMonitor")
    parser.add_argument(
        "--model-path", type=str, default="models/rendezvous_model.zip", help="Path to save the trained model"
    )
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
