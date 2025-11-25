"""
Main orchestration script for scalability experiments.

This script:
1. Checks if trained models exist, trains if needed
2. Evaluates models on multiple swarm sizes
3. Generates plots and reports
4. Saves all results for thesis

Usage:
    python evaluation/run_scalability_experiment.py --config configs/exp1.json
    or
    python evaluation/run_scalability_experiment.py --train-size 4 --test-sizes 4 8 16 32 50
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from environments.rendezvous.rendezvous_env import RendezvousEnv
from evaluation.plotting import create_scalability_report, plot_comparison
from evaluation.utils import (
    evaluate_on_multiple_sizes,
    load_model,
    print_scalability_summary,
)
from training.rendezvous_train_utils import run_training_rendezvous


def check_model_exists(model_path: Path) -> bool:
    """Check if a trained model exists."""
    model_path = Path(model_path)

    if model_path.exists():
        print(f"✓ Model found: {model_path}")
        return True
    else:
        print(f"✗ Model not found: {model_path}")
        return False


def train_model_if_needed(
    model_path: Path,
    env_config: Dict,
    train_config: Dict,
    force_retrain: bool = False,
) -> Path:
    """Train a model if it doesn't exist or if force_retrain is True.

    Args:
        model_path: Path where model should be saved
        env_config: Environment configuration
        train_config: Training configuration
        force_retrain: Force retraining even if model exists

    Returns:
        Path to the trained model
    """
    model_path = Path(model_path)

    if not force_retrain and check_model_exists(model_path):
        print(f"Using existing model: {model_path}")
        return model_path

    print(f"\n{'=' * 60}")
    print("TRAINING MODEL")
    print(f"{'=' * 60}")

    # Create models directory if it doesn't exist
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Create environment
    env = RendezvousEnv(**env_config)

    # Extract configurations with improved defaults
    embed_config = train_config.get("embed_config", {"embed_dim": 64, "phi_layers": 1})

    # Default PPO params (TRPO-like, tuned for rendezvous)
    default_ppo_params = {
        "learning_rate": 0.0003,
        "n_steps": 1024,
        "batch_size": 512,
        "n_epochs": 5,
        "clip_range": 0.1,
        "target_kl": 0.015,
        "gamma": 0.99,
        "gae_lambda": 0.98,
    }
    ppo_params = {**default_ppo_params, **train_config.get("ppo_params", {})}

    total_timesteps = train_config.get("total_timesteps", 15_000_000)
    n_envs = train_config.get("n_envs", 8)
    normalize = train_config.get("normalize", True)

    # Train model
    print(f"Training on {env_config['num_agents']} agents for {total_timesteps:,} timesteps...")
    model, info = run_training_rendezvous(
        env,
        embed_config,
        ppo_params,
        total_timesteps=total_timesteps,
        n_envs=n_envs,
        normalize=normalize,
        save_path=str(model_path),
    )

    print(f"\n✓ Model trained and saved to {model_path}")

    return model_path


def run_experiment(
    experiment_name: str,
    train_config: Dict,
    env_config: Dict,
    test_sizes: List[int],
    n_eval_episodes: int = 100,
    force_retrain: bool = False,
    output_dir: Optional[Path] = None,
) -> Dict[int, Dict[str, float]]:
    """Run a complete scalability experiment.

    Args:
        experiment_name: Name of the experiment
        train_config: Training configuration
        env_config: Environment configuration
        test_sizes: List of swarm sizes to test
        n_eval_episodes: Number of episodes per evaluation
        force_retrain: Force retraining even if model exists
        output_dir: Directory to save results (defaults to evaluation/results/<experiment_name>/)

    Returns:
        Dictionary of evaluation results
    """
    print(f"\n{'#' * 80}")
    print(f"# EXPERIMENT: {experiment_name}")
    print(f"{'#' * 80}\n")

    # Setup output directory
    if output_dir is None:
        output_dir = Path("evaluation/results") / experiment_name
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model path
    model_path = output_dir / f"{experiment_name}_model.zip"

    # Step 1: Train model if needed
    train_model_if_needed(
        model_path=model_path,
        env_config=env_config,
        train_config=train_config,
        force_retrain=force_retrain,
    )

    # Step 2: Load model
    print(f"\n{'=' * 60}")
    print("LOADING MODEL FOR EVALUATION")
    print(f"{'=' * 60}")

    model = load_model(model_path, verbose=True)

    # Step 3: Evaluate on multiple swarm sizes
    print(f"\n{'=' * 60}")
    print("EVALUATING SCALABILITY")
    print(f"{'=' * 60}")
    print(f"Test sizes: {test_sizes}")
    print(f"Episodes per size: {n_eval_episodes}")

    eval_env_config = env_config.copy()
    eval_env_config.pop("num_agents", None)

    results = evaluate_on_multiple_sizes(
        model=model,
        test_sizes=test_sizes,
        env_config=eval_env_config,
        n_episodes=n_eval_episodes,
        deterministic=True,
        verbose=True,
    )

    # Step 4: Print summary
    print_scalability_summary(results)

    # Step 5: Generate plots
    print(f"\n{'=' * 60}")
    print("GENERATING PLOTS")
    print(f"{'=' * 60}")

    create_scalability_report(
        results=results,
        output_dir=output_dir / "plots",
        experiment_name=experiment_name,
    )

    # Step 6: Save experiment configuration
    config_path = output_dir / f"{experiment_name}_config.json"
    with open(config_path, "w") as f:
        json.dump(
            {
                "experiment_name": experiment_name,
                "train_config": train_config,
                "env_config": env_config,
                "test_sizes": test_sizes,
                "n_eval_episodes": n_eval_episodes,
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    print(f"✓ Configuration saved to {config_path}")

    print(f"\n{'#' * 80}")
    print(f"# EXPERIMENT COMPLETE: {experiment_name}")
    print(f"# Results saved to: {output_dir}")
    print(f"{'#' * 80}\n")

    return results


def run_multiple_experiments(experiments: Dict[str, Dict]) -> Dict[str, Dict[int, Dict[str, float]]]:
    """Run multiple experiments and generate comparison plots.

    Args:
        experiments: Dictionary mapping experiment name to experiment config

    Returns:
        Dictionary mapping experiment name to results
    """
    all_results = {}

    for exp_name, exp_config in experiments.items():
        results = run_experiment(
            experiment_name=exp_name,
            train_config=exp_config["train_config"],
            env_config=exp_config["env_config"],
            test_sizes=exp_config.get("test_sizes", [5, 10, 25, 50, 100]),
            n_eval_episodes=exp_config.get("n_eval_episodes", 100),
            force_retrain=exp_config.get("force_retrain", False),
            output_dir=exp_config.get("output_dir", None),
        )
        all_results[exp_name] = results

    # Generate comparison plots
    if len(all_results) > 1:
        print(f"\n{'=' * 80}")
        print("GENERATING COMPARISON PLOTS")
        print(f"{'=' * 80}")

        comparison_dir = Path("evaluation/results/comparisons")
        comparison_dir.mkdir(parents=True, exist_ok=True)

        for metric in ["mean_reward", "success_rate", "mean_final_max_dist"]:
            plot_comparison(
                results_dict=all_results,
                metric=metric,
                title=f"Comparison: {metric.replace('_', ' ').title()}",
                save_path=comparison_dir / f"comparison_{metric}.png",
                show=False,
            )

        print(f"✓ Comparison plots saved to {comparison_dir}/")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Run scalability experiments for multi-agent RL")

    # Experiment selection
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to JSON config file with experiment(s) definition",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name of the experiment (used if not loading from config)",
    )

    # Training configuration
    parser.add_argument("--train-size", type=int, default=20, help="Number of agents to train on")
    parser.add_argument("--total-timesteps", type=int, default=15_000_000, help="Total training timesteps")
    parser.add_argument(
        "--n-envs", type=int, default=None, help="Number of parallel environments (overrides config file)"
    )
    parser.add_argument("--force-retrain", action="store_true", help="Force retraining even if model exists")

    # Evaluation configuration
    parser.add_argument(
        "--test-sizes",
        type=int,
        nargs="+",
        default=[5, 10, 25, 50, 100],
        help="Swarm sizes to test on",
    )
    parser.add_argument("--n-eval-episodes", type=int, default=100, help="Number of episodes per evaluation")

    # Environment configuration
    parser.add_argument("--world-size", type=float, default=100.0, help="World size")
    parser.add_argument("--max-steps", type=int, default=500, help="Max steps per episode")
    parser.add_argument(
        "--obs-model",
        type=str,
        default="local_extended",
        choices=["global_basic", "global_extended", "local_basic", "local_extended", "local_comm", "classic"],
        help="Observation model",
    )
    parser.add_argument("--comm-radius", type=float, default=70.0, help="Communication radius")
    parser.add_argument("--kinematics", type=str, default="single", choices=["single", "double"])
    parser.add_argument("--max-agents", type=int, default=250, help="Max agents for scale-invariant observations")
    parser.add_argument(
        "--break-distance-threshold", type=float, default=0.1, help="Distance threshold for successful rendezvous"
    )

    # Output
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")

    args = parser.parse_args()

    # Load from config file if provided
    if args.config:
        config_path = Path(args.config)
        with open(config_path) as f:
            config = json.load(f)

        # Override n_envs if provided via command line
        if args.n_envs is not None:
            print(f"\n{'!' * 60}")
            print(f"OVERRIDING n_envs with command-line value: {args.n_envs}")
            print(f"{'!' * 60}\n")

        # Check if single or multiple experiments
        if "experiments" in config:
            # Multiple experiments - override n_envs for all experiments
            if args.n_envs is not None:
                for exp_name, exp_config in config["experiments"].items():
                    exp_config["train_config"]["n_envs"] = args.n_envs

            run_multiple_experiments(config["experiments"])
        else:
            # Single experiment - override n_envs if provided
            if args.n_envs is not None:
                config["train_config"]["n_envs"] = args.n_envs

            run_experiment(
                experiment_name=config.get("experiment_name", "experiment"),
                train_config=config["train_config"],
                env_config=config["env_config"],
                test_sizes=config.get("test_sizes", [5, 10, 25, 50, 100]),
                n_eval_episodes=config.get("n_eval_episodes", 100),
                force_retrain=config.get("force_retrain", False),
                output_dir=config.get("output_dir", None),
            )
    else:
        # Run single experiment from command line args
        experiment_name = (
            args.experiment_name or f"exp_train{args.train_size}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        env_config = {
            "num_agents": args.train_size,
            "world_size": args.world_size,
            "max_steps": args.max_steps,
            "obs_model": args.obs_model,
            "comm_radius": args.comm_radius,
            "kinematics": args.kinematics,
            "max_agents": args.max_agents,
            "break_distance_threshold": args.break_distance_threshold,
        }

        train_config = {
            "total_timesteps": args.total_timesteps,
            "embed_config": {"embed_dim": 64, "phi_layers": 1},
            "ppo_params": {},
            "n_envs": args.n_envs if args.n_envs is not None else 1,
            "normalize": True,
        }

        run_experiment(
            experiment_name=experiment_name,
            train_config=train_config,
            env_config=env_config,
            test_sizes=args.test_sizes,
            n_eval_episodes=args.n_eval_episodes,
            force_retrain=args.force_retrain,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()
