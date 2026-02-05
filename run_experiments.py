#!/usr/bin/env python3
"""
General-purpose experiment runner for MARL swarm training.

Runs a suite of experiments from a JSON config file. Supports matrix parameter
expansion to automatically generate multiple experiment configurations from a
single config template. Each experiment trains a policy and logs results to TensorBoard.

Supports:
- Rendezvous task (training/train_rendezvous.py)
- Pursuit-Evasion task (training/train_pursuit_evasion.py)
- Any custom training script with compatible command-line interface

Features:
- Matrix parameter expansion: Automatically expands combinations of parameters
- Auto-naming: Generates experiment names from parameter combinations
- Sequential execution: Runs experiments one at a time
- TensorBoard logging: Organizes logs per experiment
- Dry-run mode: Preview experiments before running

Usage:
    python run_experiments.py --config CONFIG.json --train-script SCRIPT.py [options]

Examples:
    # Run all experiments from embedding_scaling_rendezvous.json
    python run_experiments.py \\
      --config training/configs/embedding_scaling_rendezvous.json \\
      --num-vec-envs 8

    # Run pursuit-evasion experiments
    python run_experiments.py \\
      --config training/configs/embedding_scaling_pursuit_evasion.json \\
      --train-script training/train_pursuit_evasion.py \\
      --num-vec-envs 8

    # Test first 3 experiments without running
    python run_experiments.py \\
      --config training/configs/embedding_scaling_rendezvous.json \\
      --limit 3 --dry-run

Options:
    --config PATH               Config file (required)
    --train-script PATH         Training script to use (default: training/train_rendezvous.py)
    --num-vec-envs N            Number of parallel environments (default: 8)
    --total-timesteps N         Training duration in timesteps (default: 2000000)
    --tensorboard-log PATH      Custom TensorBoard log directory (default: logs/experiments)
    --limit N                   Run only first N experiments (for testing)
    --dry-run                   Print experiments without running them
    --use-cuda                  Enable GPU/CUDA training for all experiments
    --model-dir PATH            Directory to save trained models (default: models/)
"""

import json
import subprocess
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import datetime
import io

# Force UTF-8 output on Windows
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Import config expansion utility
from training.config_utils import load_and_expand_config


class ArchitectureScalabilityRunner:
    """Manages execution of architecture scalability experiments."""

    def __init__(
        self,
        config_path: str,
        tensorboard_log: Optional[str] = None,
        num_vec_envs: int = 8,
        total_timesteps: int = 2000000,
        dry_run: bool = False,
        train_script: str = "training/train_rendezvous.py",
        use_cuda: bool = False,
        model_dir: Optional[str] = None,
    ):
        self.config_path = Path(config_path)
        self.tensorboard_log = tensorboard_log or "logs/architecture_scalability"
        self.num_vec_envs = num_vec_envs
        self.total_timesteps = total_timesteps
        self.dry_run = dry_run
        self.train_script = train_script
        self.use_cuda = use_cuda
        self.model_dir = model_dir or "models"

        # Load config and expand matrix parameters
        self.config = load_and_expand_config(str(self.config_path))

        self.experiments = self.config.get("experiments", {})
        self.total_experiments = 0
        self.completed_experiments = 0
        self.failed_experiments = 0

    def get_all_experiments(self, limit: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """Get all experiments from config, optionally limited to first N."""
        experiments = {
            name: config for name, config in self.experiments.items()
            if not name.startswith("_")  # Skip metadata entries
        }

        if limit:
            experiments = dict(list(experiments.items())[:limit])

        return experiments

    def build_train_command(self, exp_name: str, exp_config: Dict[str, Any]) -> List[str]:
        """Build training command from experiment config.

        Merges experiment config with defaults from config file.
        Priority: experiment config > defaults section > runner hardcoded defaults
        """
        # Get defaults from config file if available
        config_defaults = self.config.get("defaults", {})
        default_env_config = config_defaults.get("env_config", {})
        default_train_config = config_defaults.get("train_config", {})

        # Merge: defaults first, then experiment-specific overrides
        env_config = {**default_env_config, **exp_config.get("env_config", {})}
        train_config = {**default_train_config, **exp_config.get("train_config", {})}

        # Extract values with fallback to runner hardcoded defaults
        # Use num_pursuers for PE, num_agents for Rendezvous
        num_agents = env_config.get("num_pursuers", env_config.get("num_agents", 4))
        world_size = env_config.get("world_size", 10.0)
        max_steps = env_config.get("max_steps", 100)
        obs_model = env_config.get("obs_model", "local_basic")
        comm_radius = env_config.get("comm_radius", 8.0)
        torus = env_config.get("torus", False)
        break_distance_threshold = env_config.get("break_distance_threshold", None)
        v_max = env_config.get("v_max", 1.0)
        omega_max = env_config.get("omega_max", 1.0)

        # PE-specific parameters
        capture_radius = env_config.get("capture_radius", 0.5)
        evader_speed = env_config.get("evader_speed", 1.0)
        evader_strategy = env_config.get("evader_strategy", "huttenrauch")

        activation = train_config.get("activation", "relu")
        aggregation = train_config.get("aggregation", "mean")
        embed_dim = train_config.get("embed_dim", 64)
        phi_layers = train_config.get("phi_layers", 1)
        policy_layers = train_config.get("policy_layers", None)
        learning_rate = train_config.get("learning_rate", 0.0003)
        seed = train_config.get("seed", None)
        algorithm = train_config.get("algorithm", "trpo")
        n_steps = train_config.get("n_steps", None)
        batch_size = train_config.get("batch_size", None)

        # Compute total_timesteps from n_iterations if specified, otherwise fall back
        n_iterations = train_config.get("n_iterations", None)
        if n_iterations is not None:
            n_steps_val = n_steps if n_steps is not None else 500
            total_timesteps = n_iterations * n_steps_val * num_agents * self.num_vec_envs
        else:
            total_timesteps = train_config.get("total_timesteps", self.total_timesteps)

        # Extract environment parameters for scale invariance
        max_pursuers = env_config.get("max_pursuers", None)

        # Convert policy_layers to comma-separated string (only if explicitly set)
        policy_layers_str = ",".join(str(x) for x in policy_layers) if policy_layers is not None else None

        # Build model path
        model_path = f"{self.model_dir}/{exp_name}.zip"

        # Build log path
        log_path = f"{self.tensorboard_log}/{exp_name}"

        # Determine agent parameter name based on script
        agent_param = "--num-pursuers" if "pursuit" in self.train_script else "--num-agents"

        cmd = [
            "uv",
            "run",
            "python",
            self.train_script,
            agent_param,
            str(num_agents),
            "--world-size",
            str(world_size),
            "--max-steps",
            str(max_steps),
            "--obs-model",
            obs_model,
            "--activation",
            activation,
            "--aggregation",
            aggregation,
            "--embed-dim",
            str(embed_dim),
            "--phi-layers",
            str(phi_layers),
            "--learning-rate",
            str(learning_rate),
            "--v-max",
            str(v_max),
            "--omega-max",
            str(omega_max),
            "--total-timesteps",
            str(total_timesteps),
            "--num-vec-envs",
            str(self.num_vec_envs),
            "--algorithm",
            algorithm,
            "--model-path",
            model_path,
            "--tensorboard-log",
            log_path,
        ]

        # Add policy-layers only if explicitly specified in config
        if policy_layers_str is not None:
            cmd.extend(["--policy-layers", policy_layers_str])

        # Add max_pursuers for scale invariance if specified (PE)
        if max_pursuers is not None:
            cmd.extend(["--max-pursuers", str(max_pursuers)])

        # Add max_agents for scale invariance if specified (Rendezvous)
        max_agents = env_config.get("max_agents", None)
        if max_agents is not None:
            cmd.extend(["--max-agents", str(max_agents)])

        # Add comm-radius if explicitly specified (None means use env defaults: world_size for global, 8.0 for local)
        if comm_radius is not None:
            cmd.extend(["--comm-radius", str(comm_radius)])

        # Add PE-specific parameters if this is a pursuit-evasion task
        if "pursuit" in self.train_script:
            cmd.extend([
                "--capture-radius",
                str(capture_radius),
                "--evader-speed",
                str(evader_speed),
                "--evader-strategy",
                evader_strategy,
            ])

        # Add seed if specified
        if seed is not None:
            cmd.extend(["--seed", str(seed)])

        # Add algorithm-specific hyperparameters if specified
        if n_steps is not None:
            cmd.extend(["--n-steps", str(n_steps)])

        if batch_size is not None:
            cmd.extend(["--batch-size", str(batch_size)])

        if torus:
            cmd.append("--torus")

        # Add break-distance-threshold if specified (Rendezvous early stopping)
        if break_distance_threshold is not None:
            cmd.extend(["--break-distance-threshold", str(break_distance_threshold)])

        # Add GPU flag if enabled
        if self.use_cuda:
            cmd.append("--use-cuda")

        return cmd

    def run_experiment(self, exp_name: str, exp_config: Dict[str, Any], exp_index: int, total_exps: int) -> bool:
        """Run a single training experiment."""
        description = exp_config.get("description", exp_name)
        print(f"\n{'=' * 80}")
        print(f"[{exp_index}/{total_exps}] Experiment: {exp_name}")
        print(f"Description: {description}")
        print(f"{'=' * 80}")

        cmd = self.build_train_command(exp_name, exp_config)

        if self.dry_run:
            print(f"DRY RUN - Would execute:")
            print(" ".join(cmd))
            return True

        try:
            result = subprocess.run(cmd, check=True)
            print(f"[OK] {exp_name} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {exp_name} failed with exit code {e.returncode}")
            return False

    def run_experiments(self, limit: Optional[int] = None) -> None:
        """Run all experiments from config sequentially."""
        experiments = self.get_all_experiments(limit=limit)

        print(f"\n{'=' * 80}")
        print(f"Running {len(experiments)} experiments (sequential mode)")
        print(f"{'=' * 80}")

        # Sequential execution
        for i, (exp_name, exp_config) in enumerate(experiments.items(), 1):
            success = self.run_experiment(exp_name, exp_config, i, len(experiments))
            self.total_experiments += 1
            if success:
                self.completed_experiments += 1
            else:
                self.failed_experiments += 1

    def print_summary(self) -> None:
        """Print experiment summary."""
        print(f"\n{'=' * 80}")
        print(f"EXPERIMENT SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total experiments run: {self.total_experiments}")
        print(f"Completed successfully: {self.completed_experiments}")
        print(f"Failed: {self.failed_experiments}")
        if self.total_experiments > 0:
            success_rate = (self.completed_experiments / self.total_experiments) * 100
            print(f"Success rate: {success_rate:.1f}%")
        print(f"\nTensorBoard logs: {self.tensorboard_log}")
        print(f"Models saved to: {self.model_dir}/")
        print(f"\nTo view training progress:")
        print(f"  tensorboard --logdir {self.tensorboard_log}")
        print(f"{'=' * 80}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Run experiment suite from config"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="training/configs/architecture_scalability.json",
        help="Path to experiment config JSON file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only first N experiments (useful for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiments without running them",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="logs/experiments",
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--num-vec-envs",
        type=int,
        default=8,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2000000,
        help="Training duration in timesteps",
    )
    parser.add_argument(
        "--train-script",
        type=str,
        default="training/train_rendezvous.py",
        help="Training script to use (train_rendezvous.py or train_pursuit_evasion.py)",
    )
    parser.add_argument(
        "--use-cuda",
        action="store_true",
        help="Enable GPU/CUDA training for all experiments",
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory to save trained models (default: models/)",
    )

    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    runner = ArchitectureScalabilityRunner(
        config_path=config_path,
        tensorboard_log=args.tensorboard_log,
        num_vec_envs=args.num_vec_envs,
        total_timesteps=args.total_timesteps,
        dry_run=args.dry_run,
        train_script=args.train_script,
        use_cuda=args.use_cuda,
        model_dir=args.model_dir,
    )

    # Print experiment info
    print(f"\n{'=' * 80}")
    print(f"EXPERIMENT RUNNER - Master's Thesis")
    print(f"{'=' * 80}")
    print(f"Start time: {datetime.datetime.now()}")
    print(f"Config: {config_path}")
    print(f"TensorBoard logs: {args.tensorboard_log}")
    print(f"Parallel environments per experiment: {args.num_vec_envs}")
    print(f"GPU/CUDA enabled: {args.use_cuda}")
    print(f"Model directory: {args.model_dir}")
    print(f"Timesteps per training: {args.total_timesteps:,}")
    if args.limit:
        print(f"Limit: first {args.limit} experiments only")
    if args.dry_run:
        print(f"Mode: DRY RUN (no training will execute)")
    print(f"{'=' * 80}\n")

    # Run experiments
    runner.run_experiments(limit=args.limit)

    # Print summary
    runner.print_summary()

    return 0 if runner.failed_experiments == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
