#!/usr/bin/env python3
"""
Continue training architecture scalability experiments for Master's thesis.

This script continues training from saved models, adding 3 million more timesteps
to reach 5 million total (2M + 3M).

Usage:
    python run_architecture_scalability_continuation.py [options]

Options:
    --tier {1,2,2b,2c,3,all}   Continue specific tier(s) (default: all)
    --limit N                   Continue only first N experiments (for testing)
    --dry-run                   Print experiments without running them
    --tensorboard-log PATH      Custom TensorBoard log directory
    --num-vec-envs N            Number of parallel environments (default: 8)
    --continuation-timesteps N  Additional training duration (default: 3000000)
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


class ArchitectureScalabilityContinuation:
    """Manages continuation of architecture scalability experiments."""

    def __init__(
        self,
        config_path: str,
        tensorboard_log: Optional[str] = None,
        num_vec_envs: int = 8,
        continuation_timesteps: int = 3000000,
        dry_run: bool = False,
    ):
        self.config_path = Path(config_path)
        self.tensorboard_log = tensorboard_log or "logs/architecture_scalability"
        self.num_vec_envs = num_vec_envs
        self.continuation_timesteps = continuation_timesteps
        self.dry_run = dry_run

        # Load config
        with open(self.config_path) as f:
            self.config = json.load(f)

        self.experiments = self.config.get("experiments", {})
        self.total_experiments = 0
        self.completed_experiments = 0
        self.failed_experiments = 0
        self.skipped_experiments = 0

    def get_tier_configs(self, tier: str) -> Dict[str, Dict[str, Any]]:
        """Extract configs for a specific tier."""
        tier_configs = {}
        for exp_name, exp_config in self.experiments.items():
            if exp_name.startswith("_"):
                continue  # Skip metadata entries

            if tier == "1" and exp_name.startswith(("baseline", "activation", "aggregation", "depth", "width")):
                tier_configs[exp_name] = exp_config
            elif tier == "2" and exp_name.startswith(("deep_", "attention_")):
                tier_configs[exp_name] = exp_config
            elif tier == "2b" and exp_name.startswith("train_size"):
                tier_configs[exp_name] = exp_config
            elif tier == "2c" and exp_name.startswith("gen_"):
                tier_configs[exp_name] = exp_config
            elif tier == "3" and exp_name.startswith("robustness"):
                tier_configs[exp_name] = exp_config
            elif tier == "all":
                if not exp_name.startswith("_"):
                    tier_configs[exp_name] = exp_config

        return tier_configs

    def build_continuation_command(self, exp_name: str, exp_config: Dict[str, Any]) -> List[str]:
        """Build continuation training command from experiment config."""
        env_config = exp_config.get("env_config", {})
        train_config = exp_config.get("train_config", {})

        # Extract values with defaults
        num_agents = env_config.get("num_agents", 4)
        world_size = env_config.get("world_size", 10.0)
        obs_model = env_config.get("obs_model", "local_basic")
        comm_radius = env_config.get("comm_radius", 8.0)
        torus = env_config.get("torus", False)

        activation = train_config.get("activation", "relu")
        aggregation = train_config.get("aggregation", "mean")
        embed_dim = train_config.get("embed_dim", 64)
        phi_layers = train_config.get("phi_layers", 1)
        policy_layers = train_config.get("policy_layers", [64, 64])
        learning_rate = train_config.get("learning_rate", 0.0003)

        # Convert policy_layers to comma-separated string
        policy_layers_str = ",".join(str(x) for x in policy_layers)

        # Build model path (for saving updated model after continuation)
        model_path = f"models/architecture_scalability_{exp_name}_5m.zip"

        # Build path to saved model from first round
        resume_from = f"models/architecture_scalability_{exp_name}.zip"

        # Build log path (use same directory as original to merge logs)
        log_path = f"{self.tensorboard_log}/{exp_name}"

        cmd = [
            "uv",
            "run",
            "python",
            "training/train_rendezvous.py",
            "--num-agents",
            str(num_agents),
            "--world-size",
            str(world_size),
            "--obs-model",
            obs_model,
            "--comm-radius",
            str(comm_radius),
            "--activation",
            activation,
            "--aggregation",
            aggregation,
            "--embed-dim",
            str(embed_dim),
            "--phi-layers",
            str(phi_layers),
            "--policy-layers",
            policy_layers_str,
            "--learning-rate",
            str(learning_rate),
            "--total-timesteps",
            str(self.continuation_timesteps),
            "--num-vec-envs",
            str(self.num_vec_envs),
            "--model-path",
            model_path,
            "--tensorboard-log",
            log_path,
            "--resume-from",
            resume_from,
        ]

        if torus:
            cmd.append("--torus")

        return cmd, Path(resume_from)

    def run_continuation(self, exp_name: str, exp_config: Dict[str, Any]) -> bool:
        """Continue training a single experiment."""
        description = exp_config.get("description", exp_name)
        print(f"\n{'=' * 80}")
        print(f"Continuing: {exp_name}")
        print(f"Description: {description}")
        print(f"{'=' * 80}")

        cmd, resume_from_path = self.build_continuation_command(exp_name, exp_config)

        # Check if model exists
        if not resume_from_path.exists():
            print(f"[SKIP] Model not found: {resume_from_path}")
            return None  # Skip, not a failure

        if self.dry_run:
            print(f"DRY RUN - Would execute:")
            print(" ".join(cmd))
            return True

        try:
            result = subprocess.run(cmd, check=True)
            print(f"[OK] {exp_name} continued successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {exp_name} continuation failed with exit code {e.returncode}")
            return False

    def run_tier(self, tier: str, limit: Optional[int] = None) -> None:
        """Continue all experiments in a tier."""
        tier_configs = self.get_tier_configs(tier)
        tier_name = {
            "1": "Tier 1: Main Effects",
            "2": "Tier 2: Key Interactions",
            "2b": "Tier 2b: Training Size Effects",
            "2c": "Tier 2c: Generalization Matrix",
            "3": "Tier 3: Environmental Variations",
            "all": "All Tiers",
        }.get(tier, tier)

        print(f"\n{'=' * 80}")
        print(f"{tier_name} - CONTINUATION")
        print(f"Total experiments: {len(tier_configs)}")
        print(f"{'=' * 80}")

        experiments = list(tier_configs.items())
        if limit:
            experiments = experiments[:limit]

        for i, (exp_name, exp_config) in enumerate(experiments, 1):
            print(f"\n[{i}/{len(experiments)}]", end=" ")
            result = self.run_continuation(exp_name, exp_config)
            self.total_experiments += 1
            if result is True:
                self.completed_experiments += 1
            elif result is None:
                self.skipped_experiments += 1
            else:
                self.failed_experiments += 1

    def print_summary(self) -> None:
        """Print continuation summary."""
        print(f"\n{'=' * 80}")
        print(f"CONTINUATION SUMMARY")
        print(f"{'=' * 80}")
        print(f"Total experiments: {self.total_experiments}")
        print(f"Completed successfully: {self.completed_experiments}")
        print(f"Skipped (model not found): {self.skipped_experiments}")
        print(f"Failed: {self.failed_experiments}")
        if self.total_experiments > 0:
            success_rate = (self.completed_experiments / self.total_experiments) * 100
            print(f"Success rate: {success_rate:.1f}%")
        print(f"\nTensorBoard logs (merged 2M + 3M): {self.tensorboard_log}")
        print(f"Models after 2M steps: models/architecture_scalability_*.zip")
        print(f"Models after 5M steps: models/architecture_scalability_*_5m.zip")
        print(f"\nTo view full training progress (2M + 3M merged):")
        print(f"  tensorboard --logdir {self.tensorboard_log}")
        print(f"{'=' * 80}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Continue training architecture scalability experiments for Master's thesis"
    )
    parser.add_argument(
        "--tier",
        type=str,
        default="all",
        choices=["1", "2", "2b", "2c", "3", "all"],
        help="Which tier(s) to continue (default: all)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Continue only first N experiments (useful for testing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiments without running them",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default="logs/architecture_scalability",
        help="TensorBoard log directory",
    )
    parser.add_argument(
        "--num-vec-envs",
        type=int,
        default=8,
        help="Number of parallel environments",
    )
    parser.add_argument(
        "--continuation-timesteps",
        type=int,
        default=3000000,
        help="Additional training duration in timesteps (default: 3000000 for 5M total)",
    )

    args = parser.parse_args()

    config_path = Path("training/configs/architecture_scalability.json")
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    runner = ArchitectureScalabilityContinuation(
        config_path=config_path,
        tensorboard_log=args.tensorboard_log,
        num_vec_envs=args.num_vec_envs,
        continuation_timesteps=args.continuation_timesteps,
        dry_run=args.dry_run,
    )

    # Print continuation info
    print(f"\n{'=' * 80}")
    print(f"ARCHITECTURE SCALABILITY EXPERIMENTS - CONTINUATION")
    print(f"{'=' * 80}")
    print(f"Start time: {datetime.datetime.now()}")
    print(f"Config: {config_path}")
    print(f"Tier: {args.tier}")
    print(f"Additional training: {args.continuation_timesteps:,} timesteps (2M + 3M = 5M total)")
    print(f"TensorBoard logs: {args.tensorboard_log}")
    print(f"Parallel environments: {args.num_vec_envs}")
    if args.dry_run:
        print(f"Mode: DRY RUN (no training will execute)")
    print(f"{'=' * 80}\n")

    # Run continuations
    if args.tier == "all":
        for tier in ["1", "2", "2b", "2c", "3"]:
            runner.run_tier(tier, limit=args.limit)
    else:
        runner.run_tier(args.tier, limit=args.limit)

    # Print summary
    runner.print_summary()

    return 0 if runner.failed_experiments == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
