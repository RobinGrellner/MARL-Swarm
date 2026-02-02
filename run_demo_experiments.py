#!/usr/bin/env python3
"""Demo: train and evaluate Rendezvous + Pursuit-Evasion on multiple scales."""

import argparse
import sys
import subprocess
from pathlib import Path
from argparse import Namespace

from training.evaluate_rendezvous import evaluate as eval_rendezvous_func
from training.evaluate_pursuit_evasion import main as eval_pursuit_func


def run_command(cmd: list, description: str) -> bool:
    """Run a shell command and report status."""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"{'='*70}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n[OK] {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n[FAIL] {description} failed with exit code {e.returncode}")
        return False


def train_rendezvous() -> bool:
    """Train Rendezvous model."""
    cmd = [
        sys.executable, "training/train_rendezvous.py",
        "--num-agents", "20",
        "--world-size", "100.0",
        "--max-steps", "500",
        "--obs-model", "local_extended",
        "--comm-radius", "141.42",
        "--break-distance-threshold", "0.1",
        "--total-timesteps", "500000",
        "--num-vec-envs", "8",
        "--model-path", "models/quick_rendezvous_demo.zip",
        "--tensorboard-log", "logs/quick_rendezvous_demo"
    ]
    return run_command(cmd, "Rendezvous Training (500k steps)")


def train_pursuit() -> bool:
    """Train Pursuit-Evasion model."""
    cmd = [
        sys.executable, "training/train_pursuit_evasion.py",
        "--num-pursuers", "10",
        "--world-size", "10.0",
        "--max-steps", "100",
        "--capture-radius", "0.5",
        "--evader-speed", "1.0",
        "--v-max", "1.0",
        "--evader-strategy", "voronoi_center",
        "--obs-model", "global_basic",
        "--total-timesteps", "300000",
        "--num-vec-envs", "8",
        "--model-path", "models/quick_pursuit_demo.zip",
        "--tensorboard-log", "logs/quick_pursuit_demo"
    ]
    return run_command(cmd, "Pursuit-Evasion Training (300k steps)")


def evaluate_rendezvous() -> bool:
    """Evaluate Rendezvous model at training size."""
    try:
        print(f"\n{'='*70}")
        print(f"Rendezvous Evaluation (20 agents)")
        print(f"{'='*70}\n")

        args = Namespace(
            model_path="models/quick_rendezvous_demo.zip",
            num_agents=20,
            world_size=100.0,
            max_steps=500,
            obs_model="local_extended",
            comm_radius=141.42,
            torus=False,
            break_distance_threshold=0.1,
            kinematics="single",
            v_max=1.0,
            omega_max=1.0,
            max_agents=None,
            n_episodes=10,
            render_mode=None,
            fps=30,
            deterministic=True
        )
        eval_rendezvous_func(args)
        print(f"\n[OK] Rendezvous Evaluation completed successfully!")
        return True
    except Exception as e:
        print(f"\n[FAIL] Rendezvous Evaluation failed: {e}")
        return False


def evaluate_pursuit() -> bool:
    """Evaluate Pursuit-Evasion model at training size."""
    try:
        print(f"\n{'='*70}")
        print(f"Pursuit-Evasion Evaluation (10 pursuers)")
        print(f"{'='*70}\n")

        # Simulate command-line arguments for the main function
        old_argv = sys.argv
        sys.argv = [
            "evaluate_pursuit_evasion.py",
            "--model-path", "models/quick_pursuit_demo.zip",
            "--num-pursuers", "10",
            "--world-size", "10.0",
            "--max-steps", "100",
            "--capture-radius", "0.5",
            "--evader-speed", "1.0",
            "--obs-model", "global_basic",
            "--evader-strategy", "voronoi_center",
            "--n-episodes", "10"
        ]
        eval_pursuit_func()
        sys.argv = old_argv
        print(f"\n[OK] Pursuit-Evasion Evaluation completed successfully!")
        return True
    except Exception as e:
        print(f"\n[FAIL] Pursuit-Evasion Evaluation failed: {e}")
        sys.argv = old_argv
        return False


def create_results_directory() -> None:
    """Ensure results directory exists."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    print(f"[OK] Results directory ready: {results_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run quick demo experiments for MARL Swarm"
    )
    parser.add_argument(
        "--rendezvous-only",
        action="store_true",
        help="Train and evaluate Rendezvous only"
    )
    parser.add_argument(
        "--pursuit-only",
        action="store_true",
        help="Train and evaluate Pursuit-Evasion only"
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Train models only (skip evaluation)"
    )
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Evaluate models only (skip training)"
    )

    args = parser.parse_args()

    # Determine what to run
    run_rendezvous = not args.pursuit_only
    run_pursuit = not args.rendezvous_only

    create_results_directory()

    success_count = 0
    total_count = 0

    # Training
    if not args.eval_only:
        if run_rendezvous:
            total_count += 1
            if train_rendezvous():
                success_count += 1

        if run_pursuit:
            total_count += 1
            if train_pursuit():
                success_count += 1

    # Evaluation
    if not args.train_only:
        if run_rendezvous:
            total_count += 1
            if evaluate_rendezvous():
                success_count += 1

        if run_pursuit:
            total_count += 1
            if evaluate_pursuit():
                success_count += 1

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {success_count}/{total_count} experiments completed successfully")
    print(f"{'='*70}")
    print("\nNext steps:")
    print("1. Open analyze_experiments.ipynb to visualize results")
    print("2. Check logs/ directory for TensorBoard logs:")
    print("   tensorboard --logdir logs/")

    return 0 if success_count == total_count else 1


if __name__ == "__main__":
    sys.exit(main())
