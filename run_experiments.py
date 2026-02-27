import json
import subprocess
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
import datetime
import io
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
        use_cuda: bool = False,
        model_dir: Optional[str] = None,
    ):
        self.config_path = Path(config_path)
        if tensorboard_log is None:
            config_name = self.config_path.stem
            self.tensorboard_log = f"logs/{config_name}"
        else:
            self.tensorboard_log = tensorboard_log
        self.num_vec_envs = num_vec_envs
        self.total_timesteps = total_timesteps
        self.dry_run = dry_run
        self.use_cuda = use_cuda
        self.model_dir = model_dir or "models"

        self.config = load_and_expand_config(str(self.config_path))

        self.experiments = self.config.get("experiments", {})
        self.total_experiments = 0
        self.completed_experiments = 0
        self.failed_experiments = 0

    def get_all_experiments(self, limit: Optional[int] = None, skip: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """Get all experiments from config, optionally limited and skipping first N."""
        experiments = {
            name: config for name, config in self.experiments.items()
            if not name.startswith("_")
        }

        items = list(experiments.items())

        if skip:
            items = items[skip:]

        if limit:
            items = items[:limit]

        return dict(items)

    def build_train_command(self, exp_name: str, exp_config: Dict[str, Any]) -> List[str]:
        """Build training command from experiment config."""
        env_config = exp_config.get("env_config", {})
        train_config = exp_config.get("train_config", {})

        environment = env_config.get("environment", "rendezvous")
        if environment == "pursuit_evasion":
            train_script = "training/train_pursuit_evasion.py"
            agent_param = "--num-pursuers"
            num_agents = env_config.get("num_pursuers")
        else:
            train_script = "training/train_rendezvous.py"
            agent_param = "--num-agents"
            num_agents = env_config.get("num_agents")

        num_vec_envs = self.num_vec_envs if self.num_vec_envs is not None else train_config.get("num_vec_envs")

        n_iterations = train_config.get("n_iterations")
        if n_iterations is not None:
            n_steps = train_config.get("n_steps", 500)
            total_timesteps = n_iterations * n_steps * num_agents * num_vec_envs
        else:
            total_timesteps = train_config.get("total_timesteps", self.total_timesteps)

        model_path = f"{self.model_dir}/{exp_name}.zip"
        log_path = f"{self.tensorboard_log}/{exp_name}"

        cmd = [
            "uv",
            "run",
            "python",
            train_script,
            agent_param,
            str(num_agents),
            "--model-path",
            model_path,
            "--tensorboard-log",
            log_path,
        ]

        env_params = {
            "--world-size": env_config.get("world_size"),
            "--max-steps": env_config.get("max_steps"),
            "--obs-model": env_config.get("obs_model"),
            "--v-max": env_config.get("v_max"),
            "--omega-max": env_config.get("omega_max"),
        }
        for flag, value in env_params.items():
            if value is not None:
                cmd.extend([flag, str(value)])

        train_params = {
            "--activation": train_config.get("activation"),
            "--aggregation": train_config.get("aggregation"),
            "--embed-dim": train_config.get("embed_dim"),
            "--phi-layers": train_config.get("phi_layers"),
            "--learning-rate": train_config.get("learning_rate"),
            "--algorithm": train_config.get("algorithm"),
            "--total-timesteps": total_timesteps,
            "--num-vec-envs": num_vec_envs,
        }
        for flag, value in train_params.items():
            if value is not None:
                cmd.extend([flag, str(value)])

        optional_params = {
            "--phi-hidden-width": train_config.get("phi_hidden_width"),
            "--comm-radius": env_config.get("comm_radius"),
            "--max-pursuers": env_config.get("max_pursuers"),
            "--max-agents": env_config.get("max_agents"),
            "--capture-radius": env_config.get("capture_radius"),
            "--evader-speed": env_config.get("evader_speed"),
            "--evader-strategy": env_config.get("evader_strategy"),
            "--seed": train_config.get("seed"),
            "--n-steps": train_config.get("n_steps"),
            "--batch-size": train_config.get("batch_size"),
            "--break-distance-threshold": env_config.get("break_distance_threshold"),
            "--kinematics": env_config.get("kinematics"),
        }
        for flag, value in optional_params.items():
            if value is not None:
                cmd.extend([flag, str(value)])

        policy_layers = train_config.get("policy_layers")
        if policy_layers is not None:
            policy_layers_str = ",".join(str(x) for x in policy_layers)
            cmd.extend(["--policy-layers", policy_layers_str])

        if env_config.get("torus"):
            cmd.append("--torus")

        use_cuda = train_config.get("use_cuda", self.use_cuda)
        if use_cuda:
            cmd.append("--use-cuda")

        return cmd

    def run_experiment(self, exp_name: str, exp_config: Dict[str, Any], exp_index: int, total_exps: int) -> bool:
        """Run a single training experiment."""
        print(f"\n{'=' * 80}")
        print(f"[{exp_index}/{total_exps}] Experiment: {exp_name}")
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

    def run_experiments(self, limit: Optional[int] = None, skip: Optional[int] = None) -> None:
        """Run all experiments from config sequentially."""
        experiments = self.get_all_experiments(limit=limit, skip=skip)

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
        "--skip",
        type=int,
        default=None,
        help="Skip first N experiments",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print experiments without running them",
    )
    parser.add_argument(
        "--tensorboard-log",
        type=str,
        default=None,
        help="TensorBoard log directory (default: logs/{config_name})",
    )
    parser.add_argument(
        "--num-vec-envs",
        type=int,
        default=None,
        help="Number of parallel environments (overrides config when set)",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2000000,
        help="Training duration in timesteps",
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
    print(f"Parallel environments per experiment: {args.num_vec_envs or 'from config'}")
    print(f"GPU/CUDA enabled: {args.use_cuda}")
    print(f"Model directory: {args.model_dir}")
    print(f"Timesteps per training: {args.total_timesteps:,}")
    if args.skip:
        print(f"Skip: first {args.skip} experiments")
    if args.limit:
        print(f"Limit: {args.limit} experiments")
    if args.dry_run:
        print(f"Mode: DRY RUN (no training will execute)")
    print(f"{'=' * 80}\n")

    # Run experiments
    runner.run_experiments(limit=args.limit, skip=args.skip)

    # Print summary
    runner.print_summary()

    return 0 if runner.failed_experiments == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
