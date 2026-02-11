"""Common training utilities for MARL environments (Rendezvous and Pursuit-Evasion).

This module provides shared functions for training RL models with mean embedding
feature extraction across different multi-agent environments.
"""

from __future__ import annotations

import argparse
import copy
from typing import Any, Dict, List, Optional, Tuple
import warnings

from supersuit import flatten_v0 as supersuit_flatten
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, SubprocVecEnv
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.utils import safe_mean
from sb3_contrib import TRPO
from policies.mean_embedding_extractor import MeanEmbeddingExtractor

import torch
import numpy as np
import time
from collections import deque
from stable_baselines3.common.callbacks import BaseCallback
import psutil
import gc
import os

# Suppress render_mode warning from VecEnv
warnings.filterwarnings("ignore", message=".*render_mode.*", category=UserWarning)


class MALRMetricsCallback(BaseCallback):
    """Custom callback to log MARL-specific metrics for thesis analysis.

    Tracks:
    - Reward statistics (mean, std, success rate)
    - Training efficiency (timesteps/second)
    - Episode diversity across vectorized environments
    - Action entropy for coordination analysis

    OPTIMIZATION: Expensive statistics computations moved to _on_rollout_end()
    instead of _on_step() to avoid 2000+ redundant NumPy operations per rollout.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.start_time = time.time()
        self.reward_buffer = deque(maxlen=100)  # Last 100 episodes for rolling stats
        self.episode_lengths = deque(maxlen=100)
        self.ep_info_buffer = deque(maxlen=100)  # For success rate tracking

    def _on_step(self) -> bool:
        """Called after every environment step - collect data only, defer computation."""
        # Collect episode info from VecMonitor (cheap operation)
        if hasattr(self.model.env, "ep_info_buffer"):
            for ep_info in self.model.env.ep_info_buffer:
                self.ep_info_buffer.append(ep_info)
                self.reward_buffer.append(ep_info["r"])
                self.episode_lengths.append(ep_info["l"])

        return True

    def _on_rollout_end(self) -> None:
        """Called once per rollout (after n_steps environment steps).

        Compute expensive statistics here instead of every step.
        With n_steps=2048, this reduces computations from 2048x to 1x per rollout.
        """
        # Log rolling reward statistics
        if len(self.reward_buffer) > 1:
            reward_mean = np.mean(list(self.reward_buffer))
            reward_std = np.std(list(self.reward_buffer))
            self.logger.record("rollout/ep_rew_mean_rolling", reward_mean)
            self.logger.record("rollout/ep_rew_std", reward_std)

        # Log episode length statistics
        if len(self.episode_lengths) > 1:
            ep_len_mean = np.mean(list(self.episode_lengths))
            ep_len_std = np.std(list(self.episode_lengths))
            self.logger.record("rollout/ep_len_mean_rolling", ep_len_mean)
            self.logger.record("rollout/ep_len_std", ep_len_std)

        # Log training efficiency
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            timesteps_per_second = self.model.num_timesteps / elapsed_time
            self.logger.record("time/timesteps_per_second", timesteps_per_second)

        # Log task-specific success rate if available in ep_info
        if len(self.ep_info_buffer) > 0:
            # Check if any info contains task_success
            success_list = [ep_info.get("task_success", None) for ep_info in self.ep_info_buffer
                           if "task_success" in ep_info]
            if success_list:
                success_rate = np.mean(success_list)
                self.logger.record("task/success_rate", success_rate)

            # Track task-specific metrics if available
            convergence_vel = [ep_info.get("convergence_velocity", None) for ep_info in self.ep_info_buffer
                              if "convergence_velocity" in ep_info]
            if convergence_vel:
                self.logger.record("task/convergence_velocity_mean", np.mean(convergence_vel))

            capture_times = [ep_info.get("capture_time", None) for ep_info in self.ep_info_buffer
                            if "capture_time" in ep_info]
            if capture_times:
                self.logger.record("task/capture_time_mean", np.mean(capture_times))


class IterationCounterCallback(BaseCallback):
    """Callback to log training metrics with iteration as the x-axis step.

    SB3 on-policy algorithms call ``_on_rollout_end()`` once per iteration
    (after collecting a rollout and before the policy update).  This callback:

    1. Logs ``train/iteration`` via the standard SB3 logger (x-axis = timesteps).
    2. Writes ``by_iter/<metric>`` scalars to the **same** TensorBoard writer
       that SB3 uses, but with ``global_step=iteration``.  In TensorBoard the
       ``by_iter/`` group gives iteration-based x-axes while the standard
       metrics keep timestep-based x-axes — all in one log directory.

    Timing note: ``_on_rollout_end`` fires *before* SB3 records rollout and
    train metrics, so rollout stats are computed directly from the episode
    buffer, and train metrics from the previous iteration are captured during
    ``_on_step``.
    """

    # Metrics to snapshot from name_to_value (captured one iteration behind).
    _SNAPSHOT_METRICS = (
        # SB3 train metrics
        "train/explained_variance",
        "train/value_loss",
        "train/policy_objective",
        "train/kl_divergence_loss",
        "train/is_line_search_success",
        "train/learning_rate",
        "train/n_updates",
        "train/std",
        # MALRMetricsCallback custom metrics (ep_rew_std/ep_len_std already
        # computed directly from episode buffer in _on_rollout_end)
        "rollout/ep_rew_mean_rolling",
        "rollout/ep_len_mean_rolling",
        "task/success_rate",
        "task/convergence_velocity_mean",
        "task/capture_time_mean",
    )

    def __init__(self, verbose: int = 0, flush_every: int = 10):
        super().__init__(verbose)
        self.iteration = 0
        self._writer = None  # Resolved lazily from SB3 logger
        self._prev_train_metrics = {}  # Metrics snapshotted from previous iteration
        self._flush_every = flush_every  # Flush TensorBoard every N iterations (not every step)

    def _get_writer(self):
        """Get the SummaryWriter from SB3's TensorBoard output format."""
        if self._writer is not None:
            return self._writer
        from stable_baselines3.common.logger import TensorBoardOutputFormat
        for fmt in self.logger.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                self._writer = fmt.writer
                return self._writer
        return None

    def _on_step(self) -> bool:
        """Snapshot metrics after they've been recorded by train()/callbacks."""
        # After the first rollout+train cycle, name_to_value briefly holds
        # values before dump() clears them.  Capture any we see.
        for key in self._SNAPSHOT_METRICS:
            val = self.logger.name_to_value.get(key)
            if val is not None:
                self._prev_train_metrics[key] = val
        return True

    def _on_rollout_end(self) -> None:
        self.iteration += 1
        # Standard log (x-axis = timesteps)
        self.logger.record("train/iteration", self.iteration)

        writer = self._get_writer()
        if writer is None:
            return

        # --- Rollout metrics: compute directly from episode buffer ---
        if len(self.model.ep_info_buffer) > 0:
            rewards = [ep["r"] for ep in self.model.ep_info_buffer]
            lengths = [ep["l"] for ep in self.model.ep_info_buffer]
            writer.add_scalar("by_iter/rollout/ep_rew_mean", safe_mean(rewards), global_step=self.iteration)
            writer.add_scalar("by_iter/rollout/ep_len_mean", safe_mean(lengths), global_step=self.iteration)
            if len(rewards) > 1:
                writer.add_scalar("by_iter/rollout/ep_rew_std", float(np.std(rewards)), global_step=self.iteration)
                writer.add_scalar("by_iter/rollout/ep_len_std", float(np.std(lengths)), global_step=self.iteration)

        # --- Snapshotted metrics: write values captured from previous iteration ---
        for key, val in self._prev_train_metrics.items():
            writer.add_scalar(f"by_iter/{key}", val, global_step=self.iteration)
        self._prev_train_metrics.clear()

        # Flush periodically to reduce blocking I/O (every 10 iterations instead of every iteration)
        if self.iteration % self._flush_every == 0:
            writer.flush()

    def _on_training_end(self) -> None:
        """Ensure any remaining TensorBoard data is flushed at training end."""
        writer = self._get_writer()
        if writer is not None:
            writer.flush()


class CheckpointCallback(BaseCallback):
    """Callback for saving model checkpoints periodically during training.

    Saves the model every `save_freq` timesteps to prevent loss of progress
    in case of interruption.
    """

    def __init__(self, save_freq: int = 1_000_000, save_path: str = "models/", verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.last_save_step = 0

        # Create save directory if it doesn't exist
        import os
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        """Called after every environment step."""
        # Check if it's time to save
        if self.model.num_timesteps - self.last_save_step >= self.save_freq:
            # Save checkpoint with timestep in filename
            checkpoint_path = f"{self.save_path}{self.model.__class__.__name__}_checkpoint_{self.model.num_timesteps}"
            self.model.save(checkpoint_path)
            if self.verbose > 0:
                print(f"Checkpoint saved to {checkpoint_path}")
            self.last_save_step = self.model.num_timesteps

        return True


class MemoryDiagnosticCallback(BaseCallback):
    """Callback for monitoring memory usage and timing metrics per rollout.

    Logs memory-related metrics at the end of each rollout to detect:
    - Memory leaks (RSS growing monotonically)
    - Memory fragmentation (VMS diverging from RSS)
    - Garbage collection pressure (gen0/gen2 counts growing)
    - Resource leaks (file handles, child processes)
    - Performance degradation (rollout wall-clock time increasing)
    """

    def __init__(self, verbose: int = 0, log_every_n_rollouts: int = 10):
        super().__init__(verbose)
        self._process = psutil.Process()
        self._log_every_n_rollouts = log_every_n_rollouts
        self._rollout_count = 0
        self._rollout_start_time = None
        self._initial_rss_mb = None

    def _on_training_start(self) -> None:
        """Initialize diagnostic state at training start."""
        self._initial_rss_mb = self._process.memory_info().rss / 1024 / 1024
        if self.verbose > 0:
            print(f"[DIAG] Initial memory: {self._initial_rss_mb:.1f} MB")

    def _on_rollout_start(self) -> None:
        """Record timestamp at rollout start."""
        self._rollout_start_time = time.time()

    def _on_step(self) -> bool:
        """Called after every environment step. Returns True to continue training."""
        return True

    def _on_rollout_end(self) -> None:
        """Log memory and timing metrics at rollout end."""
        self._rollout_count += 1

        # Only log periodically to reduce I/O overhead
        if self._rollout_count % self._log_every_n_rollouts != 0:
            return

        # Collect all diagnostic metrics
        try:
            mem_info = self._process.memory_info()
            rss_mb = mem_info.rss / 1024 / 1024
            vms_mb = mem_info.vms / 1024 / 1024
            num_handles = self._process.num_handles()
            num_children = len(self._process.children(recursive=True))
            gc_counts = gc.get_count()  # (gen0, gen1, gen2)
            num_objects = len(gc.get_objects())
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return

        # Calculate rollout wall-clock time
        rollout_wall_time_s = None
        if self._rollout_start_time is not None:
            rollout_wall_time_s = time.time() - self._rollout_start_time

        # Log memory metrics
        self.logger.record("diag/rss_mb", rss_mb)
        self.logger.record("diag/vms_mb", vms_mb)
        self.logger.record("diag/num_handles", num_handles)
        self.logger.record("diag/num_children", num_children)
        self.logger.record("diag/gc_gen0", gc_counts[0])
        self.logger.record("diag/gc_gen2", gc_counts[2])
        self.logger.record("diag/num_python_objects", num_objects)

        if rollout_wall_time_s is not None:
            self.logger.record("diag/rollout_wall_time_s", rollout_wall_time_s)

        # Print periodic summary to console
        if self.verbose > 0 and self._rollout_count % (10 * self._log_every_n_rollouts) == 0:
            memory_delta_mb = rss_mb - self._initial_rss_mb if self._initial_rss_mb else 0
            print(
                f"[DIAG] Rollout {self._rollout_count}: "
                f"RSS={rss_mb:.1f} MB (d{memory_delta_mb:+.1f}), "
                f"Handles={num_handles}, "
                f"Rollout_time={rollout_wall_time_s:.2f}s"
            )


def parse_policy_layers(layers_str: str) -> List[int]:
    """Parse a comma-separated string of integers into a list.

    Args:
        layers_str: Comma-separated string of layer sizes (e.g., "64,64" or "128,128,64").

    Returns:
        List of integers representing layer sizes.

    Raises:
        argparse.ArgumentTypeError: If parsing fails or values are not positive integers.
    """
    try:
        layers = [int(x.strip()) for x in layers_str.split(",")]
        if not layers:
            raise ValueError("Empty layer list")
        if any(x <= 0 for x in layers):
            raise ValueError("Layer sizes must be positive integers")
        return layers
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid policy-layers format: '{layers_str}'. "
            f"Expected comma-separated positive integers (e.g., '64,64' or '128,128'). "
            f"Error: {e}"
        )


def add_common_training_args(parser: argparse.ArgumentParser) -> None:
    """Add common training and architecture arguments to an argument parser.

    These arguments are shared across all training scripts (architecture, algorithm, training hyperparameters).

    Args:
        parser: ArgumentParser to add arguments to.
    """
    # Architecture parameters
    parser.add_argument(
        "--activation",
        type=str,
        default="relu",
        choices=["relu", "tanh", "gelu", "leaky_relu", "elu"],
        help="Activation function for the feature extractor (default: relu)",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "max", "sum", "attention"],
        help="Aggregation operation for neighbor embeddings (default: mean)",
    )
    parser.add_argument(
        "--policy-layers",
        type=parse_policy_layers,
        default="64",
        help="Comma-separated list of hidden layer sizes for policy and value networks (default: 64)",
    )
    parser.add_argument("--embed-dim", type=int, default=64, help="Dimensionality of the mean embedding (default: 64)")
    parser.add_argument("--phi-layers", type=int, default=1, help="Number of hidden layers in the phi network (default: 1)")
    parser.add_argument("--phi-hidden-width", type=int, default=None, help="Width of phi hidden layers (default: max(64, embed_dim))")

    # Training parameters
    parser.add_argument("--algorithm", type=str, default="trpo", choices=["ppo", "trpo"], help="RL algorithm to use (default: TRPO for paper fidelity)")
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
    parser.add_argument("--model-path", type=str, default=None, help="Path to save the trained model")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to a saved model to resume training from")
    parser.add_argument("--tensorboard-log", type=str, default=None, help="TensorBoard log directory")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--use-cuda", action="store_true", help="Use CUDA/GPU for training")


def build_algo_params(args: argparse.Namespace, algorithm: str) -> Dict[str, Any]:
    """Build algorithm hyperparameters dictionary from parsed arguments.

    Args:
        args: Parsed command-line arguments
        algorithm: "ppo" or "trpo"

    Returns:
        Dictionary of algorithm hyperparameters
    """
    algo_params = {"verbose": 1}
    if args.learning_rate is not None:
        algo_params["learning_rate"] = args.learning_rate
    if args.n_steps is not None:
        algo_params["n_steps"] = args.n_steps
    if args.batch_size is not None:
        algo_params["batch_size"] = args.batch_size
    if args.n_epochs is not None and algorithm == "ppo":
        algo_params["n_epochs"] = args.n_epochs
    if args.tensorboard_log is not None:
        algo_params["tensorboard_log"] = args.tensorboard_log
    if args.seed is not None:
        algo_params["seed"] = args.seed

    # Handle GPU device with availability check
    if args.use_cuda:
        if torch.cuda.is_available():
            algo_params["device"] = "cuda"
        else:
            warnings.warn(
                "CUDA/GPU requested via --use-cuda but CUDA is not available. "
                "Falling back to CPU training.",
                UserWarning
            )
            algo_params["device"] = "cpu"
    else:
        algo_params["device"] = "cpu"

    return algo_params


def build_embed_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Build embedding configuration dictionary from parsed arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Dictionary of embedding configuration
    """
    return {
        "embed_dim": args.embed_dim,
        "phi_layers": args.phi_layers,
        "phi_hidden_width": args.phi_hidden_width,
        "activation": args.activation,
        "aggregation": args.aggregation,
        "policy_layers": args.policy_layers,
    }


def wrap_env_for_sb3(
    env, *, n_envs: int = 1, num_cpus: Optional[int] = None, monitor_keywords: Optional[Tuple[str, ...]] = None, normalize: bool = True
) -> VecMonitor:
    """Wrap a PettingZoo environment for Stable-Baselines3 training.

    The wrapping sequence is as follows:

    1. Flatten per-agent observations into single feature vectors
    2. Convert flattened PettingZoo env to a vectorized environment (MarkovVectorEnv)
    3. Concatenate n_envs copies using concat_vec_envs_v1 with synchronous execution
    4. Apply VecMonitor to record episode statistics

    DESIGN DECISION: Use num_cpus=0 (synchronous) for stability and reliability.

    Rationale:
    - Environment instances contain unpickleable objects (pygame resources, cached state)
    - Windows multiprocessing uses 'spawn' context which requires full serialization
    - Subprocess parallelism overhead exceeds benefits for NumPy-based physics
    - Training bottleneck is GPU policy updates, not environment stepping

    With num_cpus=0:
    - Still get n_envs environment copies (training diversity via auto-reset desynchronization)
    - No IPC overhead (observations/actions stay in main process)
    - No pickling failures or subprocess crashes
    - Environments naturally desynchronize through independent episode resets

    If true subprocess parallelism is needed later, use factory function pattern
    (pass environment class+kwargs, not instances).

    :param env: the PettingZoo environment to wrap (RendezvousEnv or PursuitEvasionEnv)
    :param n_envs: number of parallel environment copies
    :param num_cpus: number of worker processes (default: None = 0 for synchronous execution)
    :param monitor_keywords: optional tuple of keys from env.info to record
    :param normalize: whether to apply observation and reward normalisation (deprecated, kept for API compatibility)
    :return: a wrapped ``VecEnv`` ready for SB3
    """
    # Default to synchronous execution for stability
    if num_cpus is None:
        num_cpus = 0

    # Flatten PettingZoo observations to 1D per agent
    env_flat = supersuit_flatten(env)
    # Convert to a SuperSuit vector env
    vec_env = pettingzoo_env_to_vec_env_v1(env_flat)
    # Concatenate into an SB3 VecEnv with synchronous execution
    vec_env = concat_vec_envs_v1(vec_env, num_vec_envs=n_envs, num_cpus=num_cpus, base_class="stable_baselines3")

    # Apply VecMonitor to log episode statistics
    if monitor_keywords is not None:
        vec_env = VecMonitor(vec_env, info_keywords=monitor_keywords)
    else:
        vec_env = VecMonitor(vec_env)

    # VecNormalize removed: breaks binary masks and rewards already normalized by design
    return vec_env


def make_policy_kwargs(
    layout: Dict[str, int],
    *,
    embed_dim: int = 64,
    phi_layers: int = 1,
    phi_hidden_width: Optional[int] = None,
    policy_layers: Optional[List[int]] = None,
    activation: str = "relu",
    aggregation: str = "mean",
) -> Dict[str, Any]:
    """Build policy keyword arguments for Stable-Baselines3 algorithms.

    This function constructs the configuration dictionary for the MeanEmbeddingExtractor
    feature extractor and the policy/value network architectures.

    Args:
        layout: Dictionary containing observation layout with keys:
            - "local_dim": Number of local features
            - "neigh_dim": Number of features per neighbor
            - "max_neighbours": Maximum number of neighbors
        embed_dim: Dimension of the neighbor embedding (default: 64).
        phi_layers: Number of hidden layers in the phi network (default: 1).
        policy_layers: List of hidden layer sizes for both policy (pi) and value (vf)
            networks. If None, defaults to [64, 64].
        activation: Activation function for the feature extractor. One of "relu",
            "tanh", "gelu", "leaky_relu", "elu" (default: "relu").
        aggregation: Aggregation operation for neighbor embeddings. One of "mean",
            "max", "sum", "attention" (default: "mean").

    Returns:
        Dictionary of policy kwargs suitable for SB3 algorithm constructors.

    Example:
        >>> layout = {"local_dim": 4, "neigh_dim": 2, "max_neighbours": 5}
        >>> kwargs = make_policy_kwargs(layout, policy_layers=[128, 128], activation="tanh")
        >>> model = PPO("MlpPolicy", env, policy_kwargs=kwargs)
    """
    local_dim = layout["local_dim"]
    neigh_dim = layout["neigh_dim"]
    max_neigh = layout["max_neighbours"]

    # Determine hidden layer configuration for phi network
    width = phi_hidden_width if phi_hidden_width is not None else max(64, embed_dim)
    phi_hidden = [width] * max(phi_layers, 1)

    # Use default policy layers if not specified
    # Scale policy width with embed_dim to avoid bottleneck for large embeddings
    if policy_layers is None:
        policy_width = max(64, embed_dim)
        policy_layers = [policy_width]

    return {
        "features_extractor_class": MeanEmbeddingExtractor,
        "features_extractor_kwargs": {
            "local_dim": local_dim,
            "neigh_dim": neigh_dim,
            "max_neigh": max_neigh,
            "embed_dim": embed_dim,
            "phi_hidden": phi_hidden,
            "activation": activation,
            "aggregation": aggregation,
        },
        "net_arch": dict(pi=list(policy_layers), vf=list(policy_layers)),
    }


def setup_model(
    vec_env: VecMonitor, policy_kwargs: Dict[str, Any], algo_params: Dict[str, Any], algorithm: str = "ppo"
) -> BaseAlgorithm:
    """Setup RL model (PPO or TRPO) with given parameters.

    Args:
        vec_env: Vectorized environment
        policy_kwargs: Policy network configuration
        algo_params: Algorithm hyperparameters
        algorithm: "ppo" or "trpo"

    Returns:
        Configured RL model
    """
    algo_params = algo_params.copy()  # Don't modify original dict

    # Extract seed if present and set global random state
    # (ConcatVecEnv doesn't support seed() method, so we seed globally instead)
    seed = algo_params.pop("seed", None)
    if seed is not None:
        import numpy as np
        np.random.seed(seed)
        torch.manual_seed(seed)

    # Set device if not already provided
    if "device" not in algo_params:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        algo_params["device"] = device
    else:
        device = algo_params["device"]

    # Print device information
    print(f"\n{'=' * 60}")
    print(f"Algorithm: {algorithm.upper()}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"Selected Device: {device}")
    if seed is not None:
        print(f"Random Seed: {seed}")
    print(f"{'=' * 60}\n")

    algorithm = algorithm.lower()
    if algorithm == "ppo":
        return PPO(
            policy="MlpPolicy",
            env=vec_env,
            policy_kwargs=policy_kwargs,
            **algo_params,
        )
    elif algorithm == "trpo":
        return TRPO(
            policy="MlpPolicy",
            env=vec_env,
            policy_kwargs=policy_kwargs,
            **algo_params,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Choose 'ppo' or 'trpo'.")


def get_default_algo_params(algorithm: str) -> Dict[str, Any]:
    """Get default hyperparameters for an algorithm.

    Args:
        algorithm: "ppo" or "trpo"

    Returns:
        Dictionary of default hyperparameters
    """
    algorithm = algorithm.lower()
    if algorithm == "ppo":
        return {
            "learning_rate": 3e-4,
            "n_steps": 2048,  # Optimized: longer rollouts for better value estimates
            "batch_size": 2048,  # Optimized: larger batches for GPU utilization
            "n_epochs": 4,  # Reduced: larger batches compensate for fewer epochs
            "gamma": 0.99,
            "gae_lambda": 0.98,
            "clip_range": 0.2,
            "target_kl": 0.015,
            "verbose": 1,
        }
    elif algorithm == "trpo":
        # Match Huttenrauch's TRPO hyperparameters
        return {
            "learning_rate": 1e-3,  # vf_stepsize
            "n_steps": 2048,  # timesteps_per_batch
            "batch_size": 128,
            "gamma": 0.99,
            "gae_lambda": 0.98,
            "n_critic_updates": 5,  # vf_iters
            "cg_max_steps": 10,  # cg_iters
            "cg_damping": 0.1,
            "target_kl": 0.01,  # max_kl
            "verbose": 1,
        }
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")


def run_training(
    env,
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
    """Generic training function for any MARL environment.

    Works with any environment that has an `obs_layout` attribute and implements
    the PettingZoo ParallelEnv interface.

    Args:
        env: Environment instance (RendezvousEnv, PursuitEvasionEnv, or compatible)
        embed_config: Configuration for mean embedding extractor. Supported keys:
            - "embed_dim": Embedding dimension (default: 64)
            - "phi_layers": Number of phi network layers (default: 1)
            - "policy_layers": List of policy/value network hidden layer sizes (default: [64, 64])
            - "activation": Activation function name (default: "relu")
            - "aggregation": Aggregation operation name (default: "mean")
        algo_params: Algorithm hyperparameters (PPO or TRPO)
        algorithm: "ppo" or "trpo"
        total_timesteps: Total training timesteps
        log_info_keys: Keys to log from env info dict
        n_envs: Number of parallel environments
        normalize: Whether to apply VecNormalize (deprecated, kept for API compatibility)
        save_path: Path to save model (optional)
        resume_from: Path to a saved model to resume training from (optional)

    Returns:
        Tuple of (trained model, info dict with vec_env and environment config)
    """
    # 1. Extract layout before wrapping (avoid wrapper hiding attributes)
    layout = env.obs_layout

    # 2. Wrap environment into vector form
    vec_env = wrap_env_for_sb3(env, n_envs=n_envs, monitor_keywords=log_info_keys, normalize=normalize)

    # 3. Handle resume vs fresh training
    algorithm = algorithm.lower()
    default_params = {}  # Initialize to avoid UnboundLocalError when resuming

    if resume_from:
        # Load a previously trained model and continue training
        print(f"\n{'=' * 60}")
        print(f"Resuming training from: {resume_from}")
        print(f"{'=' * 60}\n")

        # Determine algorithm from file
        resume_algorithm = algorithm
        if resume_algorithm == "ppo":
            model = PPO.load(resume_from, env=vec_env, device=algo_params.get("device", "cpu"))
        elif resume_algorithm == "trpo":
            model = TRPO.load(resume_from, env=vec_env, device=algo_params.get("device", "cpu"))
        else:
            raise ValueError(f"Unknown algorithm: {resume_algorithm}")

        # Update learning rate if provided
        if "learning_rate" in algo_params:
            model.learning_rate = algo_params["learning_rate"]

        # Store algo_params for info dict
        default_params = algo_params
    else:
        # Create a new model from scratch
        # Build policy kwargs from layout and embedding configuration
        policy_kwargs = make_policy_kwargs(
            layout,
            embed_dim=embed_config.get("embed_dim", 64),
            phi_layers=embed_config.get("phi_layers", 1),
            phi_hidden_width=embed_config.get("phi_hidden_width", None),
            policy_layers=embed_config.get("policy_layers", None),
            activation=embed_config.get("activation", "relu"),
            aggregation=embed_config.get("aggregation", "mean"),
        )

        # Get default parameters based on algorithm
        default_params = get_default_algo_params(algorithm)

        # Update defaults with supplied parameters
        default_params.update(algo_params)

        # Create model
        model = setup_model(vec_env, policy_kwargs, default_params, algorithm=algorithm)

    # 6. Create callbacks for training
    from stable_baselines3.common.callbacks import CallbackList

    metrics_callback = MALRMetricsCallback(verbose=0)
    iteration_callback = IterationCounterCallback(verbose=0)
    memory_callback = MemoryDiagnosticCallback(verbose=1, log_every_n_rollouts=10)

    # Create checkpoint callback with automatic directory naming
    checkpoint_dir = save_path.replace(".zip", "_checkpoints/") if save_path else "models/checkpoints/"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_callback = CheckpointCallback(save_freq=1_000_000, save_path=checkpoint_dir, verbose=1)

    callbacks = CallbackList([metrics_callback, iteration_callback, memory_callback, checkpoint_callback])

    # 7. Train model with callbacks (in try/finally to ensure vec_env cleanup)
    try:
        model.learn(total_timesteps=total_timesteps, callback=callbacks)
    finally:
        # Always close vec_env to prevent zombie subprocesses
        vec_env.close()

    # 8. Save final model if path provided
    if save_path:
        model.save(save_path)
        print(f"\nFinal model saved to {save_path}")

    # Collect diagnostic info (NOTE: vec_env is closed, do not store reference)
    info = {
        "layout": layout,
        "embed_config": embed_config,
        "algo_params": default_params,
        "algorithm": algorithm,
    }
    return model, info
