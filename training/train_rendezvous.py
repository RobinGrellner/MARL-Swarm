#TODO Go over everything here
from __future__ import annotations

import argparse
import os

from environments.rendezvous.rendezvous_env import RendezvousEnv
from policies.meanEmbeddingExtractor import MeanEmbeddingExtractor

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

from supersuit import pettingzoo_env_to_vec_env_v1 
from supersuit import flatten_v0 
from supersuit import concat_vec_envs_v1

def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for the training script."""
    parser = argparse.ArgumentParser(description="Train a PPO agent on the Rendezvous environment")
    parser.add_argument("--num-agents", type=int, default=4, help="Number of agents in the environment")
    parser.add_argument("--world-size", type=float, default=10.0, help="Side length of the square world")
    parser.add_argument("--max-steps", type=int, default=100, help="Maximum number of steps per episode")
    parser.add_argument("--obs-model", type=str, default="local_basic", choices=[
        "global_basic", "global_extended", "local_basic", "local_extended", "local_comm", "classic"
    ], help="Observation model to use")
    parser.add_argument("--comm-radius", type=float, default=None, help="Communication radius for local observation models")
    parser.add_argument("--torus", action="store_true", help="Whether to wrap around the world boundaries (toroidal world)")
    parser.add_argument("--break-distance-threshold", type=float, default=None, help="Early termination threshold on pairwise distances")
    parser.add_argument("--kinematics", type=str, default="single", choices=["single", "double"], help="Agent kinematic model")
    parser.add_argument("--v-max", type=float, default=1.0, help="Maximum linear velocity for agents")
    parser.add_argument("--omega-max", type=float, default=1.0, help="Maximum angular velocity for agents")
    parser.add_argument("--total-timesteps", type=int, default=200_000, help="Total number of environment steps for training")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate for the PPO optimiser")
    parser.add_argument("--num-vec-envs", type=int, default=8, help="Number of parallel environments for better GPU utilization")
    parser.add_argument("--n-steps", type=int, default=2048, help="Number of steps to collect before each PPO update")
    parser.add_argument("--batch-size", type=int, default=256, help="Minibatch size for PPO updates")
    parser.add_argument("--n-epochs", type=int, default=10, help="Number of epochs for PPO updates")
    parser.add_argument("--model-path", type=str, default="rendezvous_model.zip", help="File to save the trained model")
    parser.add_argument("--max-agents", type=int, default=None, help="Maximum number of agents to size the observation space for scale invariance")
    parser.add_argument("--resume-from", type=str, default=None, help="Path to a saved model to resume training from")
    parser.add_argument("--tensorboard-log", type=str, default=None, help="TensorBoard log directory (default: None, no logging)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    # Instantiate the custom environment
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
    # Extract observation layout parameters from the environment BEFORE
    # applying any wrappers.  The ``obs_layout`` dictionary contains
    # information about how to slice each agent's observation vector: the
    # number of local features, neighbour features per neighbour and the
    # total number of neighbour slots.  Retrieving it now avoids the
    # flatten or vector wrappers hiding this attribute.
    layout = env.obs_layout
    local_dim = layout["local_dim"]
    neigh_dim = layout["neigh_dim"]
    max_neigh = layout["max_neighbours"]

    # Flatten the per‑agent observations into 1D vectors before vectorisation.
    # SuperSuit wrappers expect a PettingZoo environment rather than a vector
    # environment, so we apply flattening prior to ``pettingzoo_env_to_vec_env_v1``.
    env = flatten_v0(env)
    # Convert the PettingZoo environment to a SuperSuit vector environment.
    vec_env = pettingzoo_env_to_vec_env_v1(env)
    # CHANGED: Convert the SuperSuit vector environment into an SB3‑compatible
    # VecEnv.  Without this step, Stable‑Baselines3 would see a MarkovVectorEnv
    # instance and raise a ValueError.  The ``concat_vec_envs_v1`` wrapper
    # concatenates the vector environment into ``num_vec_envs`` copies and
    # assigns ``base_class="stable_baselines3"`` to ensure compatibility.
    vec_env = concat_vec_envs_v1(
        vec_env,
        num_vec_envs=args.num_vec_envs,
        num_cpus=0,
        base_class="stable_baselines3",
    )
    # CHANGED: Wrap the SB3 VecEnv with VecMonitor to record episode statistics.
    # This wrapper collects episode returns and lengths, which are useful
    # diagnostics when training.  If desired, additional info keys could be
    # passed via ``info_keywords``.
    vec_env = VecMonitor(vec_env)

    # CHANGED: Optionally normalise observations and rewards using VecNormalize.
    # When resuming training, load existing VecNormalize statistics to maintain
    # consistent normalization. Otherwise, create fresh statistics.
    if args.resume_from:
        # Try to load existing VecNormalize statistics
        vecnorm_path = args.resume_from.replace(".zip", "_vecnormalize.pkl")
        if os.path.exists(vecnorm_path):
            print(f"Loading VecNormalize stats from {vecnorm_path}...")
            vec_env = VecNormalize.load(vecnorm_path, vec_env)
            # Keep training=True to continue updating statistics
            vec_env.training = True
            vec_env.norm_reward = True
            print("VecNormalize loaded! Continuing with accumulated statistics.")
        else:
            print(f"⚠️  Warning: VecNormalize stats not found at {vecnorm_path}")
            print("⚠️  Creating fresh VecNormalize - normalization will restart from scratch.")
            vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    else:
        # Create fresh VecNormalize for new training
        vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.0)

    # NOTE: We have already extracted the observation layout prior to wrapping the
    # environment.  No further retrieval is necessary here; see above for
    # details.

    # ------------------------------------------------------------------
    # Custom feature extractor for mean embeddings
    # ------------------------------------------------------------------
    # Using MeanEmbeddingExtractor from policies module.
    # The extractor operates on a single agent's observation produced by the
    # vectorised PettingZoo environment.  Each observation is a flat vector
    # consisting of local features, neighbour feature blocks, and a mask.
    # It applies a neural network φ to each neighbour independently and
    # averages the embeddings using the mask before concatenating with
    # local features to produce a fixed-length representation.

    # Set up policy keyword arguments
    policy_kwargs = {
        # Use the custom mean embedding extractor defined above.
        "features_extractor_class": MeanEmbeddingExtractor,
        "features_extractor_kwargs": {
            "local_dim": local_dim,
            "neigh_dim": neigh_dim,
            "max_neigh": max_neigh,
            # CHANGED: removed num_agents parameter since the extractor
            # operates on individual agent observations.  The embedding
            # dimension is fixed to 64 as in Hüttenrauch et al.  (2019)【627991056406116†L684-L699】.
            "embed_dim": 64,
        },
        # Separate network architectures for the policy and value heads.
        # Updated format for SB3 v1.8.0+
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
    }

    # Create or load the PPO model
    if args.resume_from:
        print(f"Loading model from {args.resume_from} to resume training...")
        model = PPO.load(args.resume_from, env=vec_env, tensorboard_log=args.tensorboard_log)
        print(f"Model loaded! Current timesteps: {model.num_timesteps:,}")
        print(f"Will train for {args.total_timesteps:,} additional steps")
        if args.tensorboard_log:
            print(f"TensorBoard logging to: {args.tensorboard_log}")
    else:
        print("Creating new PPO model...")
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            verbose=1,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=args.tensorboard_log,
            # TRPO-like constraints for stable learning
            clip_range=0.1,  # More conservative than default 0.2
            target_kl=0.015,  # Early stopping if KL divergence too large
            gamma=0.99,
            gae_lambda=0.98,
        )
        if args.tensorboard_log:
            print(f"TensorBoard logging enabled: {args.tensorboard_log}")

    # Train the model
    # If resuming, reset_num_timesteps=False keeps the counter going
    model.learn(
        total_timesteps=args.total_timesteps,
        reset_num_timesteps=(args.resume_from is None)
    )

    # Save the trained policy
    model.save(args.model_path)
    print(f"\nModel saved to {args.model_path}")

    # Save VecNormalize statistics for proper evaluation
    # These statistics are crucial when evaluating on different swarm sizes
    # to ensure observations are normalized with the SAME mean/std as during training
    vecnorm_path = args.model_path.replace(".zip", "_vecnormalize.pkl")
    vec_env.save(vecnorm_path)
    print(f"VecNormalize stats saved to {vecnorm_path}")
    print(f"  → Use these when evaluating to maintain consistent normalization")

    print(f"Total timesteps: {model.num_timesteps:,}")


if __name__ == "__main__":
    main()