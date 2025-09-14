"""
Training script for the Rendezvous environment using Stable‑Baselines3.

This script demonstrates how to wrap the custom :class:`RendezvousEnv` in a
vectorised format compatible with Stable‑Baselines3, then train a shared
Proximal Policy Optimisation (PPO) agent. The environment is first
converted to a vector environment using ``supersuit.pettingzoo_env_to_vec_env_v1``
and flattened so that each agent's observation space is represented as a
single continuous vector.

To customise the training run, adjust the command‑line arguments:

.. code-block:: bash

    python train_rendezvous.py --num-agents 5 --world-size 20 --obs-model local_comm --comm-radius 5 --timesteps 200000

Ensure that ``stable_baselines3``, ``pettingzoo``, ``gymnasium`` and
``supersuit`` are installed in your Python environment before running this
script.
"""

from __future__ import annotations

import argparse
import numpy as np

from environments.rendezvous.rendezvous_env import RendezvousEnv
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3 import PPO

from supersuit import (
    pettingzoo_env_to_vec_env_v1,
    flatten_v0,
    concat_vec_envs_v1,
)

from policies.meanEmbeddingExtractor import MeanEmbeddingExtractor

MODEL_SAVE_PATH = "./models/"


def parse_args() -> argparse.Namespace:
    """Parse command‑line arguments for the training script."""
    parser = argparse.ArgumentParser(
        description="Train a PPO agent on the Rendezvous environment"
    )
    parser.add_argument(
        "--num-agents", type=int, default=4, help="Number of agents in the environment"
    )
    parser.add_argument(
        "--world-size", type=float, default=50.0, help="Side length of the square world"
    )
    parser.add_argument(
        "--max-steps", type=int, default=500, help="Maximum number of steps per episode"
    )
    parser.add_argument(
        "--obs-model",
        type=str,
        default="local_basic",
        choices=[
            "global_basic",
            "global_extended",
            "local_basic",
            "local_extended",
            "local_comm",
            "classic",
        ],
        help="Observation model to use",
    )
    parser.add_argument(
        "--comm-radius",
        type=float,
        default=None,
        help="Communication radius for local observation models",
    )
    parser.add_argument(
        "--torus",
        action="store_true",
        help="Whether to wrap around the world boundaries (toroidal world)",
    )
    parser.add_argument(
        "--kinematics",
        type=str,
        default="single",
        choices=["single", "double"],
        help="Agent kinematic model",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=2_000_000,
        help="Total number of environment steps for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="Learning rate for the PPO optimiser",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="rendezvous_model.zip",
        help="File to save the trained model",
    )
    parser.add_argument(
        "--n-envs",
        type=int,
        default=1,
        help="Anzahl paralleler Envs in der SB3-VecEnv (concat_vec_envs)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # 0) Base-Env erzeugen
    print("SETUP::Starting Setup....")
    env = RendezvousEnv(
        num_agents=args.num_agents,
        world_size=args.world_size,
        max_steps=args.max_steps,
        obs_model=args.obs_model,
        comm_radius=args.comm_radius,
        torus=args.torus,
        kinematics=args.kinematics,
        render_mode="",
    )
    print("SETUP::Created Environment....")
    layout = _extract_layout_from_wrapped(env)
    local_dim = layout["local_dim"]
    neigh_dim = layout["neigh_dim"]
    max_neigh = layout["max_neighbours"]
    num_agents = args.num_agents  # oder: len(env.possible_agents)

    # 2) dann erst flatten -> vectorize -> concat zu SB3-VecEnv
    env = flatten_v0(env)  # flatten auf PettingZoo-Env
    vec_env = pettingzoo_env_to_vec_env_v1(env)
    vec_env = concat_vec_envs_v1(
        vec_env,
        num_vec_envs=args.n_envs,
        num_cpus=0,
        base_class="stable_baselines3",
    )

    vec_env = VecMonitor(vec_env)

    print("SETUP::Wrapped Environment....")

    # 3) Mean-Embedding-Extractor mit den VORHER gemerkten Werten parametrieren
    print("SETUP::Creating Mean Embedding Extractor....")
    policy_kwargs = {
        "features_extractor_class": MeanEmbeddingExtractor,
        "features_extractor_kwargs": {
            "local_dim": local_dim,
            "neigh_dim": neigh_dim,
            "max_neigh": max_neigh,
            "embed_dim": 64,
        },
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
    }

    model = PPO(
        "MlpPolicy",
        env=vec_env,
        device="cuda",  # <—
        learning_rate=args.learning_rate,
        policy_kwargs=policy_kwargs,
        verbose=1,
    )
    print("SETUP::Setup Finished!")
    print("TRAIN::Starting to train")
    callback = RewardLoggingCallback(log_per_step_mean=True)
    model.learn(total_timesteps=args.total_timesteps, callback=callback)
    print("TRAIN::Training Successful!")
    print("SAVE::Starting to train")
    model.save(MODEL_SAVE_PATH + args.model_name)
    print(f"SAVE::Model saved to {args.model_name}")


def _extract_layout_from_wrapped(env):
    """
    Geht wrapper-kette runter und liefert env.obs_layout,
    egal ob aec_to_parallel_wrapper o.ä. darüber liegt.
    """
    cur = env
    seen = set()
    for _ in range(12):  # harte Obergrenze gegen Endlosschleifen
        if hasattr(cur, "obs_layout"):
            return cur.obs_layout
        # übliche Attributnamen in PettingZoo/SuperSuit-Wrappern
        for attr in ("env", "aenv", "parallel_env", "ae_env", "unwrapped"):
            if hasattr(cur, attr):
                nxt = getattr(cur, attr)
                if nxt is not None and nxt not in seen:
                    seen.add(nxt)
                    cur = nxt
                    break
        else:
            break
    raise AttributeError(
        "obs_layout nicht in Wrapper-Kette gefunden – bitte sicherstellen, dass RendezvousEnv es setzt."
    )


from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggingCallback(BaseCallback):
    def __init__(self, log_per_step_mean: bool = True):
        super().__init__()
        self.log_per_step_mean = log_per_step_mean

    def _on_step(self) -> bool:
        # Bei On-Policy ist in self.locals meist 'rewards' und 'infos' vorhanden
        rewards = self.locals.get("rewards", None)
        infos = self.locals.get("infos", None)

        # Schritt-Reward (gemittelt über VecEnv-Instanzen)
        if self.log_per_step_mean and rewards is not None:
            self.logger.record("reward/step_mean", float(np.mean(rewards)))

        # Episoden-Return/-Länge aus VecMonitor-Infos
        if infos is not None:
            ep_rews = [
                inf["episode"]["r"]
                for inf in infos
                if isinstance(inf, dict) and "episode" in inf
            ]
            ep_lens = [
                inf["episode"]["l"]
                for inf in infos
                if isinstance(inf, dict) and "episode" in inf
            ]
            if ep_rews:
                self.logger.record("rollout/ep_rew_mean_cb", float(np.mean(ep_rews)))
                self.logger.record("rollout/ep_len_mean_cb", float(np.mean(ep_lens)))
        return True


if __name__ == "__main__":
    main()
