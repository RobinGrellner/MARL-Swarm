"""
Test script for the Rendezvous environment.

This script instantiates a :class:`RendezvousEnv` with a set of parameters
and executes a short rollout using randomly sampled actions. It prints the
observations' shapes, rewards at each timestep and detects early termination
or truncation conditions. The goal of this script is to provide a quick
sanity check that the environment behaves as expected.
"""

from __future__ import annotations

import numpy as np

from environments.rendezvous.rendezvous_env import RendezvousEnv


def main() -> None:
    # Create the environment. Adjust the parameters here to test different
    # observation models, communication radii, kinematic models, etc.
    env = RendezvousEnv(
        num_agents=4,
        world_size=10.0,
        max_steps=50,
        kinematics="single",
        obs_model="local_basic",
        comm_radius=5.0,
        torus=False,
        break_distance_threshold=None,
    )
    # Reset the environment and inspect the initial observation structure
    observations, infos = env.reset(seed=123)
    print("Initial observation shapes:")
    for agent, obs in observations.items():
        print(f"  {agent}: {obs.shape}")
    # Perform a short rollout with random actions
    for step in range(10):
        actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        observations, rewards, terminations, truncations, infos = env.step(actions)
        print(f"Step {step} rewards: {rewards}")
        # Check for early termination
        if any(terminations.values()):
            print("Episode terminated early due to break distance threshold.")
            break
        if any(truncations.values()):
            print("Episode truncated due to reaching max_steps.")
            break
    env.close()


if __name__ == "__main__":
    main()