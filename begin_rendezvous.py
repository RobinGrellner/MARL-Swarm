"""Uses Stable-Baselines3 to train agents to play the Waterworld environment using SuperSuit vector envs.

For more information, see https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html

Author: Elliot (https://github.com/elliottower)
"""

from __future__ import annotations

import numpy as np
import os

import supersuit as ss
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import VecMonitor

import envs.rendezvous.rendezvous_environment as r_env


def train_butterfly_supersuit(
    env_fn, steps: int = 10_000, cycles=1, seed: int | None = 0, model_name="model", **env_kwargs
):
    # Train a single model to play as each agent in a cooperative Parallel environment
    env = env_fn(**env_kwargs)

    env.reset()

    print(f"Starting training on {model_name}.")

    env = ss.pettingzoo_env_to_vec_env_v1(env)
    env = ss.concat_vec_envs_v1(env, 8, num_cpus=4, base_class="stable_baselines3")
    env = VecMonitor(env)
    for cycle in range(cycles):
        if os.path.exists(model_name):
            print("Loading existing model...")
            model = PPO.load(
                model_name + "/" + model_name, env=env, device="cuda"
            )  # reuse same env or redefine it here
        else:
            print("Creating new model...")
            model = PPO(
                MlpPolicy, env, verbose=1, learning_rate=1e-3, batch_size=256, device="cpu"
            )

        model.learn(total_timesteps=steps)

        if not os.path.exists(model_name):
            os.makedirs(model_name)
        model.save(model_name + "/" + model_name)

        print("Model has been saved.")

    print(f"Finished training on {model_name}.")

    env.close()


def eval(
    env_fn,
    num_games: int = 100,
    model_name="model",
    render_mode: str | None = None,
    **env_kwargs,
):
    
    # Evaluate a trained agent vs a random agent
    env = env_fn(render_mode=render_mode, **env_kwargs)

    print(
        f"\nStarting evaluation on {model_name} (num_games={num_games}, render_mode={render_mode})"
    )

    if not os.path.exists(model_name):
        print(f"Policy not found. {model_name}")
        exit(0)

    model = PPO.load(model_name + "/" + model_name, device="cpu")

    rewards = {agent: 0 for agent in env.possible_agents}

    # Note: We train using the Parallel API but evaluate using the AEC API
    # # SB3 models are designed for single-agent settings, we get around this by using he same model for every agent
    for i in range(num_games):
        print("Game: ", i)
        env.reset(seed=i)

        for agent in env.agent_iter():
            obs, reward, termination, truncation, info = env.last()

            for a in env.agents:
                rewards[a] += env.rewards[a]
            if termination or truncation:
                break
            else:
                act = model.predict(obs, deterministic=True)[0]
            if render_mode == "human":
                env.render()
            env.step(act)

    env.close()

    # env = parallel_env(max_cycles=100, world_size=(200, 200), render_mode="human")

    # for i in range(num_games):
    #     obs, info = env.reset()
    #     action = np.array([1, 9])

    #     for _ in range(env.max_cycles):

    #         # for agent in env.agents:
    #         #     print( f"Action for {agent}", model.predict(obs[agent], deterministic=True))

    #         actions = {
    #             agent: model.predict(obs[agent], deterministic=True)[0] for agent in env.agents
    #         }
    #         print("Actions: ", actions)
    #         obs, rew, terminations, truncations, infos = env.step(actions)
    #         if render_mode == "human":
    #             env.render()
    #             env.clock.tick(0.5)
    #         for agent in rewards.keys():
    #             rewards[agent] += rew[agent]
    #         if all(terminations.values()):
    #             break
    #     env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    print("Rewards: ", rewards)
    print(f"Avg reward: {avg_reward}")
    return avg_reward


if __name__ == "__main__":
    env_fn = r_env.env
    env_fn_aec = r_env.raw_env
    env_kwargs = {"max_cycles": 10_000, "num_agents": 20}
    model_name = "20_agents"

    # Train a model (takes ~3 minutes on GPU)
    # train_butterfly_supersuit(
    #     env_fn,
    #     cycles=10,
    #     steps=10_000_000,
    #     model_name=model_name,
    #     **env_kwargs,
    # )

    # Evaluate 10 games (average reward should be positive but can vary significantly)
    #eval(env_fn_aec, num_games=10, render_mode=None, model_name=model_name, **env_kwargs)

    # Watch 2 games
    eval(env_fn_aec, num_games=10, render_mode="human",model_name=model_name, **env_kwargs)
