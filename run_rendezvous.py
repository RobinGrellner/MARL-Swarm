# render_two_episodes.py
import numpy as np
from environments.rendezvous.rendezvous_env import RendezvousEnv

USE_TRAINED = True
MODEL_PATH = "models"  # ggf. anpassen

def run_human_rollout(n_episodes: int = 2) -> None:
    env = RendezvousEnv(
        num_agents=30,
        world_size=20.0,
        max_steps=200,
        obs_model="local_basic",   # oder global_basic / local_extended / ...
        comm_radius=8.0,
        torus=False,
        kinematics="single",
        render_mode="human",
        fps=15,
    )

    model = None
    if USE_TRAINED:
        from stable_baselines3 import PPO
        model = PPO.load(MODEL_PATH, device="cpu")  # CPU reicht für MLP

    for ep in range(n_episodes):
        obs, infos = env.reset()
        ep_ret = 0.0
        step_rewards = []

        while True:
            if USE_TRAINED and model is not None:
                actions = {a: model.predict(obs[a], deterministic=True)[0] for a in env.agents}
            else:
                actions = {a: env.action_space(a).sample() for a in env.agents}

            obs, rewards, term, trunc, infos = env.step(actions)
            env.render()

            # Team-Reward (identisch für alle) mitteln
            r_t = float(np.mean(list(rewards.values())))
            ep_ret += r_t
            step_rewards.append(r_t)

            if all(term[a] or trunc[a] for a in env.agents):
                step_mean = float(np.mean(step_rewards)) if step_rewards else 0.0
                # Näherungsweise mittlere paarweise Distanz pro Schritt:
                # mean_pairwise ≈ − step_mean × d_c  (β vernachlässigt)
                d_c = getattr(env, "dc", env.world_size)
                mean_pairwise_est = - step_mean * d_c
                print(
                    f"Episode {ep+1}: return={ep_ret:.2f}, "
                    f"step_mean={step_mean:.3f}, "
                    f"~mean_pairwise_distance={mean_pairwise_est:.2f}"
                )
                break

    env.close()

if __name__ == "__main__":
    run_human_rollout(n_episodes=2)
