"""Zero-shot cross-size generalization rollout engine.

Loads a policy trained at one swarm size and rolls it out -- with NO further
training -- across a grid of test swarm sizes, recording per-episode reward and
the task metrics (rendezvous or pursuit-evasion). This quantifies how a
mean-embedding policy's performance holds or degrades when the swarm is
larger/smaller than what it trained on.

Why this is observationally valid (the gate):
    The per-agent observation length is fixed by ``max_agents`` (held at the
    trained value, 100), NOT by the actual agent count -- see
    ``environments/rendezvous/rendezvous_env.py`` where
    ``max_neighbours = max_agents - 1`` and the obs is
    ``local + max_neighbours*neigh_dim + mask``. Actual agents only fill
    neighbour slots; the rest are zero-padded with a binary mask. The
    ``MeanEmbeddingExtractor`` mean-pools dividing by the *masked* count, so the
    pooled statistic is consistent across N. Hence a policy trained at N=4 can be
    evaluated at N in {4..100} with identical obs length and an unbiased pool.

Wrapping note:
    We reuse ``training.common_train_utils.wrap_env_for_sb3`` verbatim so the
    eval data path matches training exactly. That helper deliberately does NOT
    apply ``VecNormalize`` -- normalisation would corrupt the binary neighbour
    mask -- so there is no normalisation state to reload.
"""

from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np
from stable_baselines3 import PPO

from environments.pursuit.pursuit_evasion_env import PursuitEvasionEnv
from environments.rendezvous.rendezvous_env import RendezvousEnv
from training.common_train_utils import wrap_env_for_sb3

# Swarm-level diagnostics exposed per-agent by RendezvousEnv._get_infos.
RENDEZVOUS_MONITOR_KEYS = ("mean_pairwise_distance", "max_pairwise_distance", "distance_to_com")
# Per-agent diagnostics from PursuitEvasionEnv._get_infos (capture rate/time are
# derived from episode termination instead -- see evaluate_variant).
PURSUIT_MONITOR_KEYS = ("min_distance_to_evader",)

TASK_MONITOR_KEYS = {
    "rendezvous": RENDEZVOUS_MONITOR_KEYS,
    "pursuit_evasion": PURSUIT_MONITOR_KEYS,
}

# Obs length is baked into the trained weights via this cap; never override it
# to the test size or the policy load / obs shape breaks.
TRAINED_MAX_AGENTS = 100

_NAN = float("nan")


@dataclass
class EpisodeResult:
    """One completed evaluation episode for one (variant, run, test_size, seed).

    Task-specific metric fields default to NaN: rendezvous episodes leave the
    pursuit fields NaN and vice versa, so both tasks share one CSV schema.
    """

    config: str          # train config stem, e.g. "embedding_scaling_rendezvous_16agents_ppo"
    variant: str         # method axis, e.g. "embed_dim16"
    run: int             # which of the 5 PPO runs (1..5)
    train_size: int      # swarm size the policy trained on
    test_size: int       # swarm size at eval (task axis)
    seed: int            # eval seed
    episode: int         # episode index within this (seed) rollout
    ep_reward: float     # mean over agent-lanes of episode return (matches rollout/ep_rew_mean scale)
    ep_length: int       # steps until termination/truncation
    converged: bool      # True if it terminated early (rendezvous reached / evader captured) vs truncated
    mean_pairwise_distance: float = _NAN  # near-terminal mean swarm spread (rendezvous)
    max_pairwise_distance: float = _NAN   # near-terminal max swarm spread (rendezvous)
    distance_to_com: float = _NAN         # near-terminal mean distance to centre of mass (rendezvous)
    min_distance_to_evader: float = _NAN  # near-terminal closest pursuer distance (pursuit-evasion)
    capture_time: float = _NAN            # steps until capture, NaN if never captured (pursuit-evasion)


def build_eval_env(
    env_config: Dict[str, Any],
    test_size: int,
    *,
    max_agents: int = TRAINED_MAX_AGENTS,
):
    """Rebuild the training env at a new swarm size.

    Overrides only the swarm size (-> ``test_size``) and forces the obs-space
    cap (``max_agents`` / ``max_pursuers``) to the trained value so the
    observation length stays identical to training. Drops the non-kwarg
    ``environment`` discriminator and silences rendering.
    """
    if test_size > max_agents:
        raise ValueError(
            f"test_size={test_size} exceeds trained obs-space cap {max_agents}; "
            "obs length is capped at the trained value and the policy cannot see more neighbours."
        )
    task = env_config.get("environment", "rendezvous")
    cfg = {k: v for k, v in env_config.items() if k != "environment"}
    if task == "pursuit_evasion":
        cfg["num_pursuers"] = test_size
        cfg["max_pursuers"] = max_agents
        cfg["render_mode"] = None
        return PursuitEvasionEnv(**cfg)
    cfg["num_agents"] = test_size
    cfg["max_agents"] = max_agents
    cfg["render_mode"] = ""
    return RendezvousEnv(**cfg)


def _terminal_metric(infos: Sequence[dict], key: str) -> float:
    """Mean of a swarm-level info key across agent lanes (identical across lanes)."""
    vals = [info[key] for info in infos if key in info]
    return float(np.mean(vals)) if vals else float("nan")


def rollout(
    model: PPO,
    vec_env,
    n_agents: int,
    n_episodes: int,
    max_steps: int,
    monitor_keys: Sequence[str] = RENDEZVOUS_MONITOR_KEYS,
) -> List[Dict[str, Any]]:
    """Run ``n_episodes`` deterministic episodes on an SB3 VecEnv of agent lanes.

    Each underlying env contributes ``n_agents`` lanes that share one episode
    and terminate together. ``monitor_keys`` names the swarm-level info keys to
    record as near-terminal metrics.

    Terminal-metric caveat: SuperSuit's ``concat_vec_envs`` auto-resets on done
    and returns *post-reset* info at the done step (verified -- it does NOT
    follow SB3's terminal-info convention). So the swarm metric in the done-step
    info belongs to a freshly randomised swarm, not the terminal state. We
    therefore record the metric captured at the *last pre-done step*, which is
    one env-step stale but correct in magnitude (the true terminal value is
    within a single step of dynamics). ``converged`` is derived from episode
    length: an early termination (rendezvous reached / evader captured) happens
    before ``max_steps``, a truncation at exactly ``max_steps``.
    """
    obs = vec_env.reset()
    ep_rewards = np.zeros(n_agents, dtype=np.float64)
    ep_len = 0
    prev_metrics = dict.fromkeys(monitor_keys, _NAN)
    episodes: List[Dict[str, Any]] = []

    while len(episodes) < n_episodes:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        ep_rewards += np.asarray(rewards, dtype=np.float64)
        ep_len += 1

        if np.any(dones):
            episodes.append(
                {
                    "ep_reward": float(ep_rewards.mean()),
                    "ep_length": ep_len,
                    "converged": bool(ep_len < max_steps),
                    **prev_metrics,
                }
            )
            ep_rewards[:] = 0.0
            ep_len = 0
            prev_metrics = dict.fromkeys(monitor_keys, _NAN)
        else:
            # Cache this step's swarm metrics; they become the terminal proxy if
            # the next step ends the episode (post-reset info would otherwise be
            # read).
            prev_metrics = {key: _terminal_metric(infos, key) for key in monitor_keys}

    return episodes


def evaluate_variant(
    zip_path: Path,
    env_config: Dict[str, Any],
    *,
    config: str,
    variant: str,
    run: int,
    train_size: int,
    test_sizes: Sequence[int],
    task: str = "rendezvous",
    n_episodes: int = 20,
    eval_seeds: Sequence[int] = (0, 1, 2),
    device: str = "cpu",
    max_agents: int = TRAINED_MAX_AGENTS,
) -> List[EpisodeResult]:
    """Evaluate one trained policy zip across all test sizes and eval seeds."""
    model = PPO.load(str(zip_path), device=device)
    max_steps = int(env_config.get("max_steps", 500))
    monitor_keys = TASK_MONITOR_KEYS[task]
    results: List[EpisodeResult] = []

    for test_size in test_sizes:
        for seed in eval_seeds:
            env = build_eval_env(env_config, test_size, max_agents=max_agents)
            vec_env = wrap_env_for_sb3(
                env, n_envs=1, monitor_keywords=monitor_keys
            )
            try:
                _seed_vec_env(vec_env, seed)
                episodes = rollout(
                    model, vec_env, n_agents=test_size, n_episodes=n_episodes,
                    max_steps=max_steps, monitor_keys=monitor_keys,
                )
            finally:
                vec_env.close()

            for ep_idx, ep in enumerate(episodes):
                metrics = {key: ep[key] for key in monitor_keys}
                if task == "pursuit_evasion":
                    # The episode ends exactly at the capture step, so the
                    # capture time (Fangzeit) is the episode length of captured
                    # episodes; uncaptured (truncated) episodes stay NaN.
                    metrics["capture_time"] = float(ep["ep_length"]) if ep["converged"] else _NAN
                results.append(
                    EpisodeResult(
                        config=config,
                        variant=variant,
                        run=run,
                        train_size=train_size,
                        test_size=test_size,
                        seed=seed,
                        episode=ep_idx,
                        ep_reward=ep["ep_reward"],
                        ep_length=ep["ep_length"],
                        converged=ep["converged"],
                        **metrics,
                    )
                )

    return results


def _seed_vec_env(vec_env, seed: int) -> None:
    """Best-effort deterministic seeding across SB3 / SuperSuit VecEnv variants."""
    try:
        vec_env.seed(seed)
    except (AttributeError, TypeError):
        # Some wrappers only accept the seed via reset(); the next reset in
        # rollout() will pick it up if action_space seeding is unavailable.
        try:
            vec_env.action_space.seed(seed)
        except (AttributeError, TypeError):
            pass


def write_raw_csv(results: Sequence[EpisodeResult], path: Path) -> Path:
    """Persist per-episode records to CSV (the raw-eval cache for the loader)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = [f.name for f in fields(EpisodeResult)]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))
    return path
