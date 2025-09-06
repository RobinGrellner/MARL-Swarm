from gymnasium import spaces
from envs.base.agent import Agent
import numpy as np

class RendezvousAgent(Agent):
    def action_space(self):
        """
        Action space definition
            [vel_delta, phi_delta]
        with    -3 < vel_delta < 3
                -20 < phi_delta < -20
        """
        return spaces.Box(
            low=np.array(-3, -20), high=np.array(3, 20), shape=(2,), dtype=np.float64
        )

    def observation_space(self, possible_agents):
        if self.obs_mode == "global":
            # Observed features per neighbor: 2; Local observed features: 2
            # Also, no zero padding because global observability
            obs_dim = (len(possible_agents) - 1) * 2 + 2
            return spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,),dtype=np.float32)
