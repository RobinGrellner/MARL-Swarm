from gymnasium import spaces
from envs.base.agent import Agent
import numpy as np

class Pursuer(Agent):
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

    def observation_space(self):
        """
        Observation space definition
            [
                [self_pos_x, self_pos_y],
                [evader]
            ]
        """
        pass