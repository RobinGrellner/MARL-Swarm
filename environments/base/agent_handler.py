from typing import List
import numpy as np
import math


class AgentHandler:
    """Handles the State of the Agents"""

    def __init__(
        self,
        *,
        num_agents: int,
        kinematics: str,
        v_max: float,
        omega_max: float,
        acc_v_max: float,
        acc_omega_max: float,
    ) -> None:

        self.num_agents = num_agents
        self.kinematics = kinematics
        self.v_max = v_max
        self.omega_max = omega_max
        self.acc_v_max = acc_v_max
        self.acc_omega_max = acc_omega_max
        self.agents: List[str] = [f"agent_{i}" for i in range(self.num_agents)]

        self._sanity_check()

        # Internal state
        self.positions: np.ndarray  # shape (num_agents, 2)
        self.linear_vels: np.ndarray  # shape (num_agents,)
        self.angular_vels: np.ndarray  # shape (num_agents,)
        self.orientations: np.ndarray  # shape (num_agents,)

    def _sanity_check(self):
        """Performs sanity checks for the initialization of the handler."""
        if self.num_agents < 1:
            raise ValueError("There must be at least one Agent present!")
        if self.kinematics not in {"single", "double"}:
            raise ValueError("kinematics must be 'single' or 'double'")

    def _clean_actions(self, actions):
        """Checks for complete Action-List and orders/Clips it for further use."""
        if set(actions.keys()) != set(self.agents):
            raise RuntimeError("Actions must be provided for all agents")

        # Convert actions to array in the same order as self.agents for handling
        clean_actions = np.array(
            [actions[agent] for agent in self.agents], dtype=np.float32
        )

        # Clip each dimension according to v_max and omega_max
        clean_actions[:, 0] = np.clip(clean_actions[:, 0], -self.v_max, self.v_max)
        clean_actions[:, 1] = np.clip(
            clean_actions[:, 1], -self.omega_max, self.omega_max
        )
        return clean_actions

    def move(self, actions):
        """
        Apply the actions of all agents to the agents and update their positions.

        Parameters
        ----------
        actions : array
            Array that contains the linear and angular actions for each agents.
        """
        actions = self._clean_actions(actions)
        lin_acs = actions[:, 0]
        ang_acs = actions[:, 1]
        if self.kinematics == "single":
            self.linear_vels = np.clip(lin_acs, -self.v_max, self.v_max)
            self.angular_vels = np.clip(ang_acs, -self.omega_max, self.omega_max)
        else:
            self.linear_vels = np.clip(
                self.linear_vels + lin_acs, -self.v_max, self.v_max
            )
            self.angular_vels = np.clip(
                self.angular_vels + ang_acs, -self.omega_max, self.omega_max
            )
        self.orientations = self.orientations + self.angular_vels
        self.orientations = (self.orientations + math.pi) % (2 * math.pi) - math.pi
        dx = (self.linear_vels * np.cos(self.orientations)).astype(np.float32)
        dy = (self.linear_vels * np.sin(self.orientations)).astype(np.float32)
        self.positions[:, 0] = self.positions[:, 0] + dx
        self.positions[:, 1] = self.positions[:, 1] + dy

    def initialize_random_positions(self, world_size):
        """Initializes the positions of the agents uniformly inside of the worlds boundaries."""
        self.positions = np.random.uniform(
            0.0, world_size, (self.num_agents, 2)
        ).astype(np.float32)
        self.linear_vels = np.zeros(self.num_agents, dtype=np.float32)
        self.angular_vels = np.zeros(self.num_agents, dtype=np.float32)
        rand_angles = np.random.uniform(-math.pi, math.pi, self.num_agents)
        self.orientations = rand_angles.astype(np.float32)
