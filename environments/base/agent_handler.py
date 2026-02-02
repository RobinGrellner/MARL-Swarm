from typing import List, Optional
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
        dt: float = 0.1,
    ) -> None:
        self.num_agents = num_agents
        self.kinematics = kinematics
        self.v_max = v_max
        self.omega_max = omega_max
        self.acc_v_max = acc_v_max
        self.acc_omega_max = acc_omega_max
        self.dt = dt  # Time step for integration
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

    def _clean_actions(self, actions: dict) -> np.ndarray:
        """Convert normalized actions to physical velocities/accelerations.

        The RL policy outputs actions in the normalized range [-1, 1].
        This method clips to [-1, 1] for safety and scales to physical units:
        - Single integrator: velocities in [-v_max, v_max] and [-omega_max, omega_max]
        - Double integrator: accelerations in [-acc_v_max, acc_v_max] and [-acc_omega_max, acc_omega_max]

        Parameters
        ----------
        actions : dict
            Dictionary mapping agent names to normalized action arrays in [-1, 1]

        Returns
        -------
        np.ndarray
            Scaled actions as array of shape (num_agents, 2) in physical units
        """
        if set(actions.keys()) != set(self.agents):
            raise RuntimeError("Actions must be provided for all agents")

        # Convert actions to array in the same order as self.agents for handling
        clean_actions = np.array([actions[agent] for agent in self.agents], dtype=np.float32)

        # Clip normalized actions to [-1, 1] and scale to physical units based on kinematics mode
        if self.kinematics == "single":
            # Single integrator: actions are normalized [-1, 1] scaled to velocities
            clean_actions[:, 0] = np.clip(clean_actions[:, 0], -1.0, 1.0) * self.v_max
            clean_actions[:, 1] = np.clip(clean_actions[:, 1], -1.0, 1.0) * self.omega_max
        else:  # double integrator
            # Double integrator: actions are normalized [-1, 1] scaled to accelerations
            clean_actions[:, 0] = np.clip(clean_actions[:, 0], -1.0, 1.0) * self.acc_v_max
            clean_actions[:, 1] = np.clip(clean_actions[:, 1], -1.0, 1.0) * self.acc_omega_max
        return clean_actions

    def move(self, actions: dict) -> None:
        """Apply the actions of all agents to the agents and update their positions.

        Parameters
        ----------
        actions : dict
            Dictionary mapping agent names to 2D action vectors [linear_action, angular_action].
        """
        actions_array = self._clean_actions(actions)
        lin_acs = actions_array[:, 0]
        ang_acs = actions_array[:, 1]

        # Update velocities based on kinematics
        if self.kinematics == "single":
            # Single integrator: actions directly set velocities
            self.linear_vels = lin_acs
            self.angular_vels = ang_acs
        else:
            # Double integrator: actions are accelerations
            self.linear_vels = np.clip(self.linear_vels + lin_acs * self.dt, -self.v_max, self.v_max)
            self.angular_vels = np.clip(self.angular_vels + ang_acs * self.dt, -self.omega_max, self.omega_max)

        # Update orientations with time step
        self.orientations = self.orientations + self.angular_vels * self.dt
        self.orientations = (self.orientations + math.pi) % (2 * math.pi) - math.pi

        # Compute cartesian displacements with time step
        dx = (self.linear_vels * np.cos(self.orientations) * self.dt).astype(np.float32)
        dy = (self.linear_vels * np.sin(self.orientations) * self.dt).astype(np.float32)

        # Update positions
        self.positions[:, 0] = self.positions[:, 0] + dx
        self.positions[:, 1] = self.positions[:, 1] + dy

    def initialize_random_positions(self, world_size: float, rng: Optional[np.random.Generator] = None) -> None:
        """Initializes the positions of the agents uniformly inside of the worlds boundaries.

        Parameters
        ----------
        world_size : float
            Size of the world (agents will be initialized in [0, world_size]^2)
        rng : np.random.Generator, optional
            Random number generator for reproducible initialization. If None, uses global RNG.
        """
        if rng is None:
            rng = np.random.default_rng()

        self.positions = rng.uniform(0.0, world_size, (self.num_agents, 2)).astype(np.float32)
        self.linear_vels = np.zeros(self.num_agents, dtype=np.float32)
        self.angular_vels = np.zeros(self.num_agents, dtype=np.float32)
        rand_angles = rng.uniform(-math.pi, math.pi, self.num_agents)
        self.orientations = rand_angles.astype(np.float32)
