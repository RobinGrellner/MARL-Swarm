from __future__ import annotations
from abc import ABC, abstractmethod
from .agent_handler import AgentHandler
from typing import Dict, Tuple, Optional
from pettingzoo import ParallelEnv
import numpy as np
from gymnasium import spaces
import functools


# TODO: Implement debug-logging
class BaseEnv(ABC, ParallelEnv):
    """Base Environment for Point-Agent Environments.
    Implements single-Integrator and double-Integrator physics for multi agent learning
    using pettingzoo. Also supports toral Worlds.
    Functionalities to implement yourself:
        - Rendering (render)
        - Environment logic (step, reset)
        - Reward Function
        - Observations
        - Logging (optional)

    Parameters
    ----------
    num_agents : int
        Number of agents in the swarm.
    world_size : float
        Side length of the square world. When torus is true, the world wraps around.
    torus : bool, optional
        If true, the world is treated as a torus. Agents that exit the world on one side appear
        on the other side.
    kinematics: str
        Kinematics of the agents. "single" for single integrator, "double" for double integrator.
    v_max: float
        Maximum linear velocity of the agents.
    omega_max: float
        Maximum angular velocity of the agents.
    acc_v_max: float
        Maximum allowed acceleration for linear velocity in double integrator kinematics.
    acc_omega_max: float
        Maximum allowed acceleration for anguar velocity in double integrator kinematics.
    max_steps: int
        Maximum allowed steps. After this many steps, the Environment automatically terminates and the episode ends.
    """

    def __init__(
        self,
        *,
        num_agents: int,
        world_size: float,
        torus: bool = False,
        kinematics: str = "single",
        v_max: float = 1.0,
        omega_max: float = 1.0,
        acc_v_max: float = 1.0,
        acc_omega_max: float = 1.0,
        max_steps: int = 1000,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        # Agent Handler setup
        self.agent_handler = AgentHandler(
            num_agents=num_agents,
            kinematics=kinematics,
            v_max=v_max,
            omega_max=omega_max,
            acc_v_max=acc_v_max,
            acc_omega_max=acc_omega_max,
        )

        # Setting up Necessary Environment variables
        self.agent_names = self.agent_handler.agents
        self.agents = self.agent_names
        self.possible_agents = self.agent_names
        self.max_steps = max_steps
        self.torus = torus
        self.world_size = world_size
        self.render_mode = render_mode
        self._setup_spaces()

    @property
    def observation_spaces(self) -> Dict[str, spaces.Box]:
        """Dictionary mapping agent identifiers to their observation spaces."""
        return self._observation_space

    @property
    def action_spaces(self) -> Dict[str, spaces.Box]:
        """Dictionary mapping agent identifiers to their action spaces."""
        return self._action_space

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str):
        """Return the observation space for a single agent."""
        return self._observation_space[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str):
        """Return the action space for a single agent."""
        return self._action_space[agent]

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Dict[str, np.ndarray]:
        """Template Method to reset the environment state.
        - Sample positions for agents
        - Velocities are set to zero
        - Step counter is cleared
        """
        # Common resets
        if seed is not None:
            np.random.seed(seed)
        self.step_count = 0

        # Specific resets
        self._reset_agents()
        # All Agents are active at the Beginning of the episode
        self.agents = list(self.agent_names)
        return self._get_observations(), self._get_infos()

    def step(
        self, actions: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, dict]]:
        """Template method for advancing the simulation by one timestep.

        Parameters
        ----------
        actions : dict
            Mapping from agent to an 2D-action-vector.
        """
        self._update_agents(actions)
        self._intermediate_steps()

        observations = self._get_observations()
        rewards = self._calculate_rewards(actions)
        terminations = self._check_terminations()
        truncations = self._check_truncations()
        infos = self._get_infos()

        # Update step count
        self.step_count += 1
        if self.step_count >= self.max_steps:
            for agent in self.agents:
                truncations[agent] = True
        self.render()
        return observations, rewards, terminations, truncations, infos

    def render(self):
        """Template method for rendering."""
        self._render()

    def close(self):
        """Template method for handling the shutdown of the environment."""
        self._close()

    def _update_agents(self, actions):
        """Move the agents in the world according to the passed actions.

        Parameters
        ----------
        acttions : dict
            Mapping from agent to an 2D-action-vector.
        """
        self.agent_handler.move(actions)

        if self.torus:
            self.agent_handler.positions = self.agent_handler.positions % self.world_size
        else:
            self.agent_handler.positions = np.clip(self.agent_handler.positions, 0.0, self.world_size)

    def _setup_spaces(self):
        """Setup of shared action and observation spaces"""
        if self.agent_handler.kinematics == "single":
            low = np.array(
                [-self.agent_handler.v_max, -self.agent_handler.omega_max],
                dtype=np.float32,
            )
            high = np.array(
                [self.agent_handler.v_max, self.agent_handler.omega_max],
                dtype=np.float32,
            )
        else:
            low = np.array(
                [-self.agent_handler.acc_v_max, -self.agent_handler.acc_omega_max],
                dtype=np.float32,
            )
            high = np.array(
                [self.agent_handler.acc_v_max, self.agent_handler.acc_omega_max],
                dtype=np.float32,
            )

        self._action_space: Dict[str, spaces.Box] = {
            agent: spaces.Box(low=low, high=high, shape=(2,), dtype=np.float32) for agent in self.agent_handler.agents
        }

        self._observation_space = self._get_observation_space()

    # Abstract methods for setup
    @abstractmethod
    def _get_observation_space(self) -> Dict[str, spaces.Box]:
        """Logic on how agents observe their surroundings."""
        raise NotImplementedError

    # Abstract methods for reset
    @abstractmethod
    def _reset_agents(self) -> None:
        """Logic for resetting the agents."""
        raise NotImplementedError

    # Abstract methods for steps
    @abstractmethod
    def _calculate_rewards(self, actions) -> dict:
        """Logic for calculating the rewards."""
        raise NotImplementedError

    @abstractmethod
    def _get_observations(self) -> dict:
        """Logic for retrieving observations."""
        raise NotImplementedError

    @abstractmethod
    def _check_terminations(self) -> dict:
        """Checking for terminations/end conditions."""
        raise NotImplementedError

    @abstractmethod
    def _check_truncations(self) -> dict:
        """Checking for truncations."""
        raise NotImplementedError

    @abstractmethod
    def _get_infos(self) -> dict:
        """Getting infos"""
        raise NotImplementedError

    @abstractmethod
    def _intermediate_steps(self):
        """Handle any additional steps that need to be done. If no steps need to be done, just pass."""
        raise NotImplementedError

    @abstractmethod
    def _render(self):
        """Render the Environment."""
        raise NotImplementedError

    @abstractmethod
    def _close(self):
        """Close the Envorinment"""
        raise NotImplementedError
