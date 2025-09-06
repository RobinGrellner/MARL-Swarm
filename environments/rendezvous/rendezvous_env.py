from environments.base.base_environment import BaseEnv
from environments.base.agent_handler import AgentHandler
from typing import Dict, List, Tuple, Optional
import numpy as np
import math
import pygame
from gymnasium import spaces

class RendezvousEnv(BaseEnv):
    def __init__(
        self,
        *,
        num_agents: int,
        world_size: float,
        max_steps: int = 1000,
        csv_log_path: Optional[str] = None,
        torus: bool = False,
        break_distance_threshold: Optional[float] = None,
        kinematics: str = "single",
        v_max: float = 1.0,
        omega_max: float = 1.0,
        acc_v_max: float = 1.0,
        acc_omega_max: float = 1.0,
        obs_model: str = "classic",
        comm_radius: Optional[float] = None,
        render_mode: str = "",
        fps: int = 60,
    ):
        # Store rendezvous-specific parameters
        self.csv_log_path = csv_log_path
        self.break_distance_threshhold=break_distance_threshold,
        self.obs_model = obs_model.lower() if obs_model is not None else "global_basic"
        self.comm_radius = comm_radius
        self.render_mode = render_mode
        self.fps = fps

        # Initialize base environment
        super().__init__(
            num_agents=num_agents,
            world_size=world_size,
            torus=torus,
            kinematics=kinematics,
            v_max=v_max,
            omega_max=omega_max,
            acc_v_max=acc_v_max,
            acc_omega_max=acc_omega_max,
            max_steps=max_steps,
        )

    # Methods for setup
    def _get_observation_space(self):
        """Logic on how agents observe their surroundings.
            global_basic:
                - local obs:
                    - distance to closest wall
                    - bearing to closest wall
                - obs to other agents:
                    - distance
                    - bearing
            global_extended:
                - local obs:
                    - distance to closest wall
                    - bearing to closest wall
                - obs to other agents:
                    - distance
                    - bearing
                    - relative orientation
                    - relative velocity
            local_basic:
                - local obs:
                    - distance to closest wall
                    - bearing to closest wall
                - obs to other agents:
                    - distance
                    - bearing
            local_extended:
                - local obs:
                    - distance to closest wall
                    - bearing to closest wall
                - obs to other agents:
                    - distance
                    - bearing
                    - relative orientation
            local_comm:
                - local obs:
                    - distance to closest wall
                    - bearing to closest wall
                - obs to other agents:
                    - distance
                    - bearing
                    - relative orientation
                    - Size of neighbourhood
        """
        self._observation_space = {}
        if self.obs_model == "classic":
            # Classic 7‑D observation:
            #   own position (x, y),
            #   mean position (x, y),
            #   own velocity(linear, angular),
            #   orientation
            obs_low = np.array(
                [
                    -self.world_size,
                    -self.world_size,
                    -self.world_size,
                    -self.world_size,
                    -self.agent_handler.v_max,
                    -self.agent_handler.omega_max,
                    -math.pi,
                ],
                dtype=np.float32,
            )
            obs_high = np.array(
                [
                    self.world_size,
                    self.world_size,
                    self.world_size,
                    self.world_size,
                    self.agent_handler.v_max,
                    self.agent_handler.omega_max,
                    math.pi,
                ],
                dtype=np.float32,
            )
            if self.agent_handler.kinematics == "single":
                # Remove omega from observation
                obs_low = np.delete(obs_low, 5)
                obs_high = np.delete(obs_high, 5)
            for agent in self.agents:
                self._observation_spaces[agent] = spaces.Box(
                    low=obs_low, high=obs_high, shape=(len(obs_low),), dtype=np.float32
                )
            # store dims for classic path
            self._local_feature_dim = len(obs_low)
            self._neighbour_feature_dim = 0
            self._max_neighbours = 0
            # no explicit mask for classic
            self.obs_layout = {
                "local_start": 0,
                "local_dim": 7,
                "neigh_start": 0,
                "neigh_dim": 0,
                "max_neighbours": 0,
                "mask_start": 0,
            }
            self.obs_total_dim = 7
        else:
            # Determine local and neighbour feature dimensions according to Appendix B
            if self.obs_model in {"global_basic", "local_basic"}:
                neighbour_feature_dim = 2  # distance and bearing
                local_feature_dim = 2  # distance to wall and bearing to wall
            elif self.obs_model == "global_extended":
                neighbour_feature_dim = (
                    5  # distance, bearing, relative orientation, rel vel x, rel vel y
                )
                local_feature_dim = 2
            elif self.obs_model == "local_extended":
                neighbour_feature_dim = 3  # distance, bearing, relative orientation
                local_feature_dim = 2
            elif self.obs_model == "local_comm":
                neighbour_feature_dim = (
                    4  # distance, bearing, relative orientation, neighbour size
                )
                local_feature_dim = (
                    3  # distance to wall, bearing to wall, own neighbourhood size
                )
            else:
                raise ValueError(f"Unknown observation model: {self.obs_model}")

            self._neighbour_feature_dim = neighbour_feature_dim
            self._local_feature_dim = local_feature_dim
            self._max_neighbours = self._num_agents - 1
            # Define layout of observation vector: [local | neighbour_pad | mask]
            self.obs_layout = {
                "local_start": 0,
                "local_dim": local_feature_dim,
                "neigh_start": None,
                "neigh_dim": neighbour_feature_dim,
                "max_neighbours": self._max_neighbours,
                "mask_start": None,
            }
            # Compute starting indices
            self.obs_layout["neigh_start"] = (
                self.obs_layout["local_start"] + local_feature_dim
            )
            self.obs_layout["mask_start"] = (
                self.obs_layout["neigh_start"]
                + self._max_neighbours * neighbour_feature_dim
            )
            self.obs_total_dim = self.obs_layout["mask_start"] + self._max_neighbours
            # Build observation space: local and neighbour features unbounded; mask bounded between 0 and 1
            low = -np.inf * np.ones(self.obs_total_dim, dtype=np.float32)
            high = np.inf * np.ones(self.obs_total_dim, dtype=np.float32)
            # mask slice
            m_start = self.obs_layout["mask_start"]
            m_end = m_start + self._max_neighbours
            low[m_start:m_end] = 0.0
            high[m_start:m_end] = 1.0
            for agent in self.agents:
                self._observation_spaces[agent] = spaces.Box(
                    low=low, high=high, shape=(self.obs_total_dim,), dtype=np.float32
                )
        pass

    def _local_feature_dim(self) -> int:
        """Dimension of the local feature block for each agent.
            - Distance to nearest wall
            - Bearing to nearest wall
            - Neighbourhood size (Only for local comms)
        """
        if self.obs_model == "local_comm":
            return 3
        return 2

    def _neigh_feature_dim(self) -> int:
        """Dimension of the per-neighbour feature vector."""
        if self.obs_model in {"global_basic", "local_basic"}:
            return 2
        if self.obs_model == "global_extended":
            return 5
        if self.obs_model == "local_extended":
            return 3
        if self.obs_model == "local_comm":
            return 4
        raise ValueError(f"Unknown observation model: {self._obs_model}")

    def _local_features(self, agent_idx: int) -> np.ndarray:
        """Compute the local feature vector for a given agent."""

        pos_i = self.agent_handler.positions[agent_idx]
        theta_i = float(self.agent_handler.orientations[agent_idx])
        # Distance and bearing to the nearest wall
        if self.torus:
            # Torus has no walls. So default to world_size for distance, and default to zero bearing.
            d_wall = self.world_size
            phi_wall = 0.0
        else:
            # Distances to the four walls in [0, world_size]
            dx_left = pos_i[0]
            dx_right = self.world_size - pos_i[0] 
            dy_bottom = pos_i[1]
            dy_top = self.world_size - pos_i[1] 
            dists = [dx_left, dx_right, dy_bottom, dy_top]
            d_wall = float(min(dists))
            # Determine which wall is closest
            min_idx = int(np.argmin(dists))
            if min_idx == 0:
                target = np.array([0.0, pos_i[1]], dtype=np.float32)
            elif min_idx == 1:
                target = np.array([self.world_size, pos_i[1]], dtype=np.float32)
            elif min_idx == 2:
                target = np.array([pos_i[0], 0.0], dtype=np.float32)
            else:
                target = np.array([pos_i[0], self.world_size], dtype=np.float32)
            delta_w = target - pos_i
            phi_wall = float(math.atan2(delta_w[1], delta_w[0]) - theta_i)
            phi_wall = (phi_wall + math.pi) % (2 * math.pi) - math.pi
        feats: List[float] = [d_wall, phi_wall]

        if self._obs_model == "local_comm":
            count = 0
            for j in range(self._num_agents):
                if j == agent_idx:
                    continue
                # minimal image distance
                dvec = self.delta(pos_i, self.positions[j])
                if np.linalg.norm(dvec) <= self.comm_radius:
                    count += 1
            feats.append(float(count))
        return np.array(feats, dtype=np.float32)

    def _neighbour_feature(self, agent_idx_i: int, agent_idx_j: int) -> np.ndarray:
        """Compute the per-neighbour feature vector from agent.
            Contains:
                - Distance from i to j
                - Bearing from i to j
                - Relative orientation (extended model)
                - Relative velocity (global extended model)
                - neighbourhood size (local_comm)
        """
        pos_i = self.agent_handler.positions[agent_idx_i]
        pos_j = self.agent_handler.positions[agent_idx_j]
        theta_i = float(self.agent_handler.orientations[agent_idx_i])
        theta_j = float(self.agent_handler.orientations[agent_idx_j])
        # Displacement from i to j using minimal image convention
        delta = self.delta(pos_i, pos_j)
        d_ij = float(np.linalg.norm(delta))
        bearing = float(math.atan2(delta[1], delta[0]) - theta_i)
        # Wrap bearing
        bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
        features: List[float] = [d_ij, bearing]
        # Relative orientation for extended and comm models
        if self._obs_model in {"global_extended", "local_extended", "local_comm"}:
            psi = float(theta_j - theta_i)
            psi = (psi + math.pi) % (2 * math.pi) - math.pi
            features.append(psi)
        # Relative velocity for global_extended
        if self._obs_model == "global_extended":
            # Compute velocity vectors v_i and v_j
            v_i = np.array([
                self.velocities[agent_idx_i] * math.cos(theta_i),
                self.velocities[agent_idx_i] * math.sin(theta_i),
            ], dtype=np.float32)
            v_j = np.array([
                self.velocities[agent_idx_j] * math.cos(theta_j),
                self.velocities[agent_idx_j] * math.sin(theta_j),
            ], dtype=np.float32)
            rel_v = v_i - v_j
            features.extend([float(rel_v[0]), float(rel_v[1])])
        # Neighbour size for local_comm
        if self._obs_model == "local_comm":
            # Count neighbours of j
            count = 0
            pos_j_local = self.positions[agent_idx_j]
            for k in range(self._num_agents):
                if k == agent_idx_j:
                    continue
                dvec = self.delta(pos_j_local, self.positions[k])
                if np.linalg.norm(dvec) <= self.comm_radius:
                    count += 1
            features.append(float(count))
        return np.array(features, dtype=np.float32)


    # Abstract methods for reset
    @abstractmethod
    def _reset_agents(self) -> None:
        """Logic for resetting the agents."""
        pass

    # Abstract methods for steps
    @abstractmethod
    def _calculate_rewards(self) -> dict:
        """Logic for calculating the rewards."""
        pass

    @abstractmethod
    def _get_observations(self) -> dict:
        """Logic for retrieving observations."""
        pass

    @abstractmethod
    def _check_terminations(self) -> dict:
        """Checking for terminations/end conditions."""
        pass

    @abstractmethod
    def _check_truncations(self) -> dict:
        """Checking for truncations."""
        pass

    @abstractmethod
    def _get_infos(self) -> dict:
        """Getting infos"""
        pass

    @abstractmethod
    def _intermediate_steps(self):
        """Handle any additional steps that need to be done. If no steps need to be done, just pass."""
        pass

    @abstractmethod
    def _render(self):
        """Render the Environment."""
        pass

    @abstractmethod
    def _close(self):
        """Close the Envorinment"""
        pass
