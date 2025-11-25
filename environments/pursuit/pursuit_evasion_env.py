"""
Pursuit-Evasion Environment for Multi-Agent Reinforcement Learning

This environment implements a pursuit-evasion scenario where multiple pursuers
attempt to capture a single evader. The pursuers are trainable agents, while
the evader follows a scripted policy (can be extended to be trainable).
"""

from __future__ import annotations

import functools
import numpy as np
from gymnasium import spaces
from typing import Dict, Tuple, Optional

from environments.base.base_environment import BaseEnv


class PursuitEvasionEnv(BaseEnv):
    """
    Pursuit-Evasion environment where multiple pursuers try to capture a single evader.

    The pursuers are the trainable agents, and the evader is part of the environment
    (currently using a scripted policy, but can be extended to be trainable).

    Parameters:
    -----------
    num_pursuers : int
        Number of pursuer agents
    world_size : float
        Side length of the square world
    max_steps : int
        Maximum number of steps per episode
    capture_radius : float
        Distance threshold for capture
    evader_speed : float
        Maximum speed of the evader
    obs_model : str
        Observation model to use ('global_basic', 'local_basic', etc.)
    comm_radius : Optional[float]
        Communication radius for local observations
    max_pursuers : Optional[int]
        Maximum number of pursuers for scale-invariant observations
    kinematics : str
        Kinematic model ('single' or 'double' integrator)
    render_mode : Optional[str]
        Rendering mode ('human', 'rgb_array', or None)
    fps : int
        Frames per second for rendering
    """

    metadata = {"name": "pursuit_evasion", "render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_pursuers: int = 10,
        world_size: float = 10.0,
        max_steps: int = 100,
        capture_radius: float = 0.5,
        evader_speed: float = 1.0,
        obs_model: str = "global_basic",
        comm_radius: Optional[float] = None,
        max_pursuers: Optional[int] = None,
        kinematics: str = "single",
        render_mode: Optional[str] = None,
        fps: int = 20,
    ):
        # Validate parameters
        if num_pursuers <= 0:
            raise ValueError("num_pursuers must be > 0")
        if world_size <= 0:
            raise ValueError("world_size must be > 0")
        if capture_radius <= 0:
            raise ValueError("capture_radius must be > 0")

        self.num_pursuers = num_pursuers
        self.capture_radius = capture_radius
        self.evader_speed = evader_speed
        self.obs_model = obs_model
        self.comm_radius = comm_radius

        # Scale-invariance support
        if max_pursuers is not None and max_pursuers < num_pursuers:
            raise ValueError("max_pursuers must be >= num_pursuers")
        self._max_pursuers = max_pursuers if max_pursuers is not None else num_pursuers

        # Evader state (managed separately from pursuers)
        self.evader_pos = np.zeros(2, dtype=np.float32)
        self.evader_vel = 0.0
        self.evader_orientation = 0.0

        # Rendering setup (not passed to BaseEnv)
        self.render_mode = render_mode
        self.fps = fps
        self.window_size = 600
        self.screen = None
        self.clock = None

        # Define observation layout for mean embedding extractor (must be before parent init)
        self._setup_observation_layout()

        # Call parent constructor
        super().__init__(
            num_agents=num_pursuers,
            world_size=world_size,
            max_steps=max_steps,
            kinematics=kinematics,
            torus=False,
            render_mode=render_mode,
        )

    def _setup_observation_layout(self):
        """Setup observation layout for compatibility with MeanEmbeddingExtractor."""
        if self.obs_model == "global_basic":
            self._neighbour_feature_dim = 2  # distance, bearing
        elif self.obs_model == "local_basic":
            self._neighbour_feature_dim = 2  # distance, bearing
        elif self.obs_model == "global_extended":
            self._neighbour_feature_dim = 4  # distance, bearing, rel_orientation, rel_velocity
        elif self.obs_model == "local_extended":
            self._neighbour_feature_dim = 3  # distance, bearing, rel_orientation
        else:
            raise ValueError(f"Unknown obs_model: {self.obs_model}")

        # Maximum neighbors = other pursuers (not including evader, which is separate)
        self._max_neighbours = self._max_pursuers - 1

        # Local features: wall_dist, wall_bearing
        local_dim = 2

        # Evader features: distance, bearing (fixed slot, always present)
        evader_dim = 2

        # Total observation dimension
        total_dim = (
            local_dim
            + self._max_neighbours * self._neighbour_feature_dim
            + evader_dim
            + self._max_neighbours  # mask for pursuers only
        )

        self.obs_layout = {
            "local_dim": local_dim,
            "neigh_dim": self._neighbour_feature_dim,
            "max_neighbours": self._max_neighbours,
            "evader_dim": evader_dim,
            "total_dim": total_dim,
        }

    def _get_observation_space(self):
        """
        Define observation space for pursuers.

        Observation: [local_features] + [pursuer_neighbor_features] + [evader_features] + [mask]
        """
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_layout["total_dim"],), dtype=np.float32)
        return {agent: obs_space for agent in self.agent_handler.agents}

    def _reset_agents(self, seed: Optional[int] = None) -> None:
        """Reset pursuer and evader positions to random locations."""
        if seed is not None:
            np.random.seed(seed)

        # Reset pursuers to random positions (this also initializes velocities and orientations)
        self.agent_handler.initialize_random_positions(self.world_size)

        # Reset evader to random position
        self.evader_pos = np.random.uniform(0, self.world_size, size=2).astype(np.float32)

        # Reset evader velocity and orientation
        self.evader_vel = 0.0
        self.evader_orientation = np.random.uniform(0, 360)

    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Construct observations for all pursuers."""
        observations = {}
        positions = self.agent_handler.positions
        orientations = self.agent_handler.orientations
        linear_vels = self.agent_handler.linear_vels
        num_agents = len(positions)

        # Precompute pairwise displacements and distances between pursuers
        pos_i = positions[:, np.newaxis, :]  # (N, 1, 2)
        pos_j = positions[np.newaxis, :, :]  # (1, N, 2)
        diff = pos_j - pos_i  # (N, N, 2)
        distances = np.linalg.norm(diff, axis=2)  # (N, N)

        # Compute bearings between pursuers
        bearings_raw = np.arctan2(diff[:, :, 1], diff[:, :, 0])  # (N, N)
        orientations_expanded = orientations[:, np.newaxis]  # (N, 1)
        bearings = bearings_raw - orientations_expanded  # (N, N) in radians
        bearings = (bearings + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-π, π]

        # Sort neighbors by distance
        sorted_indices = np.argsort(distances, axis=1)  # (N, N)

        # Select top max_neighbours (excluding self at index 0)
        actual_neighbors = min(num_agents - 1, self._max_neighbours)
        neighbor_indices_available = sorted_indices[:, 1 : actual_neighbors + 1]  # (N, actual_neighbors)

        # Pad to max_neighbours
        padding = np.zeros((num_agents, self._max_neighbours - actual_neighbors), dtype=int)
        neighbor_indices = np.concatenate([neighbor_indices_available, padding], axis=1)

        # Gather pursuer neighbor features
        row_indices = np.arange(num_agents)[:, np.newaxis]  # (N, 1)
        neighbor_dists = distances[row_indices, neighbor_indices]  # (N, max_neighbours)
        neighbor_bears = bearings[row_indices, neighbor_indices]  # (N, max_neighbours)

        # Build pursuer neighbor features based on observation model
        if self.obs_model in {"global_basic", "local_basic"}:
            neighbor_features = np.stack([neighbor_dists, neighbor_bears], axis=2)  # (N, max_neighbours, 2)

        elif self.obs_model == "global_extended":
            # Relative orientations
            orientations_matrix = orientations[np.newaxis, :] - orientations[:, np.newaxis]
            orientations_matrix = (orientations_matrix + np.pi) % (2 * np.pi) - np.pi
            neighbor_oris = orientations_matrix[row_indices, neighbor_indices]

            # Relative velocities (scalar magnitude)
            neighbor_vels = linear_vels[neighbor_indices] - linear_vels[:, np.newaxis]

            neighbor_features = np.stack(
                [neighbor_dists, neighbor_bears, neighbor_oris, neighbor_vels], axis=2
            )  # (N, max_neighbours, 4)

        elif self.obs_model == "local_extended":
            # Relative orientations only
            orientations_matrix = orientations[np.newaxis, :] - orientations[:, np.newaxis]
            orientations_matrix = (orientations_matrix + np.pi) % (2 * np.pi) - np.pi
            neighbor_oris = orientations_matrix[row_indices, neighbor_indices]

            neighbor_features = np.stack(
                [neighbor_dists, neighbor_bears, neighbor_oris], axis=2
            )  # (N, max_neighbours, 3)

        # Apply communication radius mask for local models
        if self.obs_model.startswith("local"):
            valid_mask = neighbor_dists <= self.comm_radius
            neighbor_features = np.where(valid_mask[:, :, np.newaxis], neighbor_features, 0.0)
        else:
            # Global models: all neighbors valid up to actual count
            valid_mask = np.zeros((num_agents, self._max_neighbours), dtype=bool)
            valid_mask[:, :actual_neighbors] = True

        # Compute wall distances and bearings
        wall_dists = np.zeros(num_agents, dtype=np.float32)
        wall_bearings = np.zeros(num_agents, dtype=np.float32)

        for i in range(num_agents):
            wall_dists[i] = self._calc_dist_to_closest_wall(positions[i])
            wall_bearings[i] = self._bearing_to_closest_wall(positions[i], orientations[i])

        # Compute evader features for each pursuer
        evader_diff = self.evader_pos - positions  # (N, 2)
        evader_dists = np.linalg.norm(evader_diff, axis=1)  # (N,)
        evader_bearings_raw = np.arctan2(evader_diff[:, 1], evader_diff[:, 0])  # (N,)
        evader_bearings = evader_bearings_raw - orientations  # (N,)
        evader_bearings = (evader_bearings + np.pi) % (2 * np.pi) - np.pi

        # Build final observations
        local_features = np.stack([wall_dists, wall_bearings], axis=1)  # (N, 2)
        neighbor_flat = neighbor_features.reshape(num_agents, self._max_neighbours * self._neighbour_feature_dim)
        evader_features = np.stack([evader_dists, evader_bearings], axis=1)  # (N, 2)
        mask = valid_mask.astype(np.float32)  # (N, max_neighbours)

        # Concatenate all components
        all_obs = np.concatenate([local_features, neighbor_flat, evader_features, mask], axis=1)

        # Convert to dictionary
        for idx, agent in enumerate(self.agents):
            observations[agent] = all_obs[idx]

        return observations

    def _calculate_rewards(self, actions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate rewards for all pursuers.

        Reward structure:
        - Small negative reward each timestep (time penalty)
        - Large positive reward for capture
        - Distance-based reward (negative of min distance to evader)
        """
        rewards = {}

        # Calculate minimum distance to evader
        distances = np.linalg.norm(self.agent_handler.positions - self.evader_pos, axis=1)
        min_dist = np.min(distances)

        # Check if captured
        captured = min_dist < self.capture_radius

        for i, agent in enumerate(self.agents):
            # Time penalty
            reward = -0.01

            # Distance-based reward (encourage getting closer)
            reward += -0.1 * distances[i] / self.world_size

            # Capture bonus
            if captured:
                reward += 10.0

            rewards[agent] = reward

        return rewards

    def _check_terminations(self) -> Dict[str, bool]:
        """Check if episode should terminate (evader captured)."""
        distances = np.linalg.norm(self.agent_handler.positions - self.evader_pos, axis=1)
        captured = np.any(distances < self.capture_radius)

        return {agent: captured for agent in self.agents}

    def _check_truncations(self) -> Dict[str, bool]:
        """Check if episode should truncate (max steps reached)."""
        truncated = self.step_count >= self.max_steps
        return {agent: truncated for agent in self.agents}

    def _get_infos(self) -> Dict[str, dict]:
        """Return additional information for each agent."""
        # Calculate distances to evader
        distances = np.linalg.norm(self.agent_handler.positions - self.evader_pos, axis=1)
        min_dist = np.min(distances)

        infos = {}
        for i, agent in enumerate(self.agents):
            infos[agent] = {
                "distance_to_evader": distances[i],
                "min_distance_to_evader": min_dist,
                "evader_captured": min_dist < self.capture_radius,
            }

        return infos

    def _intermediate_steps(self) -> None:
        """Perform intermediate steps including evader movement."""
        # Move evader (simple scripted policy for now)
        self._move_evader()

        # Clip evader position to world bounds
        self.evader_pos = np.clip(self.evader_pos, 0, self.world_size)

    def _move_evader(self) -> None:
        """
        Move evader using a simple scripted policy.

        Current policy: Move away from nearest pursuer.
        """
        # Find nearest pursuer
        distances = np.linalg.norm(self.agent_handler.positions - self.evader_pos, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_pos = self.agent_handler.positions[nearest_idx]

        # Calculate direction away from nearest pursuer
        direction = self.evader_pos - nearest_pos
        if np.linalg.norm(direction) > 0:
            direction = direction / np.linalg.norm(direction)

        # Move evader
        self.evader_pos += direction * self.evader_speed

        # Update evader orientation
        if np.linalg.norm(direction) > 0:
            self.evader_orientation = np.rad2deg(np.arctan2(direction[1], direction[0]))

    # Helper methods

    def _calc_dist_to_closest_wall(self, pos: np.ndarray) -> float:
        """Calculate distance to closest wall."""
        return np.min([pos[0], pos[1], self.world_size - pos[0], self.world_size - pos[1]])

    def _bearing_to_closest_wall(self, pos: np.ndarray, orientation: float) -> float:
        """Calculate bearing to closest wall using proper vector math.

        Returns bearing in radians, normalized to [-π, π].
        """
        x, y = pos

        # Calculate distances to each wall
        dist_left = x
        dist_right = self.world_size - x
        dist_bottom = y
        dist_top = self.world_size - y

        # Find closest wall and determine target point on that wall
        min_dist = min(dist_left, dist_right, dist_bottom, dist_top)

        if min_dist == dist_left:
            # Left wall: target is at (0, y)
            wall_target = np.array([0.0, y])
        elif min_dist == dist_right:
            # Right wall: target is at (world_size, y)
            wall_target = np.array([self.world_size, y])
        elif min_dist == dist_bottom:
            # Bottom wall: target is at (x, 0)
            wall_target = np.array([x, 0.0])
        else:
            # Top wall: target is at (x, world_size)
            wall_target = np.array([x, self.world_size])

        # Calculate vector from agent to wall target
        delta = wall_target - pos

        # Calculate absolute bearing to wall
        wall_bearing = np.arctan2(delta[1], delta[0])

        # Calculate relative bearing (wall bearing - agent orientation)
        relative_bearing = wall_bearing - orientation

        # Normalize to [-π, π]
        relative_bearing = (relative_bearing + np.pi) % (2 * np.pi) - np.pi

        return relative_bearing

    def _render(self) -> Optional[np.ndarray]:
        """
        Render the environment using pygame.

        Returns:
            Optional[np.ndarray]: RGB array if render_mode is 'rgb_array', None otherwise
        """
        import pygame

        if self.screen is None and self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Pursuit-Evasion Environment")
            self.clock = pygame.time.Clock()

        if self.render_mode == "human":
            # Clear screen
            self.screen.fill((255, 255, 255))

            # Calculate scaling factor
            scale = self.window_size / self.world_size

            # Draw pursuers (red circles)
            for i in range(self.num_agents):
                pos = self.agent_handler.positions[i] * scale
                pygame.draw.circle(self.screen, (255, 0, 0), pos.astype(int), 5)

                # Draw orientation line
                orient_rad = np.deg2rad(self.agent_handler.orientations[i])
                end_pos = pos + 10 * np.array([np.cos(orient_rad), np.sin(orient_rad)])
                pygame.draw.line(self.screen, (255, 0, 0), pos.astype(int), end_pos.astype(int), 2)

            # Draw evader (blue circle)
            evader_pos_scaled = self.evader_pos * scale
            pygame.draw.circle(self.screen, (0, 0, 255), evader_pos_scaled.astype(int), 5)

            # Draw evader orientation line
            orient_rad = np.deg2rad(self.evader_orientation)
            end_pos = evader_pos_scaled + 10 * np.array([np.cos(orient_rad), np.sin(orient_rad)])
            pygame.draw.line(self.screen, (0, 0, 255), evader_pos_scaled.astype(int), end_pos.astype(int), 2)

            # Update display
            pygame.display.flip()
            self.clock.tick(self.fps)

            return None

        elif self.render_mode == "rgb_array":
            # Return RGB array for video recording
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2))

    def _close(self) -> None:
        """Clean up resources."""
        if self.screen is not None:
            import pygame

            pygame.quit()
            self.screen = None
