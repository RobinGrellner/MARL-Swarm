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
from environments.pursuit.evasion_agent import create_evasion_agent


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
    v_max : float
        Maximum linear velocity of the pursuers
    obs_model : str
        Observation model to use ('global_basic', 'local_basic', etc.)
    comm_radius : Optional[float]
        Communication radius for local observations
    max_pursuers : Optional[int]
        Maximum number of pursuers for scale-invariant observations
    kinematics : str
        Kinematic model ('single' or 'double' integrator)
    omega_max : float
        Maximum angular velocity of the pursuers
    evader_strategy : str
        Strategy for evader behavior
    render_mode : Optional[str]
        Rendering mode ('human', 'rgb_array', or None)
    fps : int
        Frames per second for rendering
    torus : bool
        Whether the world has toroidal topology (wraparound)
    """

    metadata = {"name": "pursuit_evasion", "render_modes": ["human", "rgb_array"]}

    def __init__(
        self,
        num_pursuers: int = 10,
        world_size: float = 10.0,
        max_steps: int = 100,
        capture_radius: float = 0.1,
        evader_speed: float = 1.0,
        v_max: float = 1.0,
        obs_model: str = "global_basic",
        comm_radius: Optional[float] = None,
        max_pursuers: Optional[int] = None,
        kinematics: str = "single",
        omega_max: float = 1.0,
        evader_strategy: str = "voronoi_center",
        render_mode: Optional[str] = None,
        fps: int = 20,
        torus: bool = False,
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
        self.evader_strategy = evader_strategy
        self.torus = torus

        # Set comm_radius based on observation model (Hüttenrauch et al. 2019)
        # Global models: ALWAYS use world_size to see all agents regardless of distance
        # Local models: Use provided comm_radius, or default to 8.0 for sparse communication
        if obs_model.startswith("global"):
            self.comm_radius = world_size
        elif comm_radius is None:
            self.comm_radius = 8.0  # Default sparse communication radius for local models
        else:
            self.comm_radius = comm_radius

        # Scale-invariance support
        if max_pursuers is not None and max_pursuers < num_pursuers:
            raise ValueError("max_pursuers must be >= num_pursuers")
        self._max_pursuers = max_pursuers if max_pursuers is not None else num_pursuers

        # Observation radius for reward and observation normalization (matching Hüttenrauch)
        # Always use comm_radius / 2 (Hüttenrauch et al. 2019)
        self.obs_radius = self.comm_radius / 2.0

        # Evader state (managed separately from pursuers)
        self.evader_pos = np.zeros(2, dtype=np.float32)
        self.evader_vel = 0.0
        self.evader_orientation = 0.0

        # Cache for evader distances (computed once per step, reused in rewards/terminations/infos)
        self._cached_evader_distances: Optional[np.ndarray] = None

        # Track if evader was captured this episode (for capture_time logging)
        self._was_captured = False

        # Create evasion agent
        self.evasion_agent = create_evasion_agent(
            strategy=evader_strategy, world_size=world_size, max_speed=evader_speed, torus=torus
        )

        # Rendering setup (not passed to BaseEnv)
        self.render_mode = render_mode
        self.fps = fps
        self.window_size = 600
        self.screen = None
        self.clock = None
        self._font = None

        # Define observation layout for mean embedding extractor (must be before parent init)
        self._setup_observation_layout()

        # Call parent constructor
        super().__init__(
            num_agents=num_pursuers,
            world_size=world_size,
            max_steps=max_steps,
            kinematics=kinematics,
            v_max=v_max,
            omega_max=omega_max,
            torus=torus,
            render_mode=render_mode,
        )

    def _setup_observation_layout(self):
        """Setup observation layout for compatibility with MeanEmbeddingExtractor.

        Observation structure (matching Hüttenrauch et al. 2019):
        [local_features | neighbor_features | mask]

        Local features (6 dims): wall_dist, wall_bearing_cos, wall_bearing_sin, evader_dist, evader_bearing_cos, evader_bearing_sin
        The evader is treated as part of the agent's local state (like wall info), not as a neighbor.
        Bearings use (cos, sin) representation for neural network stability.
        This maintains scale-invariance: evader features don't participate in mean aggregation.
        """
        if self.obs_model == "global_basic":
            self._neighbour_feature_dim = 3  # distance, bearing_cos, bearing_sin
        elif self.obs_model == "local_basic":
            self._neighbour_feature_dim = 3  # distance, bearing_cos, bearing_sin
        elif self.obs_model == "global_extended":
            self._neighbour_feature_dim = 6  # distance, bearing_cos, bearing_sin, ori_cos, ori_sin, rel_velocity
        elif self.obs_model == "local_extended":
            self._neighbour_feature_dim = 5  # distance, bearing_cos, bearing_sin, ori_cos, ori_sin
        else:
            raise ValueError(f"Unknown obs_model: {self.obs_model}")

        # Maximum neighbors = other pursuers (not including self)
        self._max_neighbours = self._max_pursuers - 1

        # Local features: wall_dist, wall_bearing_cos, wall_bearing_sin, evader_dist, evader_bearing_cos, evader_bearing_sin (6 dims)
        local_dim = 6

        # Total observation dimension: [local | neighbors*max | mask]
        total_dim = (
            local_dim
            + self._max_neighbours * self._neighbour_feature_dim
            + self._max_neighbours  # mask for pursuers only
        )

        self.obs_layout = {
            "local_dim": local_dim,
            "neigh_dim": self._neighbour_feature_dim,
            "max_neighbours": self._max_neighbours,
            "total_dim": total_dim,
        }

    def _get_observation_space(self):
        """
        Define observation space for pursuers.

        Observation structure: [local_features] + [pursuer_neighbor_features] + [mask]

        Local features (6 dims): [wall_dist, wall_bearing_cos, wall_bearing_sin, evader_dist, evader_bearing_cos, evader_bearing_sin]
        Neighbor features: distance, bearing_cos, bearing_sin, (ori_cos, ori_sin), (relative_velocity) per neighbor
        Mask: binary mask indicating valid neighbors
        """
        # Observation bounds match Huttenrauch: all values in [-1, 1]
        # Distances are [0, 1], bearings/cos/sin are [-1, 1], mask is [0, 1]
        obs_space = spaces.Box(low=-1.0, high=1.0, shape=(self.obs_layout["total_dim"],), dtype=np.float32)
        return {agent: obs_space for agent in self.agent_handler.agents}

    def _reset_agents(self) -> None:
        """Reset pursuer and evader positions to random locations."""
        # Reset pursuers to random positions (this also initializes velocities and orientations)
        self.agent_handler.initialize_random_positions(self.world_size, rng=self._rng)

        # Reset evader to random position
        self.evader_pos = self._rng.uniform(0, self.world_size, size=2).astype(np.float32)

        # Reset evader velocity and orientation (in radians, matching agent_handler convention)
        self.evader_vel = 0.0
        self.evader_orientation = self._rng.uniform(-np.pi, np.pi)

        # Reset capture tracking flag for new episode
        self._was_captured = False

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

        # CRITICAL: Apply torus wrapping to distances when enabled
        if self.torus:
            half = self.world_size / 2.0
            diff = np.where(diff > half, diff - self.world_size, diff)
            diff = np.where(diff < -half, diff + self.world_size, diff)

        distances = np.linalg.norm(diff, axis=2)  # (N, N)

        # Compute bearings between pursuers as (cos, sin) pairs for neural network stability
        bearings_raw = np.arctan2(diff[:, :, 1], diff[:, :, 0])  # (N, N)
        orientations_expanded = orientations[:, np.newaxis]  # (N, 1)
        bearings = bearings_raw - orientations_expanded  # (N, N) in radians
        bearings = (bearings + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-π, π]
        bearings_cos = np.cos(bearings)  # (N, N)
        bearings_sin = np.sin(bearings)  # (N, N)

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
        neighbor_bears_cos = bearings_cos[row_indices, neighbor_indices]  # (N, max_neighbours)
        neighbor_bears_sin = bearings_sin[row_indices, neighbor_indices]  # (N, max_neighbours)

        # Normalize neighbor distances BEFORE constructing features (matching paper specification)
        # Clamp to [0, 1] to match Huttenrauch implementation and ensure bounded observations
        neighbor_dists_normalized = np.minimum(neighbor_dists / self.comm_radius, 1.0) if self.comm_radius is not None else np.minimum(neighbor_dists / self.world_size, 1.0)

        # Build pursuer neighbor features based on observation model
        if self.obs_model in {"global_basic", "local_basic"}:
            neighbor_features = np.stack([neighbor_dists_normalized, neighbor_bears_cos, neighbor_bears_sin], axis=2)  # (N, max_neighbours, 3)

        elif self.obs_model == "global_extended":
            # Relative orientations as (cos, sin) pairs
            orientations_matrix = orientations[np.newaxis, :] - orientations[:, np.newaxis]
            orientations_matrix = (orientations_matrix + np.pi) % (2 * np.pi) - np.pi
            neighbor_oris_cos = np.cos(orientations_matrix)  # (N, N)
            neighbor_oris_sin = np.sin(orientations_matrix)  # (N, N)
            neighbor_oris_cos_gathered = neighbor_oris_cos[row_indices, neighbor_indices]
            neighbor_oris_sin_gathered = neighbor_oris_sin[row_indices, neighbor_indices]

            # Relative velocities (scalar magnitude)
            neighbor_vels = linear_vels[neighbor_indices] - linear_vels[:, np.newaxis]

            neighbor_features = np.stack(
                [neighbor_dists_normalized, neighbor_bears_cos, neighbor_bears_sin, neighbor_oris_cos_gathered, neighbor_oris_sin_gathered, neighbor_vels], axis=2
            )  # (N, max_neighbours, 6)

        elif self.obs_model == "local_extended":
            # Relative orientations as (cos, sin) pairs
            orientations_matrix = orientations[np.newaxis, :] - orientations[:, np.newaxis]
            orientations_matrix = (orientations_matrix + np.pi) % (2 * np.pi) - np.pi
            neighbor_oris_cos = np.cos(orientations_matrix)  # (N, N)
            neighbor_oris_sin = np.sin(orientations_matrix)  # (N, N)
            neighbor_oris_cos_gathered = neighbor_oris_cos[row_indices, neighbor_indices]
            neighbor_oris_sin_gathered = neighbor_oris_sin[row_indices, neighbor_indices]

            neighbor_features = np.stack(
                [neighbor_dists_normalized, neighbor_bears_cos, neighbor_bears_sin, neighbor_oris_cos_gathered, neighbor_oris_sin_gathered], axis=2
            )  # (N, max_neighbours, 5)

        # Apply communication radius mask for local models
        if self.obs_model.startswith("local"):
            # After normalizing by comm_radius, mask is at 1.0 (neighbors within comm_radius)
            normalized_comm_radius = 1.0
            valid_mask = neighbor_dists_normalized <= normalized_comm_radius
            neighbor_features = np.where(valid_mask[:, :, np.newaxis], neighbor_features, 0.0)
        else:
            # Global models: all neighbors valid up to actual count
            valid_mask = np.zeros((num_agents, self._max_neighbours), dtype=bool)
            valid_mask[:, :actual_neighbors] = True

        # Compute wall distances and bearings (vectorized, no Python loop)
        dx_left = positions[:, 0]
        dx_right = self.world_size - positions[:, 0]
        dy_bottom = positions[:, 1]
        dy_top = self.world_size - positions[:, 1]
        all_dists = np.stack([dx_left, dx_right, dy_bottom, dy_top], axis=1)  # (N, 4)
        wall_dists = np.min(all_dists, axis=1).astype(np.float32)  # (N,)
        min_indices = np.argmin(all_dists, axis=1)  # (N,)

        # Compute wall bearing for each agent (vectorized)
        wall_targets = np.zeros((num_agents, 2), dtype=np.float32)
        mask_left = min_indices == 0
        mask_right = min_indices == 1
        mask_bottom = min_indices == 2
        mask_top = min_indices == 3

        wall_targets[mask_left, 0] = 0.0
        wall_targets[mask_left, 1] = positions[mask_left, 1]

        wall_targets[mask_right, 0] = self.world_size
        wall_targets[mask_right, 1] = positions[mask_right, 1]

        wall_targets[mask_bottom, 0] = positions[mask_bottom, 0]
        wall_targets[mask_bottom, 1] = 0.0

        wall_targets[mask_top, 0] = positions[mask_top, 0]
        wall_targets[mask_top, 1] = self.world_size

        delta_w = wall_targets - positions
        wall_bearings = np.arctan2(delta_w[:, 1], delta_w[:, 0]) - orientations
        wall_bearings = (wall_bearings + np.pi) % (2 * np.pi) - np.pi

        # Convert wall bearing to (cos, sin) for neural network stability
        wall_bearings_cos = np.cos(wall_bearings).astype(np.float32)
        wall_bearings_sin = np.sin(wall_bearings).astype(np.float32)

        # Compute evader features for each pursuer
        evader_diff = self.evader_pos - positions  # (N, 2)

        # CRITICAL: Apply torus wrapping to evader distances when enabled
        if self.torus:
            half = self.world_size / 2.0
            evader_diff = np.where(evader_diff > half, evader_diff - self.world_size, evader_diff)
            evader_diff = np.where(evader_diff < -half, evader_diff + self.world_size, evader_diff)

        evader_dists = np.linalg.norm(evader_diff, axis=1)  # (N,)
        evader_bearings_raw = np.arctan2(evader_diff[:, 1], evader_diff[:, 0])  # (N,)
        evader_bearings = evader_bearings_raw - orientations  # (N,)
        evader_bearings = (evader_bearings + np.pi) % (2 * np.pi) - np.pi

        # Convert evader bearing to (cos, sin) for neural network stability
        evader_bearings_cos = np.cos(evader_bearings).astype(np.float32)
        evader_bearings_sin = np.sin(evader_bearings).astype(np.float32)

        # Normalize distances by obs_radius (matching reward function and Hüttenrauch's implementation)
        evader_dists_normalized = np.minimum(evader_dists / self.obs_radius, 1.0).astype(np.float32)
        wall_dists_normalized = np.minimum(wall_dists / self.world_size, 1.0).astype(np.float32)

        # Mask evader bearings when out of observation range (matching paper specification)
        # When evader is beyond obs_radius, zero out the bearing to indicate "not observed"
        evader_in_range = evader_dists <= self.obs_radius
        evader_bearings_cos = np.where(evader_in_range, evader_bearings_cos, 0.0).astype(np.float32)
        evader_bearings_sin = np.where(evader_in_range, evader_bearings_sin, 0.0).astype(np.float32)

        # Build final observations
        # Local features (6 dims): [wall_dist, wall_bearing_cos, wall_bearing_sin, evader_dist, evader_bearing_cos, evader_bearing_sin]
        # Distances are normalized to [0, 1], bearings (cos/sin) are in [-1, 1]
        local_features = np.stack([wall_dists_normalized, wall_bearings_cos, wall_bearings_sin, evader_dists_normalized, evader_bearings_cos, evader_bearings_sin], axis=1)  # (N, 6)
        neighbor_flat = neighbor_features.reshape(num_agents, self._max_neighbours * self._neighbour_feature_dim)
        mask = valid_mask.astype(np.float32)  # (N, max_neighbours)

        # Concatenate all components: [local | neighbors | mask] (following Hüttenrauch et al. 2019)
        all_obs = np.concatenate([local_features, neighbor_flat, mask], axis=1)

        # Convert to dictionary
        for idx, agent in enumerate(self.agents):
            observations[agent] = all_obs[idx]

        return observations

    def _calculate_rewards(self, actions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """Calculate rewards following Hüttenrauch et al. (2019): r = -min(d_min, obs_radius) / obs_radius"""
        assert self._cached_evader_distances is not None, "Evader distances not cached!"
        distances = self._cached_evader_distances
        min_dist = np.min(distances)

        # Use stored obs_radius for consistency with observations
        shared_reward = -np.minimum(min_dist, self.obs_radius) / self.obs_radius
        rewards = {agent: shared_reward for agent in self.agents}

        return rewards

    def _check_terminations(self) -> Dict[str, bool]:
        """Check if episode should terminate (evader captured)."""
        # Use cached evader distances (computed once in _intermediate_steps)
        assert self._cached_evader_distances is not None, "Evader distances not cached!"
        distances = self._cached_evader_distances
        captured = np.any(distances < self.capture_radius)

        return {agent: captured for agent in self.agents}

    def _check_truncations(self) -> Dict[str, bool]:
        """Truncations are handled via step_count in BaseEnv."""
        return {agent: False for agent in self.agents}

    def _get_infos(self) -> Dict[str, dict]:
        """Return additional information for each agent including task success and capture time."""
        # Use cached evader distances (computed once in _intermediate_steps)
        assert self._cached_evader_distances is not None, "Evader distances not cached!"
        distances = self._cached_evader_distances
        min_dist = np.min(distances)

        # Check if evader is captured
        captured = min_dist < self.capture_radius

        # Record capture time on first capture
        capture_time = None
        if captured and not getattr(self, "_was_captured", False):
            capture_time = self.step_count
            self._was_captured = True

        infos = {}
        for i, agent in enumerate(self.agents):
            infos[agent] = {
                "distance_to_evader": distances[i],
                "min_distance_to_evader": min_dist,
                "evader_captured": captured,
                "task_success": captured,  # For consistency with Rendezvous naming
                "capture_time": capture_time,
            }

        return infos

    def _intermediate_steps(self) -> None:
        """Perform intermediate steps including evader movement and distance caching."""
        # Move evader (simple scripted policy for now)
        self._move_evader()

        # Handle world boundaries (matching Hüttenrauch's approach in base.py)
        if self.torus:
            # Torus: wrap around when crossing boundaries
            self.evader_pos = np.where(self.evader_pos < 0, self.evader_pos + self.world_size, self.evader_pos)
            self.evader_pos = np.where(self.evader_pos > self.world_size, self.evader_pos - self.world_size, self.evader_pos)
        else:
            # Non-torus: clip to world bounds
            self.evader_pos = np.clip(self.evader_pos, 0, self.world_size)

        # Cache evader distances for reuse in rewards, terminations, and infos
        # CRITICAL: Apply torus wrapping to distances when enabled
        diff = self.agent_handler.positions - self.evader_pos  # (N, 2)
        if self.torus:
            half = self.world_size / 2.0
            diff = np.where(diff > half, diff - self.world_size, diff)
            diff = np.where(diff < -half, diff + self.world_size, diff)
        self._cached_evader_distances = np.linalg.norm(diff, axis=1)

    def _move_evader(self) -> None:
        """
        Move evader using the configured evasion strategy.

        Current strategy: Voronoi-based (from Hüttenrauch et al. 2019)
        """
        # Get evasion action from the evasion agent
        direction, speed = self.evasion_agent.compute_evasion_action(
            evader_pos=self.evader_pos, pursuer_positions=self.agent_handler.positions, torus=self.torus
        )

        # Move evader with time scaling to match pursuer kinematics
        self.evader_pos += direction * speed * self.agent_handler.dt

        # Update evader orientation (stored in radians, matching agent_handler convention)
        if np.linalg.norm(direction) > 0:
            self.evader_orientation = np.arctan2(direction[1], direction[0])

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
            self._font = pygame.font.SysFont("Arial", 18)

        if self.render_mode == "human":
            # Clear screen
            self.screen.fill((255, 255, 255))

            # Calculate scaling factor
            scale = self.window_size / self.world_size

            # Draw pursuers (red circles)
            for i in range(self.num_agents):
                pos = self.agent_handler.positions[i] * scale
                pygame.draw.circle(self.screen, (255, 0, 0), pos.astype(int), 5)

                # Draw orientation line (orientations are in radians)
                orient_rad = self.agent_handler.orientations[i]
                end_pos = pos + 10 * np.array([np.cos(orient_rad), np.sin(orient_rad)])
                pygame.draw.line(self.screen, (255, 0, 0), pos.astype(int), end_pos.astype(int), 2)

            # Draw evader (blue circle)
            evader_pos_scaled = self.evader_pos * scale
            pygame.draw.circle(self.screen, (0, 0, 255), evader_pos_scaled.astype(int), 5)

            # Draw evader orientation line (evader_orientation is in radians)
            end_pos = evader_pos_scaled + 10 * np.array([np.cos(self.evader_orientation), np.sin(self.evader_orientation)])
            pygame.draw.line(self.screen, (0, 0, 255), evader_pos_scaled.astype(int), end_pos.astype(int), 2)

            # Display step count
            if self._font is not None:
                text_surface = self._font.render(f"Steps: {self.step_count}", True, (0, 0, 0))
                self.screen.blit(text_surface, (10, 10))

            # Update display and process events to keep window responsive
            pygame.display.flip()
            pygame.event.pump()
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
