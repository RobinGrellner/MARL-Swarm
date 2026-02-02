"""
Evasion Agent for Pursuit-Evasion Environments.

Implements the Voronoi ridge-based evasion strategy from Hüttenrauch et al. (2019).

The evader computes Voronoi diagrams with boundary reflections and moves perpendicular
to Voronoi ridges that separate it from nearby pursuers. This creates a game-theoretic
optimal escape strategy along the "line of control" between the evader and closest threats.

Reference:
- Hüttenrauch, M., Šošić, A., & Neumann, G. (2019). Deep reinforcement
  learning for swarm systems. Journal of Machine Learning Research, 20(54), 1-31.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Voronoi, distance
from shapely.geometry import LineString, Point
from typing import Optional


class HüttenrauchEvasionAgent:
    """
    Evasion agent using Voronoi ridge-based strategy from Hüttenrauch et al. 2019.

    The agent computes Voronoi diagrams with boundary reflections to find control
    lines (ridges) between itself and nearby pursuers, then moves perpendicular to
    these ridges to optimally escape.

    Parameters:
    -----------
    world_size : float
        Side length of the square world
    max_speed : float
        Maximum speed of the evader
    obs_radius : float
        Observation radius (consider only pursuers within this distance)
    torus : bool
        Whether the world has toroidal topology (wraparound)
    """

    def __init__(
        self,
        world_size: float = 10.0,
        max_speed: float = 1.0,
        obs_radius: float = np.inf,
        torus: bool = False,
    ):
        self.world_size = world_size
        self.max_speed = max_speed
        self.obs_radius = obs_radius
        self.torus = torus

    def compute_evasion_action(
        self,
        evader_pos: np.ndarray,
        pursuer_positions: np.ndarray,
        torus: bool = False,
    ) -> tuple[np.ndarray, float]:
        """
        Compute evasion action using Voronoi ridge-based strategy.

        Args:
            evader_pos: Current position of evader (x, y)
            pursuer_positions: Array of pursuer positions, shape (n_pursuers, 2)
            torus: Whether the world has toroidal topology

        Returns:
            Tuple of (direction vector, speed)
        """
        if len(pursuer_positions) == 0:
            # No pursuers, stay still
            return np.array([0.0, 0.0]), self.max_speed

        if len(pursuer_positions) == 1:
            # Only one pursuer, move directly away
            direction = evader_pos - pursuer_positions[0]
            norm = np.linalg.norm(direction)
            if norm > 0:
                direction = direction / norm
            else:
                direction = np.array([1.0, 0.0])
            return direction, self.max_speed

        # Use Voronoi ridge-based approach for multiple pursuers
        return self._voronoi_ridge_strategy(evader_pos, pursuer_positions, torus)

    def _voronoi_ridge_strategy(
        self,
        evader_pos: np.ndarray,
        pursuer_positions: np.ndarray,
        torus: bool = False,
    ) -> tuple[np.ndarray, float]:
        """
        Compute movement direction using Voronoi ridge analysis.

        This follows the approach from Hüttenrauch et al. 2019 exactly:
        1. Create Voronoi diagram with boundary reflections (evader included)
        2. Find the ridge between evader and closest pursuer
        3. Compute optimal escape direction using line-of-control algorithm

        Handles both torus and non-torus worlds.
        """
        try:
            if torus:
                direction = self._voronoi_ridge_strategy_torus(evader_pos, pursuer_positions)
            else:
                direction = self._voronoi_ridge_strategy_nontorus(evader_pos, pursuer_positions)

            if direction is not None:
                norm = np.linalg.norm(direction)
                if norm > 0:
                    direction = direction / norm
                else:
                    direction = self._weighted_escape_direction(evader_pos, pursuer_positions)
            else:
                direction = self._weighted_escape_direction(evader_pos, pursuer_positions)

            return direction, self.max_speed

        except Exception as e:
            # If Voronoi computation fails, use weighted escape
            direction = self._weighted_escape_direction(evader_pos, pursuer_positions)
            return direction, self.max_speed

    def _voronoi_ridge_strategy_nontorus(
        self,
        evader_pos: np.ndarray,
        pursuer_positions: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Voronoi ridge strategy for non-torus world (Hüttenrauch lines 83-106).

        Exactly replicates Hüttenrauch's non-torus approach:
        1. Build nodes with ALL pursuers + evader
        2. Compute distances from evader to all pursuers
        3. Filter by obs_radius, limit to 10
        4. Build Voronoi from subset
        5. Find ridge to closest pursuer
        """
        # Build nodes with ALL pursuers + evader (matching Hüttenrauch lines 84-86)
        nodes = np.vstack([pursuer_positions, evader_pos])
        evader_idx = len(nodes) - 1  # Evader is the last point

        # Compute distances from evader to ALL pursuers
        distances = np.linalg.norm(nodes[:-1, :] - evader_pos, axis=1)

        # Find closest pursuer (Hüttenrauch line 89)
        closest_pursuer = np.where(distances == distances.min())[0]

        # Filter by obs_radius first, then limit to 10 (Hüttenrauch lines 90-92)
        sub_list = list(np.where(distances < self.obs_radius)[0])
        if len(sub_list) > 10:
            sub_list = list(np.argsort(distances)[0:10])

        # Append evader to subset (Hüttenrauch line 93-94)
        sub_list.append(evader_idx)
        evader_sub = len(sub_list) - 1

        # Extract subset of nodes for Voronoi (Hüttenrauch line 96)
        nodes_center_sub = nodes[sub_list, :]

        # Add boundary reflections to subset (Hüttenrauch lines 97-104)
        reflected = self._add_boundary_reflections_correct(nodes_center_sub, torus=False)
        points = np.vstack([nodes_center_sub, reflected])

        # Compute Voronoi diagram (Hüttenrauch line 108)
        vor = Voronoi(points)

        # Find the ridge between evader and closest pursuer (Hüttenrauch lines 112-158)
        direction = self._compute_ridge_escape_direction_nontorus(
            evader_pos, nodes, sub_list, closest_pursuer, evader_sub, vor
        )

        return direction

    def _voronoi_ridge_strategy_torus(
        self,
        evader_pos: np.ndarray,
        pursuer_positions: np.ndarray,
    ) -> Optional[np.ndarray]:
        """
        Voronoi ridge strategy for torus world (Hüttenrauch lines 28-81).

        Exactly replicates Hüttenrauch's torus approach:
        1. Create wraparound copies of pursuers and evader
        2. Stack as [center_pursuers, wraparound_pursuers, center_evader, wraparound_evader]
        3. Determine evader quadrant
        4. Get distances to appropriate evader quadrant copy
        5. Filter and build Voronoi
        6. Find ridge using full node coordinates
        """
        num_pursuers = len(pursuer_positions)

        # Create wraparound copies (Hüttenrauch lines 30-36)
        pursuers_down_right = np.hstack([
            pursuer_positions[:, 0:1] + self.world_size,
            pursuer_positions[:, 1:2]
        ])
        pursuers_up_left = np.hstack([
            pursuer_positions[:, 0:1],
            pursuer_positions[:, 1:2] + self.world_size
        ])
        pursuers_up_right = np.hstack([
            pursuer_positions[:, 0:1] + self.world_size,
            pursuer_positions[:, 1:2] + self.world_size
        ])

        evader_down_right = np.hstack([
            evader_pos[0:1] + self.world_size,
            evader_pos[1:2]
        ])
        evader_up_left = np.hstack([
            evader_pos[0:1],
            evader_pos[1:2] + self.world_size
        ])
        evader_up_right = np.hstack([
            evader_pos[0:1] + self.world_size,
            evader_pos[1:2] + self.world_size
        ])

        # Stack all copies (Hüttenrauch lines 42-49)
        nodes = np.vstack([
            pursuer_positions,           # 0:num_pursuers
            pursuers_down_right,         # num_pursuers:2*num_pursuers
            pursuers_up_left,            # 2*num_pursuers:3*num_pursuers
            pursuers_up_right,           # 3*num_pursuers:4*num_pursuers
            evader_pos.reshape(1, -1),   # 4*num_pursuers
            evader_down_right,           # 4*num_pursuers+1
            evader_up_left,              # 4*num_pursuers+2
            evader_up_right,             # 4*num_pursuers+3
        ])

        # Compute distance matrix (Hüttenrauch line 51)
        dist_matrix = np.linalg.norm(nodes[:, np.newaxis, :] - nodes[np.newaxis, :, :], axis=2)

        # Determine evader quadrant (Hüttenrauch lines 53-61)
        quadrant_check = np.sign(evader_pos - self.world_size / 2)
        if np.all(quadrant_check == np.array([1, 1])):
            evader_quadrant = 0
        elif np.all(quadrant_check == np.array([-1, 1])):
            evader_quadrant = 1
        elif np.all(quadrant_check == np.array([1, -1])):
            evader_quadrant = 2
        else:  # [-1, -1]
            evader_quadrant = 3

        # Get distances from center pursuers to appropriate evader quadrant (Hüttenrauch line 63)
        evader_dist = dist_matrix[:-4, 4 * num_pursuers + evader_quadrant]

        # Find closest pursuer (Hüttenrauch line 69)
        closest_pursuer = np.where(evader_dist == evader_dist.min())[0]

        # Filter by obs_radius, limit to 10 (Hüttenrauch lines 64-66)
        sub_list = list(np.where(evader_dist < self.obs_radius)[0])
        if len(sub_list) > 10:
            sub_list = list(np.argsort(evader_dist)[0:10])

        # Add evader to subset (Hüttenrauch lines 67-68)
        sub_list.append(4 * num_pursuers + evader_quadrant)
        evader_sub = len(sub_list) - 1

        # Extract subset of nodes (Hüttenrauch line 71)
        nodes_center_sub = nodes[sub_list, :]

        # Add boundary reflections (Hüttenrauch lines 72-81, with torus-aware bounding box)
        reflected = self._add_boundary_reflections_correct(nodes_center_sub, torus=True)
        points = np.vstack([nodes_center_sub, reflected])

        # Compute Voronoi (Hüttenrauch line 108)
        vor = Voronoi(points)

        # Find ridge (Hüttenrauch lines 112-158)
        direction = self._compute_ridge_escape_direction_torus(
            evader_pos, nodes, sub_list, closest_pursuer, evader_sub, vor, num_pursuers
        )

        return direction

    def _add_boundary_reflections_correct(
        self,
        nodes: np.ndarray,
        torus: bool = False,
    ) -> np.ndarray:
        """
        Add boundary reflections matching Hüttenrauch's exact formula.

        Uses mirror reflection across boundary planes:
        - Left (x=0): x' = -x
        - Right (x=world_size or 2*world_size for torus): x' = bbox[1] - x
        - Down (y=0): y' = -y
        - Up (y=world_size or 2*world_size for torus): y' = bbox[3] - y

        For torus worlds, bbox is [0, 2*world_size, 0, 2*world_size] because nodes can be
        shifted by world_size. For non-torus, bbox is [0, world_size, 0, world_size].

        This is the EXACT approach from the original Hüttenrauch code (lines 96-106 for non-torus,
        lines 72-81 with adjusted bbox for torus).
        """
        # Torus worlds have coordinates up to 2*world_size due to wraparound copies
        if torus:
            bbox = np.array([0.0, 2.0 * self.world_size, 0.0, 2.0 * self.world_size])
        else:
            bbox = np.array([0.0, self.world_size, 0.0, self.world_size])

        # Left reflection: x' = bbox[0] - (x - bbox[0]) = -x
        nodes_left = np.copy(nodes)
        nodes_left[:, 0] = bbox[0] - (nodes[:, 0] - bbox[0])

        # Right reflection: x' = bbox[1] + (bbox[1] - x) = 2*bbox[1] - x
        nodes_right = np.copy(nodes)
        nodes_right[:, 0] = bbox[1] + (bbox[1] - nodes[:, 0])

        # Down reflection: y' = bbox[2] - (y - bbox[2]) = -y
        nodes_down = np.copy(nodes)
        nodes_down[:, 1] = bbox[2] - (nodes[:, 1] - bbox[2])

        # Up reflection: y' = bbox[3] + (bbox[3] - y) = 2*bbox[3] - y
        nodes_up = np.copy(nodes)
        nodes_up[:, 1] = bbox[3] + (bbox[3] - nodes[:, 1])

        # Stack only 4 reflections (not 8): [down, left, right, up]
        return np.vstack([nodes_down, nodes_left, nodes_right, nodes_up])

    def _compute_ridge_escape_direction_nontorus(
        self,
        evader_pos: np.ndarray,
        nodes: np.ndarray,
        sub_list: list,
        closest_pursuer_indices: np.ndarray,
        evader_sub: int,
        vor: Voronoi,
    ) -> Optional[np.ndarray]:
        """
        Compute escape direction for non-torus world (Hüttenrauch lines 112-158).

        Args:
            evader_pos: Evader position [x, y]
            nodes: Full nodes array with ALL pursuers + evader
            sub_list: List of indices into nodes to use for Voronoi
            closest_pursuer_indices: Array of indices (in full nodes) of closest pursuers (ties possible)
            evader_sub: Index of evader within sub_list
            vor: Voronoi diagram of the subset
        """
        d = None

        # Iterate through all ridges in Voronoi (Hüttenrauch line 112)
        for i, ridge in enumerate(vor.ridge_points):
            # Check: evader is one of the ridge points AND both are from original set (Hüttenrauch line 113)
            if evader_sub in set(ridge) and np.all([r <= evader_sub for r in ridge]):
                # Find the neighbor (other point in the ridge, mapped to original indices)
                neighbor = min([sub_list[r] for r in ridge])

                # Only process if this neighbor is the closest pursuer (Hüttenrauch line 120)
                if neighbor in closest_pursuer_indices:
                    ridge_inds = vor.ridge_vertices[i]

                    # Get ridge vertices
                    if ridge_inds[0] >= 0 and ridge_inds[1] >= 0:
                        a = vor.vertices[ridge_inds[0], :]
                        b = vor.vertices[ridge_inds[1], :]

                        # Line of control: the Voronoi ridge (Hüttenrauch line 125-126)
                        line_of_control = b - a
                        L_i = np.linalg.norm(line_of_control)

                        if L_i < 1e-6:
                            continue

                        # Vector from evader to closest pursuer (Hüttenrauch line 131)
                        xi = nodes[neighbor, :] - evader_pos
                        xi_norm = np.linalg.norm(xi)

                        if xi_norm < 1e-6:
                            continue

                        # Basis vectors (Hüttenrauch lines 132-133)
                        eta_h_i = xi / xi_norm
                        eta_v_i = np.array([-eta_h_i[1], eta_h_i[0]])

                        # Find intersection (Hüttenrauch lines 135-153)
                        try:
                            line1 = LineString([evader_pos, nodes[neighbor, :]])
                            line2 = LineString([a, b])
                            intersection = line1.intersection(line2)

                            if not intersection.is_empty:
                                inter_point = np.array(intersection.coords[0])
                                if np.dot(line_of_control, eta_v_i) > 0:
                                    l_i = np.linalg.norm(a - inter_point)
                                else:
                                    l_i = np.linalg.norm(b - inter_point)
                            else:
                                if np.dot(line_of_control, eta_v_i) > 0:
                                    l_i = 0.0
                                else:
                                    l_i = L_i
                        except Exception:
                            l_i = L_i / 2.0

                        # Directional components (Hüttenrauch lines 155-156)
                        alpha_h_i = -L_i / 2.0
                        alpha_v_i = (l_i ** 2 - (L_i - l_i) ** 2) / (2.0 * xi_norm)

                        denom = np.sqrt(alpha_h_i ** 2 + alpha_v_i ** 2)
                        if denom < 1e-6:
                            continue

                        # Escape direction (Hüttenrauch line 158)
                        d = (alpha_h_i * eta_h_i - alpha_v_i * eta_v_i) / denom

        return d

    def _compute_ridge_escape_direction_torus(
        self,
        evader_pos: np.ndarray,
        nodes: np.ndarray,
        sub_list: list,
        closest_pursuer_indices: np.ndarray,
        evader_sub: int,
        vor: Voronoi,
        num_pursuers: int,
    ) -> Optional[np.ndarray]:
        """
        Compute escape direction for torus world (Hüttenrauch lines 112-158).

        Args:
            evader_pos: Center evader position [x, y] (used for reference)
            nodes: Full nodes array with ALL wraparound copies
            sub_list: List of indices into nodes to use for Voronoi
            closest_pursuer_indices: Array of indices (in full nodes) of closest pursuers
            evader_sub: Index of evader within sub_list
            vor: Voronoi diagram of the subset
            num_pursuers: Number of original pursuers (for wrapping calculations)
        """
        d = None

        # Iterate through all ridges (Hüttenrauch line 112)
        for i, ridge in enumerate(vor.ridge_points):
            # Check: evader is in ridge AND both from original set (Hüttenrauch line 113)
            if evader_sub in set(ridge) and np.all([r <= evader_sub for r in ridge]):
                # Find neighbor, map through sub_list to get original node index (Hüttenrauch line 115-118)
                neighbor = min([sub_list[r] for r in ridge])

                # Only process if neighbor is the closest pursuer (Hüttenrauch line 120)
                if neighbor in closest_pursuer_indices:
                    ridge_inds = vor.ridge_vertices[i]

                    # Get ridge vertices
                    if ridge_inds[0] >= 0 and ridge_inds[1] >= 0:
                        a = vor.vertices[ridge_inds[0], :]
                        b = vor.vertices[ridge_inds[1], :]

                        # Line of control (Hüttenrauch line 125-126)
                        line_of_control = b - a
                        L_i = np.linalg.norm(line_of_control)

                        if L_i < 1e-6:
                            continue

                        # Vector from evader to closest pursuer (using FULL node positions for torus!)
                        # This is the key difference from non-torus: use nodes[neighbor] not pursuer_positions
                        xi = nodes[neighbor, :] - nodes[sub_list[evader_sub], :]
                        xi_norm = np.linalg.norm(xi)

                        if xi_norm < 1e-6:
                            continue

                        # Basis vectors (Hüttenrauch lines 132-133)
                        eta_h_i = xi / xi_norm
                        eta_v_i = np.array([-eta_h_i[1], eta_h_i[0]])

                        # Find intersection (Hüttenrauch lines 135-153)
                        try:
                            line1 = LineString([nodes[sub_list[evader_sub], :], nodes[neighbor, :]])
                            line2 = LineString([a, b])
                            intersection = line1.intersection(line2)

                            if not intersection.is_empty:
                                inter_point = np.array(intersection.coords[0])
                                if np.dot(line_of_control, eta_v_i) > 0:
                                    l_i = np.linalg.norm(a - inter_point)
                                else:
                                    l_i = np.linalg.norm(b - inter_point)
                            else:
                                if np.dot(line_of_control, eta_v_i) > 0:
                                    l_i = 0.0
                                else:
                                    l_i = L_i
                        except Exception:
                            l_i = L_i / 2.0

                        # Directional components (Hüttenrauch lines 155-156)
                        alpha_h_i = -L_i / 2.0
                        alpha_v_i = (l_i ** 2 - (L_i - l_i) ** 2) / (2.0 * xi_norm)

                        denom = np.sqrt(alpha_h_i ** 2 + alpha_v_i ** 2)
                        if denom < 1e-6:
                            continue

                        # Escape direction (Hüttenrauch line 158)
                        d = (alpha_h_i * eta_h_i - alpha_v_i * eta_v_i) / denom

        return d

    def _compute_ridge_escape_direction_correct(
        self,
        evader_pos: np.ndarray,
        pursuer_positions: np.ndarray,
        nodes: np.ndarray,
        evader_idx: int,
        closest_pursuer_idx: int,
        vor: Voronoi,
    ) -> Optional[np.ndarray]:
        """
        Compute escape direction using line-of-control algorithm from Hüttenrauch et al. 2019.

        This exactly implements the ridge selection from the original code (lines 112-158):
        1. Find ridges that border the evader's Voronoi cell
        2. Filter to only the ridge connected to the closest pursuer
        3. Compute directional components (alpha_h_i, alpha_v_i)
        4. Return the escape direction
        """
        closest_pursuer = pursuer_positions[closest_pursuer_idx]

        # Iterate through all ridges in the Voronoi diagram
        for i, ridge in enumerate(vor.ridge_points):
            ridge_vertices = vor.ridge_vertices[i]

            # CRITICAL FILTER (Hüttenrauch line 113):
            # Ridge must include evader AND both points must be from original (non-reflected) set
            if evader_idx in set(ridge) and np.all([r <= evader_idx for r in ridge]):
                # Find the neighbor (the other point in this ridge, which should be a pursuer)
                neighbor = min([r for r in ridge if r != evader_idx])

                # CRITICAL CHECK (Hüttenrauch line 120):
                # Only process if this neighbor is the closest pursuer
                if neighbor == closest_pursuer_idx:
                    ridge_inds = ridge_vertices

                    # Only process if ridge has valid vertices
                    if ridge_inds[0] >= 0 and ridge_inds[1] >= 0:
                        a = vor.vertices[ridge_inds[0], :]
                        b = vor.vertices[ridge_inds[1], :]

                        # Line of control: the Voronoi ridge
                        line_of_control = b - a
                        L_i = np.linalg.norm(line_of_control)

                        if L_i < 1e-6:
                            continue

                        # Vector from evader to closest pursuer
                        xi = closest_pursuer - evader_pos
                        xi_norm = np.linalg.norm(xi)

                        if xi_norm < 1e-6:
                            continue

                        # Horizontal and vertical basis vectors
                        eta_h_i = xi / xi_norm  # Unit vector along xi
                        eta_v_i = np.array([-eta_h_i[1], eta_h_i[0]])  # Perpendicular

                        # Find intersection of line (evader to pursuer) with ridge (a to b)
                        try:
                            line1 = LineString([evader_pos, closest_pursuer])
                            line2 = LineString([a, b])
                            intersection = line1.intersection(line2)

                            if not intersection.is_empty:
                                inter_point = np.array(intersection.coords[0])
                                # l_i is distance along ridge from a to intersection point
                                if np.dot(line_of_control, eta_v_i) > 0:
                                    l_i = np.linalg.norm(a - inter_point)
                                else:
                                    l_i = np.linalg.norm(b - inter_point)
                            else:
                                # No intersection, use endpoint logic
                                if np.dot(line_of_control, eta_v_i) > 0:
                                    l_i = 0.0
                                else:
                                    l_i = L_i
                        except Exception:
                            l_i = L_i / 2.0

                        # Compute directional components (Hüttenrauch line 155-156)
                        alpha_h_i = -L_i / 2.0
                        alpha_v_i = (l_i ** 2 - (L_i - l_i) ** 2) / (2.0 * xi_norm)

                        denom = np.sqrt(alpha_h_i ** 2 + alpha_v_i ** 2)
                        if denom < 1e-6:
                            continue

                        # Optimal escape direction (Hüttenrauch line 158)
                        d = (alpha_h_i * eta_h_i - alpha_v_i * eta_v_i) / denom
                        return d

        # No valid ridge found
        return None

    def _weighted_escape_direction(
        self,
        evader_pos: np.ndarray,
        pursuer_positions: np.ndarray,
    ) -> np.ndarray:
        """
        Fallback: compute weighted escape direction away from all pursuers.

        Weights inversely by distance cubed (closer pursuers matter much more).
        """
        displacements = evader_pos - pursuer_positions
        distances = np.linalg.norm(displacements, axis=1)
        distances = np.maximum(distances, 1e-3)

        # Weight by inverse distance cubed
        weights = 1.0 / (distances**3)
        weights = weights / np.sum(weights)

        # Weighted sum of escape directions
        escape_direction = np.sum(weights[:, np.newaxis] * displacements, axis=0)

        # Normalize to unit vector
        norm = np.linalg.norm(escape_direction)
        if norm > 1e-6:
            escape_direction = escape_direction / norm
        else:
            # Random direction if trapped
            angle = np.random.uniform(0, 2 * np.pi)
            escape_direction = np.array([np.cos(angle), np.sin(angle)])

        return escape_direction


def create_evasion_agent(
    strategy: str = "huttenrauch",
    world_size: float = 10.0,
    max_speed: float = 1.0,
    obs_radius: float = np.inf,
    torus: bool = False,
) -> HüttenrauchEvasionAgent:
    """
    Factory function to create an evasion agent.

    Args:
        strategy: Evasion strategy. Currently only "huttenrauch" is fully supported.
                 Other strategies are mapped to the Hüttenrauch implementation.
        world_size: Size of the world
        max_speed: Maximum speed of the evader
        obs_radius: Observation radius for nearby pursuers
        torus: Whether the world has toroidal topology

    Returns:
        Evasion agent instance
    """
    # All strategies now use the Hüttenrauch approach
    return HüttenrauchEvasionAgent(
        world_size=world_size,
        max_speed=max_speed,
        obs_radius=obs_radius,
        torus=torus,
    )
