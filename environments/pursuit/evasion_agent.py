"""
Evasion Agent for Pursuit-Evasion Environments.

Implements a Voronoi-based evasion strategy where the evader moves toward
the largest gap between pursuers, maximizing its minimum distance to all pursuers.

This strategy is inspired by Hüttenrauch et al. (2019) and is based on:
1. Finding the direction that maximizes distance to the nearest pursuer
2. Considering Voronoi regions (areas closer to the evader than to any pursuer)
3. Moving toward the center of the largest Voronoi cell

Reference:
- Hüttenrauch, M., Šošić, A., & Neumann, G. (2019). Deep reinforcement
  learning for swarm systems. Journal of Machine Learning Research, 20(54), 1-31.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
from typing import Optional


class VoronoiEvasionAgent:
    """
    Evasion agent using Voronoi-based strategy.

    The agent computes the Voronoi diagram of pursuer positions and moves
    toward the point that maximizes the minimum distance to all pursuers.

    Parameters:
    -----------
    world_size : float
        Side length of the square world
    max_speed : float
        Maximum speed of the evader
    strategy : str
        Evasion strategy to use:
        - "voronoi_center": Move toward center of largest Voronoi cell
        - "max_min_distance": Move toward direction that maximizes minimum distance
        - "weighted_escape": Weighted combination of escaping all pursuers
    """

    def __init__(
        self,
        world_size: float = 10.0,
        max_speed: float = 1.0,
        strategy: str = "max_min_distance",
    ):
        self.world_size = world_size
        self.max_speed = max_speed
        self.strategy = strategy

    def compute_evasion_action(
        self,
        evader_pos: np.ndarray,
        pursuer_positions: np.ndarray,
        torus: bool = False,
    ) -> tuple[np.ndarray, float]:
        """
        Compute evasion action for the evader.

        Args:
            evader_pos: Current position of evader (x, y)
            pursuer_positions: Array of pursuer positions, shape (n_pursuers, 2)
            torus: Whether the world has toroidal topology

        Returns:
            Tuple of (direction vector, speed)
        """
        if self.strategy == "voronoi_center":
            return self._voronoi_center_strategy(evader_pos, pursuer_positions)
        elif self.strategy == "max_min_distance":
            return self._max_min_distance_strategy(evader_pos, pursuer_positions, torus)
        elif self.strategy == "weighted_escape":
            return self._weighted_escape_strategy(evader_pos, pursuer_positions, torus)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _max_min_distance_strategy(
        self,
        evader_pos: np.ndarray,
        pursuer_positions: np.ndarray,
        torus: bool = False,
    ) -> tuple[np.ndarray, float]:
        """
        Move in the direction that maximizes the minimum distance to pursuers.

        This is a greedy approximation of the Voronoi-based strategy that
        samples multiple candidate directions and chooses the one that leads
        to the largest gap between pursuers.
        """
        # Sample candidate directions (8 cardinal + intercardinal directions)
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        candidate_directions = np.array([[np.cos(angle), np.sin(angle)] for angle in angles])

        best_direction = None
        best_min_distance = -np.inf

        # Evaluate each candidate direction
        for direction in candidate_directions:
            # Simulate moving in this direction
            candidate_pos = evader_pos + direction * self.max_speed

            # Clip to world bounds if not torus
            if not torus:
                candidate_pos = np.clip(candidate_pos, 0, self.world_size)

            # Compute minimum distance to all pursuers
            if torus:
                min_dist = self._min_torus_distance(candidate_pos, pursuer_positions)
            else:
                distances = np.linalg.norm(pursuer_positions - candidate_pos, axis=1)
                min_dist = np.min(distances)

            # Choose direction that maximizes minimum distance
            if min_dist > best_min_distance:
                best_min_distance = min_dist
                best_direction = direction

        return best_direction, self.max_speed

    def _weighted_escape_strategy(
        self,
        evader_pos: np.ndarray,
        pursuer_positions: np.ndarray,
        torus: bool = False,
    ) -> tuple[np.ndarray, float]:
        """
        Move away from all pursuers with weights inversely proportional to distance.

        Closer pursuers have more influence on the escape direction.
        """
        if torus:
            # Compute torus-aware displacement vectors
            displacements = self._torus_displacements(evader_pos, pursuer_positions)
        else:
            displacements = evader_pos - pursuer_positions

        distances = np.linalg.norm(displacements, axis=1)

        # Avoid division by zero
        distances = np.maximum(distances, 1e-6)

        # Weight by inverse distance squared (closer pursuers matter more)
        weights = 1.0 / (distances**2)
        weights = weights / np.sum(weights)  # Normalize

        # Weighted sum of escape directions
        escape_direction = np.sum(weights[:, np.newaxis] * displacements, axis=0)

        # Normalize to unit vector
        norm = np.linalg.norm(escape_direction)
        if norm > 0:
            escape_direction = escape_direction / norm
        else:
            # If at exact same position as a pursuer, move randomly
            angle = np.random.uniform(0, 2 * np.pi)
            escape_direction = np.array([np.cos(angle), np.sin(angle)])

        return escape_direction, self.max_speed

    def _voronoi_center_strategy(
        self,
        evader_pos: np.ndarray,
        pursuer_positions: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Move toward the center of the largest Voronoi cell.

        This uses scipy.spatial.Voronoi to compute the exact Voronoi diagram
        and finds the largest cell to escape into.
        """
        # Need at least 3 points for Voronoi
        if len(pursuer_positions) < 3:
            # Fall back to weighted escape if too few pursuers
            return self._weighted_escape_strategy(evader_pos, pursuer_positions, False)

        try:
            # Add corner points to bound the Voronoi diagram
            corners = np.array([[0, 0], [0, self.world_size], [self.world_size, 0], [self.world_size, self.world_size]])
            points = np.vstack([pursuer_positions, corners])

            # Compute Voronoi diagram
            vor = Voronoi(points)

            # Find the Voronoi cell containing the evader
            # This is a simplified heuristic: move toward the vertex of the
            # Voronoi diagram that is farthest from all pursuers

            vertices = vor.vertices
            if len(vertices) == 0:
                # Fall back if Voronoi computation fails
                return self._weighted_escape_strategy(evader_pos, pursuer_positions, False)

            # Find the Voronoi vertex farthest from all pursuers
            min_dists_to_pursuers = []
            valid_vertices = []

            for vertex in vertices:
                # Skip vertices outside the world
                if vertex[0] < 0 or vertex[0] > self.world_size or vertex[1] < 0 or vertex[1] > self.world_size:
                    continue

                # Compute minimum distance from this vertex to any pursuer
                dists = np.linalg.norm(pursuer_positions - vertex, axis=1)
                min_dist = np.min(dists)

                valid_vertices.append(vertex)
                min_dists_to_pursuers.append(min_dist)

            if len(valid_vertices) == 0:
                # Fall back if no valid vertices
                return self._weighted_escape_strategy(evader_pos, pursuer_positions, False)

            # Choose the vertex with the largest minimum distance (safest spot)
            best_idx = np.argmax(min_dists_to_pursuers)
            target = valid_vertices[best_idx]

            # Move toward this target
            direction = target - evader_pos
            norm = np.linalg.norm(direction)

            if norm > 0:
                direction = direction / norm
            else:
                # Already at target, stay still
                direction = np.array([0.0, 0.0])

            return direction, self.max_speed

        except Exception as e:
            # If Voronoi computation fails (e.g., degenerate configuration),
            # fall back to weighted escape
            print(f"Voronoi computation failed: {e}. Falling back to weighted escape.")
            return self._weighted_escape_strategy(evader_pos, pursuer_positions, False)

    def _min_torus_distance(
        self,
        pos: np.ndarray,
        points: np.ndarray,
    ) -> float:
        """Compute minimum torus-wrapped distance from pos to any point in points."""
        displacements = self._torus_displacements(pos, points)
        distances = np.linalg.norm(displacements, axis=1)
        return np.min(distances)

    def _torus_displacements(
        self,
        pos: np.ndarray,
        points: np.ndarray,
    ) -> np.ndarray:
        """Compute torus-wrapped displacement vectors from pos to all points."""
        diff = pos - points
        half = self.world_size / 2.0

        # Wrap displacements
        diff = np.where(diff > half, diff - self.world_size, diff)
        diff = np.where(diff < -half, diff + self.world_size, diff)

        return diff


class SimpleEvasionAgent:
    """
    Simple evasion agent that moves away from the nearest pursuer.

    This is a baseline strategy for comparison.
    """

    def __init__(self, max_speed: float = 1.0):
        self.max_speed = max_speed

    def compute_evasion_action(
        self,
        evader_pos: np.ndarray,
        pursuer_positions: np.ndarray,
        torus: bool = False,
    ) -> tuple[np.ndarray, float]:
        """Move directly away from the nearest pursuer."""
        # Find nearest pursuer
        distances = np.linalg.norm(pursuer_positions - evader_pos, axis=1)
        nearest_idx = np.argmin(distances)
        nearest_pos = pursuer_positions[nearest_idx]

        # Move away from nearest pursuer
        direction = evader_pos - nearest_pos
        norm = np.linalg.norm(direction)

        if norm > 0:
            direction = direction / norm
        else:
            # If at same position, move randomly
            angle = np.random.uniform(0, 2 * np.pi)
            direction = np.array([np.cos(angle), np.sin(angle)])

        return direction, self.max_speed


def create_evasion_agent(
    strategy: str = "max_min_distance",
    world_size: float = 10.0,
    max_speed: float = 1.0,
) -> VoronoiEvasionAgent | SimpleEvasionAgent:
    """
    Factory function to create an evasion agent.

    Args:
        strategy: Evasion strategy ("simple", "voronoi_center", "max_min_distance", "weighted_escape")
        world_size: Size of the world
        max_speed: Maximum speed of the evader

    Returns:
        Evasion agent instance
    """
    if strategy == "simple":
        return SimpleEvasionAgent(max_speed=max_speed)
    else:
        return VoronoiEvasionAgent(
            world_size=world_size,
            max_speed=max_speed,
            strategy=strategy,
        )
