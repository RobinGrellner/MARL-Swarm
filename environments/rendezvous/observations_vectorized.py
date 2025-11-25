"""
Vectorized observation construction for rendezvous environment.

This module provides high-performance observation computation using
vectorized NumPy operations instead of Python loops.
"""

from typing import Dict
import numpy as np


def compute_observations_vectorized(
    positions: np.ndarray,
    orientations: np.ndarray,
    linear_vels: np.ndarray,
    angular_vels: np.ndarray,
    agent_names: list,
    obs_model: str,
    kinematics: str,
    world_size: float,
    torus: bool,
    comm_radius: float,
    max_neighbours: int,
    neighbour_feature_dim: int,
    v_max: float = 1.0,
) -> Dict[str, np.ndarray]:
    """Construct observations for all agents using vectorized operations.

    Parameters
    ----------
    positions : np.ndarray
        Agent positions, shape (N, 2)
    orientations : np.ndarray
        Agent orientations, shape (N,)
    linear_vels : np.ndarray
        Linear velocities, shape (N,)
    angular_vels : np.ndarray
        Angular velocities, shape (N,)
    agent_names : list
        List of agent name strings
    obs_model : str
        Observation model type
    kinematics : str
        "single" or "double"
    world_size : float
        Side length of square world
    torus : bool
        Whether world wraps around
    comm_radius : float
        Communication radius for local observations
    max_neighbours : int
        Maximum number of neighbors in observation
    neighbour_feature_dim : int
        Feature dimension per neighbor
    v_max : float
        Maximum linear velocity (for velocity normalization)

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary mapping agent names to observation vectors
    """
    num_agents = len(positions)

    # ===================================================================
    # Classic observation model (special case)
    # ===================================================================
    if obs_model == "classic":
        observations = {}
        mean_pos = np.mean(positions, axis=0)
        for idx, name in enumerate(agent_names):
            if kinematics == "single":
                obs_vec = np.array(
                    [
                        positions[idx, 0],
                        positions[idx, 1],
                        mean_pos[0],
                        mean_pos[1],
                        linear_vels[idx],
                        orientations[idx],
                    ],
                    dtype=np.float32,
                )
            else:
                obs_vec = np.array(
                    [
                        positions[idx, 0],
                        positions[idx, 1],
                        mean_pos[0],
                        mean_pos[1],
                        linear_vels[idx],
                        angular_vels[idx],
                        orientations[idx],
                    ],
                    dtype=np.float32,
                )
            observations[name] = obs_vec
        return observations

    # ===================================================================
    # Vectorized observation construction
    # ===================================================================

    # Step 1: Compute ALL pairwise displacements (N, N, 2)
    pos_i = positions[:, np.newaxis, :]  # (N, 1, 2)
    pos_j = positions[np.newaxis, :, :]  # (1, N, 2)
    diff = pos_j - pos_i  # (N, N, 2) via broadcasting

    # Apply torus wrapping if needed
    if torus:
        half = world_size / 2.0
        diff = np.where(diff > half, diff - world_size, diff)
        diff = np.where(diff < -half, diff + world_size, diff)

    # Step 2: Compute ALL pairwise distances (N, N)
    distances = np.linalg.norm(diff, axis=2)

    # Step 3: Compute ALL pairwise bearings (N, N)
    bearings_raw = np.arctan2(diff[:, :, 1], diff[:, :, 0])  # (N, N)
    orientations_expanded = orientations[:, np.newaxis]  # (N, 1)
    bearings = bearings_raw - orientations_expanded  # (N, N)
    bearings = (bearings + np.pi) % (2 * np.pi) - np.pi  # Wrap to [-π, π]

    # Step 4: Sort neighbors by distance for each agent
    # Use stable sort for deterministic tie-breaking (ensures reproducibility)
    sorted_indices = np.argsort(distances, axis=1, kind="stable")  # (N, N)

    # Step 5: Compute local features (wall distance and bearing)
    if torus:
        wall_dists = np.full(num_agents, world_size, dtype=np.float32)
        wall_bearings = np.zeros(num_agents, dtype=np.float32)
    else:
        # Vectorized wall distance computation
        dx_left = positions[:, 0]
        dx_right = world_size - positions[:, 0]
        dy_bottom = positions[:, 1]
        dy_top = world_size - positions[:, 1]
        all_dists = np.stack([dx_left, dx_right, dy_bottom, dy_top], axis=1)  # (N, 4)
        wall_dists = np.min(all_dists, axis=1).astype(np.float32)  # (N,)
        min_indices = np.argmin(all_dists, axis=1)  # (N,)

    # Normalize wall distances by world_size
    wall_dists = wall_dists / world_size

    if not torus:
        # Compute wall bearing for each agent
        wall_targets = np.zeros((num_agents, 2), dtype=np.float32)
        mask_left = min_indices == 0
        mask_right = min_indices == 1
        mask_bottom = min_indices == 2
        mask_top = min_indices == 3

        wall_targets[mask_left, 0] = 0.0
        wall_targets[mask_left, 1] = positions[mask_left, 1]

        wall_targets[mask_right, 0] = world_size
        wall_targets[mask_right, 1] = positions[mask_right, 1]

        wall_targets[mask_bottom, 0] = positions[mask_bottom, 0]
        wall_targets[mask_bottom, 1] = 0.0

        wall_targets[mask_top, 0] = positions[mask_top, 0]
        wall_targets[mask_top, 1] = world_size

        delta_w = wall_targets - positions
        wall_bearings = np.arctan2(delta_w[:, 1], delta_w[:, 0]) - orientations
        wall_bearings = (wall_bearings + np.pi) % (2 * np.pi) - np.pi
        wall_bearings = wall_bearings.astype(np.float32)

    # Step 6: Select top max_neighbours for each agent (excluding self)
    # sorted_indices[:, 0] is self, so take indices 1 onwards
    # Handle scale-invariant case where max_neighbours > num_agents - 1
    actual_neighbors = min(num_agents - 1, max_neighbours)
    neighbor_indices_available = sorted_indices[:, 1 : actual_neighbors + 1]  # (N, actual_neighbors)

    # Pad to max_neighbours (zero-width if actual_neighbors == max_neighbours)
    padding = np.zeros((num_agents, max_neighbours - actual_neighbors), dtype=int)
    neighbor_indices = np.concatenate([neighbor_indices_available, padding], axis=1)

    # Gather distances and bearings for selected neighbors
    row_indices = np.arange(num_agents)[:, np.newaxis]  # (N, 1)
    neighbor_dists = distances[row_indices, neighbor_indices]  # (N, max_neighbours)
    neighbor_bears = bearings[row_indices, neighbor_indices]  # (N, max_neighbours)

    # Normalize neighbor distances by world_size
    neighbor_dists = neighbor_dists / world_size

    # Step 7: Compute additional features based on observation model
    if obs_model in {"global_basic", "local_basic"}:
        # Only distance and bearing
        neighbor_features = np.stack([neighbor_dists, neighbor_bears], axis=2)  # (N, max_neighbours, 2)

    elif obs_model in {"global_extended", "local_extended", "local_comm"}:
        # Compute relative orientations (needed by all three)
        orientations_matrix = orientations[np.newaxis, :] - orientations[:, np.newaxis]  # (N, N)
        orientations_matrix = (orientations_matrix + np.pi) % (2 * np.pi) - np.pi
        neighbor_oris = orientations_matrix[row_indices, neighbor_indices]  # (N, max_neighbours)

        if obs_model == "global_extended":
            # Need distance, bearing, relative orientation, relative velocity
            velocity_vectors = np.stack(
                [linear_vels * np.cos(orientations), linear_vels * np.sin(orientations)], axis=1
            )  # (N, 2)

            vel_diff = velocity_vectors[:, np.newaxis, :] - velocity_vectors[np.newaxis, :, :]  # (N, N, 2)
            neighbor_vels = vel_diff[row_indices, neighbor_indices, :]  # (N, max_neighbours, 2)

            # Normalize velocities by 2 * v_max (following Hüttenrauch)
            neighbor_vels = neighbor_vels / (2.0 * v_max)

            neighbor_features = np.stack(
                [neighbor_dists, neighbor_bears, neighbor_oris, neighbor_vels[:, :, 0], neighbor_vels[:, :, 1]], axis=2
            )  # (N, max_neighbours, 5)

        elif obs_model == "local_extended":
            # Need distance, bearing, relative orientation
            neighbor_features = np.stack(
                [neighbor_dists, neighbor_bears, neighbor_oris], axis=2
            )  # (N, max_neighbours, 3)

        elif obs_model == "local_comm":
            # Need distance, bearing, relative orientation, neighbor's neighborhood count
            # Compute neighborhood counts for all agents (vectorized)
            within_comm = distances <= comm_radius  # (N, N)
            np.fill_diagonal(within_comm, False)  # Don't count self
            neighborhood_counts = within_comm.sum(axis=1).astype(np.float32)  # (N,)

            # Gather neighbor's neighborhood counts
            neighbor_counts = neighborhood_counts[neighbor_indices]  # (N, max_neighbours)

            # Normalize neighborhood counts by (num_agents - 1)
            neighbor_counts = neighbor_counts / (num_agents - 1)

            neighbor_features = np.stack(
                [neighbor_dists, neighbor_bears, neighbor_oris, neighbor_counts], axis=2
            )  # (N, max_neighbours, 4)

    # Step 8: Apply communication radius mask (local vs global)
    if obs_model.startswith("local"):
        # Normalize comm_radius for comparison with normalized distances
        normalized_comm_radius = comm_radius / world_size
        valid_mask = neighbor_dists <= normalized_comm_radius  # (N, max_neighbours)
        # Zero out features for invalid neighbors
        neighbor_features = np.where(valid_mask[:, :, np.newaxis], neighbor_features, 0.0)
    else:
        # Global models: all neighbors valid up to actual number of agents
        # Handle case where max_neighbours > num_agents - 1 (padding)
        actual_neighbors = min(num_agents - 1, max_neighbours)
        valid_mask = np.zeros((num_agents, max_neighbours), dtype=bool)
        valid_mask[:, :actual_neighbors] = True

    # Step 9: Build complete local features
    if obs_model == "local_comm":
        # Include own neighborhood count
        within_comm = distances <= comm_radius
        np.fill_diagonal(within_comm, False)
        own_counts = within_comm.sum(axis=1).astype(np.float32)  # (N,)

        # Normalize own neighborhood count by (num_agents - 1)
        own_counts = own_counts / (num_agents - 1)

        local_features = np.stack([wall_dists, wall_bearings, own_counts], axis=1)  # (N, 3)
    else:
        local_features = np.stack([wall_dists, wall_bearings], axis=1)  # (N, 2)

    # Step 10: Flatten neighbor features and create binary mask
    neighbor_flat = neighbor_features.reshape(num_agents, max_neighbours * neighbour_feature_dim)
    mask = valid_mask.astype(np.float32)  # (N, max_neighbours)

    # Step 11: Concatenate all components into final observations
    all_obs = np.concatenate([local_features, neighbor_flat, mask], axis=1)  # (N, obs_dim)

    # Step 12: Convert to dictionary
    observations = {}
    for idx, name in enumerate(agent_names):
        observations[name] = all_obs[idx]

    return observations
