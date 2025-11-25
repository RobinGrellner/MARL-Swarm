"""
Tests for observation structure and mask correctness in RendezvousEnv.

Verifies:
- Binary mask values (0 or 1)
- Mask correctly indicates valid neighbors
- Observation dimensions match obs_layout
- Local and neighbor features are properly structured
"""

import numpy as np
import pytest
from environments.rendezvous.rendezvous_env import RendezvousEnv


class TestObservationStructure:
    """Test observation structure for different observation models."""

    @pytest.mark.parametrize(
        "obs_model", ["global_basic", "global_extended", "local_basic", "local_extended", "local_comm"]
    )
    def test_observation_dimensions(self, obs_model):
        """Test that observations match declared dimensions."""
        env = RendezvousEnv(num_agents=5, obs_model=obs_model, world_size=10.0)
        obs, _ = env.reset()

        # Check all agents have correct observation dimension
        for agent_name in env.agents:
            expected_dim = env.obs_layout["total_dim"]
            actual_dim = obs[agent_name].shape[0]
            assert actual_dim == expected_dim, (
                f"Agent {agent_name} observation dim {actual_dim} does not match expected {expected_dim}"
            )

        env.close()

    @pytest.mark.parametrize("obs_model", ["local_basic", "local_extended", "local_comm"])
    def test_local_observation_layout(self, obs_model):
        """Test that local observation models have correct layout."""
        env = RendezvousEnv(num_agents=5, obs_model=obs_model, world_size=10.0, comm_radius=5.0)
        obs, _ = env.reset()

        layout = env.obs_layout
        local_dim = layout["local_dim"]
        neigh_dim = layout["neigh_dim"]
        max_neigh = layout["max_neighbours"]
        total_dim = layout["total_dim"]

        # Verify layout consistency
        expected_total = local_dim + (max_neigh * neigh_dim) + max_neigh
        assert total_dim == expected_total, (
            f"Total dim {total_dim} != local({local_dim}) + neigh({max_neigh}*{neigh_dim}) + mask({max_neigh})"
        )

        env.close()


class TestMaskCorrectness:
    """Test that binary masks are correct."""

    def test_mask_is_binary(self):
        """Test that mask values are strictly 0 or 1."""
        env = RendezvousEnv(num_agents=6, obs_model="local_basic", comm_radius=5.0)
        obs, _ = env.reset()

        layout = env.obs_layout
        local_dim = layout["local_dim"]
        neigh_block_len = layout["max_neighbours"] * layout["neigh_dim"]
        mask_start = local_dim + neigh_block_len
        mask_end = mask_start + layout["max_neighbours"]

        for agent_name in env.agents:
            mask = obs[agent_name][mask_start:mask_end]

            # Check all mask values are 0 or 1
            assert np.all((mask == 0) | (mask == 1)), f"Agent {agent_name} has non-binary mask values: {mask}"

        env.close()

    def test_mask_indicates_valid_neighbors(self):
        """Test that mask correctly indicates which neighbors are valid."""
        # Use small comm_radius to ensure some neighbors are out of range
        env = RendezvousEnv(num_agents=5, obs_model="local_basic", world_size=20.0, comm_radius=3.0)
        obs, _ = env.reset()

        layout = env.obs_layout
        local_dim = layout["local_dim"]
        neigh_dim = layout["neigh_dim"]
        max_neigh = layout["max_neighbours"]
        neigh_block_len = max_neigh * neigh_dim
        mask_start = local_dim + neigh_block_len

        # Get agent positions to manually verify distances
        positions = env.agent_handler.positions

        for i, agent_name in enumerate(env.agents):
            agent_obs = obs[agent_name]
            mask = agent_obs[mask_start : mask_start + max_neigh]
            neigh_block = agent_obs[local_dim : local_dim + neigh_block_len].reshape(max_neigh, neigh_dim)

            # Count valid neighbors (mask == 1)
            num_valid = int(mask.sum())

            # Manually count neighbors within comm_radius
            agent_pos = positions[i]
            distances = np.linalg.norm(positions - agent_pos, axis=1)
            distances[i] = np.inf  # Exclude self

            if env.torus:
                # Handle toroidal distance
                for j in range(len(positions)):
                    if j == i:
                        continue
                    delta = positions[j] - agent_pos
                    # Wrap around
                    delta = np.where(delta > env.world_size / 2, delta - env.world_size, delta)
                    delta = np.where(delta < -env.world_size / 2, delta + env.world_size, delta)
                    distances[j] = np.linalg.norm(delta)

            within_radius = (distances <= env.comm_radius).sum()
            expected_valid = min(within_radius, max_neigh)

            assert num_valid == expected_valid, f"Agent {agent_name}: mask sum {num_valid} != expected {expected_valid}"

            # Verify that valid neighbors have non-zero distance
            for j in range(num_valid):
                if mask[j] == 1:
                    neighbor_dist = neigh_block[j, 0]  # First feature is distance
                    assert neighbor_dist > 0, f"Valid neighbor {j} has zero distance"

        env.close()

    def test_padded_slots_have_zero_mask(self):
        """Test that padded neighbor slots have mask=0."""
        # Use max_agents to force padding
        env = RendezvousEnv(
            num_agents=3,
            max_agents=10,  # This creates padding
            obs_model="local_basic",
            comm_radius=100.0,  # Large radius to see all neighbors
        )
        obs, _ = env.reset()

        layout = env.obs_layout
        local_dim = layout["local_dim"]
        max_neigh = layout["max_neighbours"]  # Should be 9 (10-1)
        neigh_block_len = max_neigh * layout["neigh_dim"]
        mask_start = local_dim + neigh_block_len

        for agent_name in env.agents:
            mask = obs[agent_name][mask_start : mask_start + max_neigh]

            # With 3 agents, each agent has 2 actual neighbors
            # Remaining slots (3 to 8) should be padded with mask=0
            num_actual_neighbors = 2
            num_valid = int(mask.sum())

            assert num_valid == num_actual_neighbors, (
                f"Agent {agent_name}: Expected {num_actual_neighbors} valid neighbors, got {num_valid}"
            )

            # Check that padded slots (beyond actual neighbors) are zero
            assert np.all(mask[num_actual_neighbors:] == 0), (
                f"Padded neighbor slots have non-zero mask: {mask[num_actual_neighbors:]}"
            )

        env.close()

    def test_mask_consistency_across_steps(self):
        """Test that mask updates correctly as agents move."""
        env = RendezvousEnv(
            num_agents=4, obs_model="local_basic", world_size=10.0, comm_radius=3.0, kinematics="single"
        )
        obs, _ = env.reset(seed=42)

        layout = env.obs_layout
        local_dim = layout["local_dim"]
        max_neigh = layout["max_neighbours"]
        neigh_block_len = max_neigh * layout["neigh_dim"]
        mask_start = local_dim + neigh_block_len

        # Take a few steps
        for _ in range(5):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, term, trunc, info = env.step(actions)

            # Verify mask is still binary after each step
            for agent_name in env.agents:
                mask = obs[agent_name][mask_start : mask_start + max_neigh]
                assert np.all((mask == 0) | (mask == 1)), f"Mask became non-binary after step: {mask}"

        env.close()


class TestMaskWithDifferentModels:
    """Test masks work correctly across different observation models."""

    @pytest.mark.parametrize(
        "obs_model,expected_neigh_dim",
        [
            ("local_basic", 2),  # distance, bearing
            ("local_extended", 3),  # distance, bearing, relative orientation
            ("local_comm", 4),  # distance, bearing, relative orientation, neighbor count
        ],
    )
    def test_mask_with_different_feature_dims(self, obs_model, expected_neigh_dim):
        """Test mask works with different neighbor feature dimensions."""
        env = RendezvousEnv(num_agents=5, obs_model=obs_model, comm_radius=5.0)
        obs, _ = env.reset()

        layout = env.obs_layout
        assert layout["neigh_dim"] == expected_neigh_dim, (
            f"Expected neighbor dim {expected_neigh_dim}, got {layout['neigh_dim']}"
        )

        local_dim = layout["local_dim"]
        max_neigh = layout["max_neighbours"]
        neigh_block_len = max_neigh * expected_neigh_dim
        mask_start = local_dim + neigh_block_len

        for agent_name in env.agents:
            mask = obs[agent_name][mask_start : mask_start + max_neigh]
            assert np.all((mask == 0) | (mask == 1)), f"Model {obs_model}: mask is not binary"

        env.close()

    def test_global_model_all_neighbors_valid(self):
        """Test that global models have all neighbors marked as valid."""
        env = RendezvousEnv(num_agents=5, obs_model="global_basic")
        obs, _ = env.reset()

        layout = env.obs_layout
        local_dim = layout["local_dim"]
        max_neigh = layout["max_neighbours"]  # Should be num_agents - 1 = 4
        neigh_block_len = max_neigh * layout["neigh_dim"]
        mask_start = local_dim + neigh_block_len

        for agent_name in env.agents:
            mask = obs[agent_name][mask_start : mask_start + max_neigh]

            # Global models: all neighbors should be valid (no comm radius limit)
            expected_valid = env.num_agents - 1
            num_valid = int(mask.sum())

            assert num_valid == expected_valid, (
                f"Global model should have all {expected_valid} neighbors valid, got {num_valid}"
            )

        env.close()


class TestLocalCommObservation:
    """Test local_comm observation model specific features."""

    def test_local_comm_has_neighborhood_count(self):
        """Test that local_comm includes own neighborhood count in local features."""
        env = RendezvousEnv(num_agents=5, obs_model="local_comm", comm_radius=5.0)
        obs, _ = env.reset()

        layout = env.obs_layout
        assert layout["local_dim"] == 3, "local_comm should have 3 local features"

        # Local features: [wall_dist, wall_bearing, own_neighbor_count]
        for agent_name in env.agents:
            local_features = obs[agent_name][:3]
            own_neighbor_count = local_features[2]

            # Neighbor count should be non-negative integer value
            assert own_neighbor_count >= 0, f"Own neighbor count is negative: {own_neighbor_count}"
            assert own_neighbor_count <= env.num_agents - 1, (
                f"Own neighbor count {own_neighbor_count} exceeds max possible"
            )

        env.close()

    def test_neighbor_counts_in_features(self):
        """Test that neighbor's neighborhood counts are in neighbor features."""
        env = RendezvousEnv(num_agents=5, obs_model="local_comm", comm_radius=5.0)
        obs, _ = env.reset()

        layout = env.obs_layout
        local_dim = layout["local_dim"]
        neigh_dim = layout["neigh_dim"]  # Should be 4: dist, bearing, ori, count
        max_neigh = layout["max_neighbours"]

        assert neigh_dim == 4, "local_comm neighbor features should be 4-dimensional"

        for agent_name in env.agents:
            agent_obs = obs[agent_name]
            neigh_block = agent_obs[local_dim : local_dim + max_neigh * neigh_dim]
            neigh_features = neigh_block.reshape(max_neigh, neigh_dim)

            # Fourth feature (index 3) is neighbor's neighborhood count
            for i in range(max_neigh):
                neighbor_count = neigh_features[i, 3]
                assert neighbor_count >= 0, f"Neighbor {i} has negative count: {neighbor_count}"

        env.close()


class TestTorusObservations:
    """Test observations work correctly in toroidal worlds."""

    def test_torus_wall_distance(self):
        """Test wall distance in toroidal world is normalized to 1.0."""
        env = RendezvousEnv(num_agents=4, obs_model="local_basic", world_size=10.0, torus=True)
        obs, _ = env.reset()

        # In torus, wall distance is world_size, normalized to 1.0
        for agent_name in env.agents:
            wall_dist = obs[agent_name][0]
            assert wall_dist == 1.0, f"Torus wall distance (normalized) should be 1.0, got {wall_dist}"

        env.close()

    def test_torus_neighbor_distances(self):
        """Test neighbor distances consider toroidal wrapping."""
        env = RendezvousEnv(
            num_agents=4,
            obs_model="local_basic",
            world_size=10.0,
            torus=True,
            comm_radius=100.0,  # See all neighbors
        )

        # Place agents at opposite corners to test wrapping
        env.reset(seed=123)

        # Manually set positions at corners
        positions = np.array([[0.5, 0.5], [9.5, 9.5], [0.5, 9.5], [9.5, 0.5]])
        env.agent_handler.positions = positions

        # Get observations
        obs = env._get_observations()

        layout = env.obs_layout
        local_dim = layout["local_dim"]
        neigh_dim = layout["neigh_dim"]

        # Check that distances are wrapped (should be ~1.41 diagonal, not ~13.4)
        for agent_name in env.agents:
            agent_obs = obs[agent_name]
            neigh_block = agent_obs[local_dim : local_dim + 3 * neigh_dim]
            neighbor_dists = neigh_block.reshape(3, neigh_dim)[:, 0]

            # In toroidal world with corners, diagonal distance is sqrt(2) ≈ 1.41
            # without wrapping it would be ~13.4
            for dist in neighbor_dists:
                assert dist < 3.0, f"Distance {dist} too large, torus wrapping may not be working"

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
