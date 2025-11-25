"""
Standalone test runner for observation and mask tests.
Run with: python test_observations_standalone.py
"""

import numpy as np
from environments.rendezvous.rendezvous_env import RendezvousEnv


def test_mask_is_binary():
    """Test that mask values are strictly 0 or 1."""
    print("Testing: Mask is binary...")
    env = RendezvousEnv(num_agents=6, world_size=10.0, obs_model="local_basic", comm_radius=5.0)
    obs, _ = env.reset()

    layout = env.obs_layout
    local_dim = layout["local_dim"]
    neigh_block_len = layout["max_neighbours"] * layout["neigh_dim"]
    mask_start = local_dim + neigh_block_len
    mask_end = mask_start + layout["max_neighbours"]

    for agent_name in env.agents:
        mask = obs[agent_name][mask_start:mask_end]
        assert np.all((mask == 0) | (mask == 1)), f"Agent {agent_name} has non-binary mask: {mask}"

    env.close()
    print("✓ PASSED: All masks are binary (0 or 1)")


def test_padded_slots_have_zero_mask():
    """Test that padded neighbor slots have mask=0."""
    print("\nTesting: Padded slots have zero mask...")
    # Use global model to ensure all neighbors are visible
    # max_agents creates padding beyond actual neighbors
    env = RendezvousEnv(
        num_agents=3,
        world_size=10.0,
        max_agents=10,
        obs_model="global_basic",  # Global sees all neighbors regardless of distance
    )
    obs, _ = env.reset()

    layout = env.obs_layout
    local_dim = layout["local_dim"]
    max_neigh = layout["max_neighbours"]  # Should be 9 (10-1)
    neigh_block_len = max_neigh * layout["neigh_dim"]
    mask_start = local_dim + neigh_block_len

    for agent_name in env.agents:
        mask = obs[agent_name][mask_start : mask_start + max_neigh]
        num_actual_neighbors = 2  # With 3 agents, each has 2 neighbors
        num_valid = int(mask.sum())

        assert num_valid == num_actual_neighbors, (
            f"Agent {agent_name}: Expected {num_actual_neighbors} valid neighbors, got {num_valid}"
        )
        # Slots beyond actual neighbors should be padded with mask=0
        assert np.all(mask[num_actual_neighbors:] == 0), (
            f"Padded slots have non-zero mask: {mask[num_actual_neighbors:]}"
        )

    env.close()
    print(f"✓ PASSED: Padded slots correctly have mask=0 (max_agents=10, actual=3)")


def test_observation_dimensions():
    """Test that observations match declared dimensions."""
    print("\nTesting: Observation dimensions...")

    obs_models = ["global_basic", "global_extended", "local_basic", "local_extended", "local_comm"]

    for obs_model in obs_models:
        env = RendezvousEnv(num_agents=5, obs_model=obs_model, world_size=10.0)
        obs, _ = env.reset()

        for agent_name in env.agents:
            expected_dim = env.obs_layout["total_dim"]
            actual_dim = obs[agent_name].shape[0]
            assert actual_dim == expected_dim, (
                f"Model {obs_model}, agent {agent_name}: dim {actual_dim} != expected {expected_dim}"
            )

        env.close()
        print(f"  ✓ {obs_model}: dimension {expected_dim} correct")

    print("✓ PASSED: All observation dimensions match obs_layout")


def test_mask_consistency_across_steps():
    """Test that mask updates correctly as agents move."""
    print("\nTesting: Mask consistency across steps...")
    env = RendezvousEnv(num_agents=4, obs_model="local_basic", world_size=10.0, comm_radius=3.0, kinematics="single")
    obs, _ = env.reset(seed=42)

    layout = env.obs_layout
    local_dim = layout["local_dim"]
    max_neigh = layout["max_neighbours"]
    neigh_block_len = max_neigh * layout["neigh_dim"]
    mask_start = local_dim + neigh_block_len

    for step in range(10):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, term, trunc, info = env.step(actions)

        for agent_name in env.agents:
            mask = obs[agent_name][mask_start : mask_start + max_neigh]
            assert np.all((mask == 0) | (mask == 1)), f"Step {step}: Mask became non-binary: {mask}"

    env.close()
    print("✓ PASSED: Masks remain binary across 10 steps")


def test_global_model_all_neighbors_valid():
    """Test that global models have all neighbors marked as valid."""
    print("\nTesting: Global model neighbors...")
    env = RendezvousEnv(num_agents=5, world_size=10.0, obs_model="global_basic")
    obs, _ = env.reset()

    layout = env.obs_layout
    local_dim = layout["local_dim"]
    max_neigh = layout["max_neighbours"]
    neigh_block_len = max_neigh * layout["neigh_dim"]
    mask_start = local_dim + neigh_block_len

    for agent_name in env.agents:
        mask = obs[agent_name][mask_start : mask_start + max_neigh]
        expected_valid = env.num_agents - 1
        num_valid = int(mask.sum())

        assert num_valid == expected_valid, (
            f"Global model should have all {expected_valid} neighbors valid, got {num_valid}"
        )

    env.close()
    print(f"✓ PASSED: Global model has all {env.num_agents - 1} neighbors valid (no comm radius limit)")


def test_local_comm_neighborhood_count():
    """Test that local_comm includes neighborhood count."""
    print("\nTesting: Local comm neighborhood count...")
    env = RendezvousEnv(num_agents=5, world_size=10.0, obs_model="local_comm", comm_radius=5.0)
    obs, _ = env.reset()

    layout = env.obs_layout
    assert layout["local_dim"] == 3, "local_comm should have 3 local features"
    assert layout["neigh_dim"] == 4, "local_comm neighbor features should be 4D"

    for agent_name in env.agents:
        local_features = obs[agent_name][:3]
        own_neighbor_count = local_features[2]

        assert own_neighbor_count >= 0 and own_neighbor_count <= env.num_agents - 1, (
            f"Own neighbor count {own_neighbor_count} out of valid range"
        )

    env.close()
    print("✓ PASSED: Local comm includes valid neighborhood counts")


def test_torus_wall_distance():
    """Test wall distance in toroidal world is normalized to 1.0."""
    print("\nTesting: Torus wall distance...")
    env = RendezvousEnv(num_agents=4, obs_model="local_basic", world_size=10.0, torus=True)
    obs, _ = env.reset()

    for agent_name in env.agents:
        wall_dist = obs[agent_name][0]
        # Wall distance is normalized, so world_size / world_size = 1.0
        assert wall_dist == 1.0, f"Torus wall distance (normalized) should be 1.0, got {wall_dist}"

    env.close()
    print(f"✓ PASSED: Torus wall distance is {env.world_size} (infinite wall)")


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("OBSERVATION AND MASK TESTS")
    print("=" * 70)

    tests = [
        test_mask_is_binary,
        test_padded_slots_have_zero_mask,
        test_observation_dimensions,
        test_mask_consistency_across_steps,
        test_global_model_all_neighbors_valid,
        test_local_comm_neighborhood_count,
        test_torus_wall_distance,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"\n❌ FAILED: {test_func.__name__}")
            print(f"   Error: {e}")
            failed += 1
        except Exception as e:
            print(f"\n❌ ERROR: {test_func.__name__}")
            print(f"   Error: {e}")
            failed += 1

    print("\n" + "=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
