"""
Test reward calculation for RendezvousEnv.

Tests cover:
- Basic reward calculation
- Distance clipping (d_c parameter)
- Action penalty
- Torus wrapping effects on rewards
- Edge cases (all agents at same position, maximum distance)
- Reward normalization properties
"""

import pytest
import numpy as np
from environments.rendezvous.rendezvous_env import RendezvousEnv


class TestRendezvousRewardCalculation:
    """Test basic reward calculation for rendezvous environment."""

    @pytest.fixture
    def env(self):
        """Create a basic rendezvous environment."""
        return RendezvousEnv(
            num_agents=3,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
            torus=False,
        )

    def test_reward_all_agents_same_position(self, env):
        """Test reward when all agents are at the same position."""
        env.reset(seed=42)

        # Place all agents at same position
        env.agent_handler.positions = np.array([[5.0, 5.0], [5.0, 5.0], [5.0, 5.0]], dtype=np.float32)

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(3)}
        rewards = env._calculate_rewards(actions)

        # All distances are 0, so distance reward should be 0
        # No actions, so action penalty is 0
        # Total reward should be 0
        for agent_name, reward in rewards.items():
            assert reward == 0.0, f"{agent_name} should have reward 0.0 when all agents at same position"

    def test_reward_two_agents_known_distance(self):
        """Test reward calculation with known distances."""
        env = RendezvousEnv(
            num_agents=2,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Place agents at known positions
        env.agent_handler.positions = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)

        # Distance = sqrt(9 + 16) = 5.0
        # alpha = -1 / (n*(n-1)/2 * dc) = -1 / (1 * 10.0) = -0.1
        # reward_distance = -0.1 * 5.0 = -0.5

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(2)}
        rewards = env._calculate_rewards(actions)

        expected_reward = -0.5
        for reward in rewards.values():
            assert np.isclose(reward, expected_reward, atol=1e-5), \
                f"Expected reward {expected_reward}, got {reward}"

    def test_reward_distance_clipping(self):
        """Test that distances are clipped to d_c."""
        env = RendezvousEnv(
            num_agents=2,
            world_size=100.0,
            max_steps=100,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Place agents far apart (distance > world_size = d_c)
        env.agent_handler.positions = np.array([[0.0, 0.0], [200.0, 0.0]], dtype=np.float32)

        # Distance = 200.0, but should be clipped to d_c = 100.0
        # alpha = -1 / (1 * 100.0) = -0.01
        # reward_distance = -0.01 * 100.0 = -1.0

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(2)}
        rewards = env._calculate_rewards(actions)

        expected_reward = -1.0
        for reward in rewards.values():
            assert np.isclose(reward, expected_reward, atol=1e-5), \
                f"Distance should be clipped to d_c"

    def test_action_penalty(self):
        """Test that action penalty is correctly applied."""
        env = RendezvousEnv(
            num_agents=2,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Place agents at same position (no distance reward)
        env.agent_handler.positions = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)

        # Apply actions with known magnitude
        actions = {
            "agent_0": np.array([0.6, 0.8], dtype=np.float32),  # Magnitude = 1.0
            "agent_1": np.array([0.3, 0.4], dtype=np.float32),  # Magnitude = 0.5
        }

        # beta = -1e-3
        # reward_action = -1e-3 * (1.0 + 0.5) = -0.0015

        rewards = env._calculate_rewards(actions)

        expected_reward = -0.0015
        for reward in rewards.values():
            assert np.isclose(reward, expected_reward, atol=1e-6), \
                f"Expected action penalty {expected_reward}, got {reward}"

    def test_combined_reward(self):
        """Test combined distance and action rewards."""
        env = RendezvousEnv(
            num_agents=2,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Place agents at distance 5.0
        env.agent_handler.positions = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)

        # Actions with total magnitude 1.5
        actions = {
            "agent_0": np.array([0.6, 0.8], dtype=np.float32),  # Magnitude = 1.0
            "agent_1": np.array([0.3, 0.4], dtype=np.float32),  # Magnitude = 0.5
        }

        # reward_distance = -0.1 * 5.0 = -0.5
        # reward_action = -1e-3 * 1.5 = -0.0015
        # total = -0.5015

        rewards = env._calculate_rewards(actions)

        expected_reward = -0.5015
        for reward in rewards.values():
            assert np.isclose(reward, expected_reward, atol=1e-6), \
                f"Expected combined reward {expected_reward}, got {reward}"

    def test_reward_shared_among_all_agents(self):
        """Test that all agents receive the same reward (global reward)."""
        env = RendezvousEnv(
            num_agents=4,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        env.agent_handler.positions = np.random.uniform(0, 10, (4, 2)).astype(np.float32)
        actions = {f"agent_{i}": np.random.uniform(-1, 1, 2).astype(np.float32) for i in range(4)}

        rewards = env._calculate_rewards(actions)

        # All rewards should be identical (global reward)
        reward_values = list(rewards.values())
        for i in range(1, len(reward_values)):
            assert reward_values[i] == reward_values[0], \
                "All agents should receive the same global reward"


class TestRendezvousRewardWithTorus:
    """Test reward calculation with toroidal world wrapping."""

    def test_torus_wrapping_affects_distances(self):
        """Test that torus wrapping correctly computes shortest distances."""
        env = RendezvousEnv(
            num_agents=2,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
            torus=True,
        )
        env.reset(seed=42)

        # Place agents near opposite edges
        env.agent_handler.positions = np.array([[0.5, 5.0], [9.5, 5.0]], dtype=np.float32)

        # Direct distance = 9.0
        # Wrapped distance = 1.0 (wrapping around)
        # Should use wrapped distance

        # alpha = -0.1
        # reward = -0.1 * 1.0 = -0.1

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(2)}
        rewards = env._calculate_rewards(actions)

        expected_reward = -0.1
        for reward in rewards.values():
            assert np.isclose(reward, expected_reward, atol=1e-5), \
                f"Torus wrapping should give shortest distance"

    def test_torus_wrapping_diagonal(self):
        """Test torus wrapping with diagonal distances."""
        env = RendezvousEnv(
            num_agents=2,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
            torus=True,
        )
        env.reset(seed=42)

        # Place agents at opposite corners
        env.agent_handler.positions = np.array([[0.5, 0.5], [9.5, 9.5]], dtype=np.float32)

        # Direct distance = sqrt(81 + 81) = sqrt(162) ≈ 12.73
        # Wrapped distance = sqrt(1 + 1) = sqrt(2) ≈ 1.414

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(2)}
        rewards = env._calculate_rewards(actions)

        # reward = -0.1 * sqrt(2)
        expected_reward = -0.1 * np.sqrt(2)
        for reward in rewards.values():
            assert np.isclose(reward, expected_reward, atol=1e-5), \
                f"Torus wrapping should handle diagonal distances"


class TestRendezvousRewardNormalization:
    """Test reward normalization properties."""

    def test_maximum_reward_is_zero(self):
        """Test that maximum achievable reward is 0 (all agents at same position, no actions)."""
        env = RendezvousEnv(
            num_agents=5,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        env.agent_handler.positions = np.zeros((5, 2), dtype=np.float32)
        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(5)}

        rewards = env._calculate_rewards(actions)

        for reward in rewards.values():
            assert reward == 0.0

    def test_worst_case_reward_approximately_minus_one(self):
        """Test that worst-case reward is approximately -1."""
        env = RendezvousEnv(
            num_agents=3,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Place agents at maximum distance (all pairwise distances = d_c)
        # This gives sum of clipped distances = n*(n-1)/2 * d_c
        env.agent_handler.positions = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0]
        ], dtype=np.float32)

        # All pairwise distances are >= d_c (10.0)
        # Sum of clipped = 3 * 10.0 = 30.0
        # alpha = -1 / (3 * 10.0) = -1/30
        # reward = -1/30 * 30.0 = -1.0

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(3)}
        rewards = env._calculate_rewards(actions)

        expected_reward = -1.0
        for reward in rewards.values():
            assert np.isclose(reward, expected_reward, atol=1e-5), \
                f"Worst case reward should be approximately -1.0"

    def test_reward_scale_invariant_to_world_size(self):
        """Test that reward normalization makes it scale-invariant."""
        num_agents = 4

        # Test with different world sizes
        for world_size in [10.0, 50.0, 100.0]:
            env = RendezvousEnv(
                num_agents=num_agents,
                world_size=world_size,
                max_steps=100,
                obs_model="global_basic",
            )
            env.reset(seed=42)

            # Place agents at maximum distances (scaled to world_size)
            env.agent_handler.positions = np.array([
                [0.0, 0.0],
                [world_size, 0.0],
                [0.0, world_size],
                [world_size, world_size]
            ], dtype=np.float32)

            actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(num_agents)}
            rewards = env._calculate_rewards(actions)

            # Should be approximately -1.0 regardless of world_size
            for reward in rewards.values():
                assert np.isclose(reward, -1.0, atol=0.1), \
                    f"Reward normalization should be scale-invariant (world_size={world_size})"


class TestRendezvousRewardMultipleAgents:
    """Test reward calculation with varying numbers of agents."""

    @pytest.mark.parametrize("num_agents", [2, 5, 10, 20])
    def test_reward_calculation_scales_with_agents(self, num_agents):
        """Test that reward calculation works correctly with different agent counts."""
        env = RendezvousEnv(
            num_agents=num_agents,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Random positions
        env.agent_handler.positions = np.random.uniform(0, 10, (num_agents, 2)).astype(np.float32)
        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(num_agents)}

        rewards = env._calculate_rewards(actions)

        # Check that:
        # 1. All agents have same reward
        # 2. Reward is in reasonable range [-1.0, 0.0] (ignoring action penalty)
        reward_values = list(rewards.values())
        assert all(r == reward_values[0] for r in reward_values), "All agents should have same reward"
        assert -1.5 <= reward_values[0] <= 0.0, f"Reward should be in reasonable range, got {reward_values[0]}"

    def test_pairwise_distance_count(self):
        """Test that correct number of pairwise distances are computed."""
        num_agents = 5
        env = RendezvousEnv(
            num_agents=num_agents,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Place all agents at unit distance from each other in a line
        env.agent_handler.positions = np.array([[i * 1.0, 0.0] for i in range(num_agents)], dtype=np.float32)

        # Number of pairwise distances = n*(n-1)/2 = 5*4/2 = 10
        # But distances vary: (0,1)=1, (0,2)=2, ..., (3,4)=1
        # Sum of distances: 1+2+3+4 + 1+2+3 + 1+2 + 1 = 10+6+3+1 = 20

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(num_agents)}
        rewards = env._calculate_rewards(actions)

        # alpha = -1 / (10 * 10.0) = -0.01
        # reward = -0.01 * 20.0 = -0.2
        expected_reward = -0.2
        for reward in rewards.values():
            assert np.isclose(reward, expected_reward, atol=1e-5)


class TestRendezvousRewardEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_none_actions(self):
        """Test that reward can be calculated with None actions."""
        env = RendezvousEnv(
            num_agents=2,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        env.agent_handler.positions = np.array([[0.0, 0.0], [3.0, 4.0]], dtype=np.float32)

        # Pass None for actions
        rewards = env._calculate_rewards(None)

        # Should only have distance component, no action penalty
        expected_reward = -0.5  # Same as test_reward_two_agents_known_distance
        for reward in rewards.values():
            assert np.isclose(reward, expected_reward, atol=1e-5)

    def test_negative_action_values(self):
        """Test that negative action values contribute to penalty correctly."""
        env = RendezvousEnv(
            num_agents=2,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        env.agent_handler.positions = np.array([[0.0, 0.0], [0.0, 0.0]], dtype=np.float32)

        # Negative actions should still contribute magnitude
        actions = {
            "agent_0": np.array([-0.6, -0.8], dtype=np.float32),  # Magnitude = 1.0
            "agent_1": np.array([-0.3, -0.4], dtype=np.float32),  # Magnitude = 0.5
        }

        rewards = env._calculate_rewards(actions)

        expected_reward = -0.0015  # Same as test_action_penalty
        for reward in rewards.values():
            assert np.isclose(reward, expected_reward, atol=1e-6)

    def test_very_small_distances(self):
        """Test reward calculation with very small distances."""
        env = RendezvousEnv(
            num_agents=2,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Place agents very close together
        env.agent_handler.positions = np.array([[0.0, 0.0], [1e-6, 1e-6]], dtype=np.float32)

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(2)}
        rewards = env._calculate_rewards(actions)

        # Reward should be very close to 0
        for reward in rewards.values():
            assert np.isclose(reward, 0.0, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
