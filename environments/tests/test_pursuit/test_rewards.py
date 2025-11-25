"""
Test reward calculation for PursuitEvasionEnv.

Tests cover:
- Time penalty
- Distance-based rewards
- Capture bonus
- Individual vs shared rewards
- Edge cases
"""

import pytest
import numpy as np
from environments.pursuit.pursuit_evasion_env import PursuitEvasionEnv


class TestPursuitEvasionRewardCalculation:
    """Test basic reward calculation for pursuit-evasion environment."""

    @pytest.fixture
    def env(self):
        """Create a basic pursuit-evasion environment."""
        return PursuitEvasionEnv(
            num_pursuers=3,
            world_size=10.0,
            max_steps=100,
            capture_radius=0.5,
            obs_model="global_basic",
        )

    def test_time_penalty_only(self, env):
        """Test that time penalty is applied each step."""
        env.reset(seed=42)

        # Place evader far from all pursuers
        env.evader_pos = np.array([5.0, 5.0], dtype=np.float32)
        env.agent_handler.positions = np.array([
            [0.0, 0.0],
            [10.0, 0.0],
            [0.0, 10.0]
        ], dtype=np.float32)

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(3)}
        rewards = env._calculate_rewards(actions)

        # Each agent should have time penalty (-0.01) + distance penalty
        # Distance from [0,0] to [5,5] = sqrt(50) ≈ 7.07
        # Distance penalty = -0.1 * 7.07 / 10.0 ≈ -0.0707
        # Total ≈ -0.01 + -0.0707 = -0.0807

        expected_reward_agent0 = -0.01 + (-0.1 * np.sqrt(50) / 10.0)
        assert np.isclose(rewards["agent_0"], expected_reward_agent0, atol=1e-5), \
            f"Expected reward {expected_reward_agent0}, got {rewards['agent_0']}"

    def test_distance_based_reward(self):
        """Test that distance-based rewards are correctly calculated."""
        env = PursuitEvasionEnv(
            num_pursuers=2,
            world_size=10.0,
            max_steps=100,
            capture_radius=0.5,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Place evader at origin
        env.evader_pos = np.array([0.0, 0.0], dtype=np.float32)

        # Place pursuers at known distances
        env.agent_handler.positions = np.array([
            [3.0, 4.0],    # Distance = 5.0
            [6.0, 8.0]     # Distance = 10.0
        ], dtype=np.float32)

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(2)}
        rewards = env._calculate_rewards(actions)

        # Agent 0: -0.01 + (-0.1 * 5.0 / 10.0) = -0.01 + -0.05 = -0.06
        # Agent 1: -0.01 + (-0.1 * 10.0 / 10.0) = -0.01 + -0.1 = -0.11

        assert np.isclose(rewards["agent_0"], -0.06, atol=1e-5)
        assert np.isclose(rewards["agent_1"], -0.11, atol=1e-5)

    def test_capture_bonus(self):
        """Test that capture bonus is awarded when evader is caught."""
        env = PursuitEvasionEnv(
            num_pursuers=2,
            world_size=10.0,
            max_steps=100,
            capture_radius=0.5,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Place evader at origin
        env.evader_pos = np.array([0.0, 0.0], dtype=np.float32)

        # Place one pursuer within capture radius, one far
        env.agent_handler.positions = np.array([
            [0.3, 0.0],    # Distance = 0.3 < capture_radius
            [5.0, 5.0]     # Distance = sqrt(50) > capture_radius
        ], dtype=np.float32)

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(2)}
        rewards = env._calculate_rewards(actions)

        # Both agents get capture bonus (10.0) because one caught the evader
        # Agent 0: -0.01 + (-0.1 * 0.3 / 10.0) + 10.0 = 9.987
        # Agent 1: -0.01 + (-0.1 * sqrt(50) / 10.0) + 10.0 ≈ 9.919

        assert rewards["agent_0"] > 9.9, "Agent 0 should receive capture bonus"
        assert rewards["agent_1"] > 9.9, "Agent 1 should receive capture bonus"

    def test_capture_at_exact_radius(self):
        """Test capture detection at exact boundary."""
        env = PursuitEvasionEnv(
            num_pursuers=1,
            world_size=10.0,
            max_steps=100,
            capture_radius=1.0,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Test just inside capture radius
        env.evader_pos = np.array([0.0, 0.0], dtype=np.float32)
        env.agent_handler.positions = np.array([[0.99, 0.0]], dtype=np.float32)

        actions = {"agent_0": np.array([0.0, 0.0], dtype=np.float32)}
        rewards = env._calculate_rewards(actions)
        assert rewards["agent_0"] > 9.0, "Should capture when distance < radius"

        # Test exactly at capture radius (should not capture - threshold is exclusive)
        env.agent_handler.positions = np.array([[1.0, 0.0]], dtype=np.float32)
        rewards = env._calculate_rewards(actions)
        assert rewards["agent_0"] < 1.0, "Should not capture when distance == radius"

        # Test just outside capture radius
        env.agent_handler.positions = np.array([[1.01, 0.0]], dtype=np.float32)
        rewards = env._calculate_rewards(actions)
        assert rewards["agent_0"] < 1.0, "Should not capture when distance > radius"

    def test_individual_rewards_differ_by_distance(self):
        """Test that each pursuer gets individual reward based on their distance."""
        env = PursuitEvasionEnv(
            num_pursuers=3,
            world_size=10.0,
            max_steps=100,
            capture_radius=0.5,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        env.evader_pos = np.array([0.0, 0.0], dtype=np.float32)

        # Place pursuers at different distances
        env.agent_handler.positions = np.array([
            [1.0, 0.0],    # Distance = 1.0
            [2.0, 0.0],    # Distance = 2.0
            [3.0, 0.0]     # Distance = 3.0
        ], dtype=np.float32)

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(3)}
        rewards = env._calculate_rewards(actions)

        # Closer agents should have better (less negative) rewards
        assert rewards["agent_0"] > rewards["agent_1"], "Closer agent should have better reward"
        assert rewards["agent_1"] > rewards["agent_2"], "Closer agent should have better reward"


class TestPursuitEvasionTermination:
    """Test termination conditions for pursuit-evasion environment."""

    def test_termination_on_capture(self):
        """Test that episode terminates when evader is captured."""
        env = PursuitEvasionEnv(
            num_pursuers=2,
            world_size=10.0,
            max_steps=100,
            capture_radius=0.5,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Place pursuer within capture radius
        env.evader_pos = np.array([5.0, 5.0], dtype=np.float32)
        env.agent_handler.positions = np.array([
            [5.2, 5.0],    # Distance = 0.2 < 0.5
            [0.0, 0.0]
        ], dtype=np.float32)

        terminations = env._check_terminations()

        # All agents should be terminated when evader is captured
        for agent_name, terminated in terminations.items():
            assert terminated, f"{agent_name} should be terminated when evader captured"

    def test_no_termination_when_not_captured(self):
        """Test that episode doesn't terminate when evader not captured."""
        env = PursuitEvasionEnv(
            num_pursuers=2,
            world_size=10.0,
            max_steps=100,
            capture_radius=0.5,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Place all pursuers outside capture radius
        env.evader_pos = np.array([5.0, 5.0], dtype=np.float32)
        env.agent_handler.positions = np.array([
            [0.0, 0.0],    # Distance > 0.5
            [10.0, 10.0]   # Distance > 0.5
        ], dtype=np.float32)

        terminations = env._check_terminations()

        # No agent should be terminated
        for agent_name, terminated in terminations.items():
            assert not terminated, f"{agent_name} should not be terminated when evader not captured"


class TestPursuitEvasionRewardNormalization:
    """Test reward normalization and range."""

    def test_reward_range_without_capture(self):
        """Test that rewards are in reasonable range without capture."""
        env = PursuitEvasionEnv(
            num_pursuers=5,
            world_size=10.0,
            max_steps=100,
            capture_radius=0.5,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Random positions
        env.evader_pos = np.random.uniform(0, 10, 2).astype(np.float32)
        env.agent_handler.positions = np.random.uniform(0, 10, (5, 2)).astype(np.float32)

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(5)}
        rewards = env._calculate_rewards(actions)

        # Without capture, rewards should be negative (time penalty + distance penalty)
        for reward in rewards.values():
            assert -1.0 <= reward <= 0.0, f"Reward without capture should be in [-1, 0], got {reward}"

    def test_reward_range_with_capture(self):
        """Test that capture bonus significantly outweighs penalties."""
        env = PursuitEvasionEnv(
            num_pursuers=2,
            world_size=10.0,
            max_steps=100,
            capture_radius=0.5,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Place pursuer at capture
        env.evader_pos = np.array([0.0, 0.0], dtype=np.float32)
        env.agent_handler.positions = np.array([
            [0.1, 0.0],    # Very close, captured
            [0.1, 0.0]
        ], dtype=np.float32)

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(2)}
        rewards = env._calculate_rewards(actions)

        # With capture, rewards should be strongly positive (10.0 bonus dominates)
        for reward in rewards.values():
            assert reward > 9.0, f"Reward with capture should be > 9.0, got {reward}"

    def test_distance_reward_normalization(self):
        """Test that distance rewards are normalized by world_size."""
        # Test with different world sizes
        for world_size in [10.0, 50.0, 100.0]:
            env = PursuitEvasionEnv(
                num_pursuers=1,
                world_size=world_size,
                max_steps=100,
                capture_radius=0.5,
                obs_model="global_basic",
            )
            env.reset(seed=42)

            # Place pursuer at maximum distance (diagonal of world)
            env.evader_pos = np.array([0.0, 0.0], dtype=np.float32)
            env.agent_handler.positions = np.array([[world_size, world_size]], dtype=np.float32)

            actions = {"agent_0": np.array([0.0, 0.0], dtype=np.float32)}
            rewards = env._calculate_rewards(actions)

            # Distance penalty = -0.1 * sqrt(2 * world_size^2) / world_size
            # = -0.1 * sqrt(2) ≈ -0.1414
            # Total reward ≈ -0.01 + -0.1414 = -0.1514

            expected_distance_penalty = -0.1 * np.sqrt(2)
            expected_reward = -0.01 + expected_distance_penalty

            assert np.isclose(rewards["agent_0"], expected_reward, atol=1e-3), \
                f"Distance reward should be normalized (world_size={world_size})"


class TestPursuitEvasionMultiplePursuers:
    """Test reward calculation with varying numbers of pursuers."""

    @pytest.mark.parametrize("num_pursuers", [2, 5, 10, 20])
    def test_rewards_scale_with_pursuers(self, num_pursuers):
        """Test that reward calculation works with different pursuer counts."""
        env = PursuitEvasionEnv(
            num_pursuers=num_pursuers,
            world_size=10.0,
            max_steps=100,
            capture_radius=0.5,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Random positions
        env.evader_pos = np.random.uniform(0, 10, 2).astype(np.float32)
        env.agent_handler.positions = np.random.uniform(0, 10, (num_pursuers, 2)).astype(np.float32)

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(num_pursuers)}
        rewards = env._calculate_rewards(actions)

        # Check that all pursuers get rewards
        assert len(rewards) == num_pursuers

        # Rewards should be in reasonable range
        for reward in rewards.values():
            assert -1.0 <= reward <= 10.0, f"Reward should be reasonable, got {reward}"

    def test_closest_pursuer_detection(self):
        """Test that capture is detected when any pursuer is close enough."""
        env = PursuitEvasionEnv(
            num_pursuers=5,
            world_size=10.0,
            max_steps=100,
            capture_radius=0.5,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        env.evader_pos = np.array([5.0, 5.0], dtype=np.float32)

        # Only one pursuer close enough
        env.agent_handler.positions = np.array([
            [0.0, 0.0],     # Far
            [10.0, 10.0],   # Far
            [5.2, 5.0],     # Close! Distance = 0.2 < 0.5
            [0.0, 10.0],    # Far
            [10.0, 0.0]     # Far
        ], dtype=np.float32)

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(5)}
        rewards = env._calculate_rewards(actions)

        # All pursuers should get capture bonus (cooperative task)
        for reward in rewards.values():
            assert reward > 9.0, "All pursuers should benefit from capture"


class TestPursuitEvasionEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_evader_at_pursuer_exact_position(self):
        """Test when evader is at exact same position as pursuer."""
        env = PursuitEvasionEnv(
            num_pursuers=1,
            world_size=10.0,
            max_steps=100,
            capture_radius=0.5,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Same position (distance = 0)
        env.evader_pos = np.array([5.0, 5.0], dtype=np.float32)
        env.agent_handler.positions = np.array([[5.0, 5.0]], dtype=np.float32)

        actions = {"agent_0": np.array([0.0, 0.0], dtype=np.float32)}
        rewards = env._calculate_rewards(actions)

        # Should definitely be captured (distance 0 < capture_radius)
        # Reward = -0.01 + 0.0 (distance reward) + 10.0 = 9.99
        assert rewards["agent_0"] > 9.9

    def test_all_pursuers_equidistant(self):
        """Test when all pursuers are equidistant from evader."""
        env = PursuitEvasionEnv(
            num_pursuers=4,
            world_size=10.0,
            max_steps=100,
            capture_radius=0.5,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Place evader at center, pursuers in circle around it
        env.evader_pos = np.array([5.0, 5.0], dtype=np.float32)
        radius = 2.0
        angles = [0, np.pi/2, np.pi, 3*np.pi/2]
        env.agent_handler.positions = np.array([
            [5.0 + radius * np.cos(angle), 5.0 + radius * np.sin(angle)]
            for angle in angles
        ], dtype=np.float32)

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(4)}
        rewards = env._calculate_rewards(actions)

        # All rewards should be equal (all equidistant)
        reward_values = list(rewards.values())
        for i in range(1, len(reward_values)):
            assert np.isclose(reward_values[i], reward_values[0], atol=1e-5), \
                "All equidistant pursuers should have same reward"

    def test_zero_capture_radius(self):
        """Test with very small capture radius (difficult capture)."""
        env = PursuitEvasionEnv(
            num_pursuers=1,
            world_size=10.0,
            max_steps=100,
            capture_radius=0.01,  # Very small
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Place pursuer very close but not within tiny radius
        env.evader_pos = np.array([0.0, 0.0], dtype=np.float32)
        env.agent_handler.positions = np.array([[0.02, 0.0]], dtype=np.float32)

        actions = {"agent_0": np.array([0.0, 0.0], dtype=np.float32)}
        rewards = env._calculate_rewards(actions)

        # Should not capture (distance 0.02 > radius 0.01)
        assert rewards["agent_0"] < 1.0, "Should not capture with tiny radius"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
