"""
Test termination and truncation conditions for RendezvousEnv.

Tests cover:
- Early termination when all agents reach rendezvous threshold
- No termination when threshold not met
- Truncation at max_steps
- Torus wrapping effects on distance calculations for termination
"""

import pytest
import numpy as np
from environments.rendezvous.rendezvous_env import RendezvousEnv


class TestRendezvousTermination:
    """Test termination conditions for rendezvous environment."""

    def test_no_early_termination_when_threshold_none(self):
        """Test that no early termination occurs when break_distance_threshold is None."""
        env = RendezvousEnv(
            num_agents=3,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
            break_distance_threshold=None,  # No early termination
        )
        env.reset(seed=42)

        # Place agents very close together
        env.agent_handler.positions = np.array([[5.0, 5.0], [5.01, 5.01], [5.02, 5.02]], dtype=np.float32)

        terminations = env._check_terminations()

        # No agent should be terminated
        for agent_name, terminated in terminations.items():
            assert not terminated, f"{agent_name} should not be terminated when threshold is None"

    def test_early_termination_when_threshold_met(self):
        """Test that agents terminate when all pairwise distances below threshold."""
        env = RendezvousEnv(
            num_agents=3,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
            break_distance_threshold=0.5,  # Terminate when all distances < 0.5
        )
        env.reset(seed=42)

        # Place agents very close together (all distances < 0.5)
        env.agent_handler.positions = np.array(
            [
                [5.0, 5.0],
                [5.1, 5.0],  # Distance from first: 0.1
                [5.0, 5.1],  # Distance from first: 0.1, from second: sqrt(0.1^2 + 0.1^2) ≈ 0.14
            ],
            dtype=np.float32,
        )

        terminations = env._check_terminations()

        # All agents should be terminated
        for agent_name, terminated in terminations.items():
            assert terminated, f"{agent_name} should be terminated when all distances below threshold"

    def test_no_termination_when_one_agent_far(self):
        """Test that no termination occurs if even one agent is too far."""
        env = RendezvousEnv(
            num_agents=3,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
            break_distance_threshold=0.5,
        )
        env.reset(seed=42)

        # Two agents close, one far
        env.agent_handler.positions = np.array(
            [
                [5.0, 5.0],
                [5.1, 5.0],  # Close to first
                [8.0, 8.0],  # Far from both
            ],
            dtype=np.float32,
        )

        terminations = env._check_terminations()

        # No agent should be terminated
        for agent_name, terminated in terminations.items():
            assert not terminated, f"{agent_name} should not be terminated when one agent is far"

    def test_termination_threshold_boundary(self):
        """Test termination at exact threshold boundary."""
        threshold = 1.0
        env = RendezvousEnv(
            num_agents=2,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
            break_distance_threshold=threshold,
        )
        env.reset(seed=42)

        # Test just below threshold
        env.agent_handler.positions = np.array([[0.0, 0.0], [0.999, 0.0]], dtype=np.float32)
        terminations = env._check_terminations()
        for terminated in terminations.values():
            assert terminated, "Should terminate when distance < threshold"

        # Test exactly at threshold (should not terminate - threshold is exclusive)
        env.agent_handler.positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        terminations = env._check_terminations()
        for terminated in terminations.values():
            assert not terminated, "Should not terminate when distance == threshold"

        # Test just above threshold
        env.agent_handler.positions = np.array([[0.0, 0.0], [1.001, 0.0]], dtype=np.float32)
        terminations = env._check_terminations()
        for terminated in terminations.values():
            assert not terminated, "Should not terminate when distance > threshold"

    def test_termination_with_torus_wrapping(self):
        """Test that termination uses torus-wrapped distances."""
        env = RendezvousEnv(
            num_agents=2,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
            break_distance_threshold=2.0,
            torus=True,
        )
        env.reset(seed=42)

        # Place agents near opposite edges
        # Direct distance = 9.0, but wrapped distance = 1.0
        env.agent_handler.positions = np.array([[0.5, 5.0], [9.5, 5.0]], dtype=np.float32)

        terminations = env._check_terminations()

        # Should terminate because wrapped distance (1.0) < threshold (2.0)
        for terminated in terminations.values():
            assert terminated, "Should terminate using wrapped distance in torus mode"

    def test_termination_multiple_agents_torus(self):
        """Test termination with multiple agents in torus mode."""
        env = RendezvousEnv(
            num_agents=4,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
            break_distance_threshold=2.0,
            torus=True,
        )
        env.reset(seed=42)

        # Place all agents in a small cluster that wraps around edges
        env.agent_handler.positions = np.array(
            [
                [0.2, 0.2],
                [9.9, 0.2],  # Wrapped distance from first ≈ 0.3
                [0.2, 9.9],  # Wrapped distance from first ≈ 0.3
                [9.9, 9.9],  # Wrapped distance from first ≈ 0.42
            ],
            dtype=np.float32,
        )

        terminations = env._check_terminations()

        # All wrapped distances should be < 2.0, so should terminate
        for terminated in terminations.values():
            assert terminated, "Should terminate when all wrapped distances below threshold"


class TestRendezvousTruncation:
    """Test truncation conditions (max_steps)."""

    def test_no_truncation_before_max_steps(self):
        """Test that no truncation occurs before max_steps."""
        env = RendezvousEnv(
            num_agents=2,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Step multiple times but not to max_steps
        for _ in range(50):
            actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(2)}
            env.step(actions)

            truncations = env._check_truncations()
            for truncated in truncations.values():
                assert not truncated, "Should not truncate before max_steps"

    def test_truncation_at_max_steps(self):
        """Test that truncation occurs exactly at max_steps."""
        max_steps = 10
        env = RendezvousEnv(
            num_agents=2,
            world_size=10.0,
            max_steps=max_steps,
            obs_model="global_basic",
        )
        env.reset(seed=42)

        # Step to max_steps - 1
        for _ in range(max_steps - 1):
            actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(2)}
            _, _, _, truncations, _ = env.step(actions)

        # Step one more time to reach max_steps
        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(2)}
        _, _, _, truncations, _ = env.step(actions)

        # All agents should be truncated
        for agent_name, truncated in truncations.items():
            assert truncated, f"{agent_name} should be truncated at max_steps"

    def test_truncation_independent_of_termination(self):
        """Test that truncation and termination are independent."""
        env = RendezvousEnv(
            num_agents=2,
            world_size=10.0,
            max_steps=5,
            obs_model="global_basic",
            break_distance_threshold=0.5,
        )
        env.reset(seed=42)

        # Place agents close enough to terminate
        env.agent_handler.positions = np.array([[0.0, 0.0], [0.1, 0.0]], dtype=np.float32)

        # Step once
        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(2)}
        _, _, terminations, truncations, _ = env.step(actions)

        # Should be terminated but not truncated (only 1 step taken)
        for terminated in terminations.values():
            assert terminated, "Should be terminated due to distance threshold"
        for truncated in truncations.values():
            assert not truncated, "Should not be truncated (only 1 step taken)"


class TestRendezvousFullEpisode:
    """Test full episode flow with termination/truncation."""

    def test_episode_terminates_early(self):
        """Test that episode can terminate early when threshold is met."""
        env = RendezvousEnv(
            num_agents=2,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
            break_distance_threshold=0.5,
        )
        observations, infos = env.reset(seed=42)

        # Move agents toward each other manually
        env.agent_handler.positions = np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32)

        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(2)}
        observations, rewards, terminations, truncations, infos = env.step(actions)

        # Should terminate immediately
        assert all(terminations.values()), "Episode should terminate when agents at same position"
        assert not any(truncations.values()), "Should not truncate when terminated early"

    def test_episode_truncates_at_max_steps(self):
        """Test that episode truncates at max_steps when threshold not met."""
        max_steps = 5
        env = RendezvousEnv(
            num_agents=2,
            world_size=10.0,
            max_steps=max_steps,
            obs_model="global_basic",
            break_distance_threshold=0.1,  # Very strict threshold
        )
        env.reset(seed=42)

        # Keep agents far apart
        for step in range(max_steps):
            env.agent_handler.positions = np.array([[0.0, 0.0], [10.0, 10.0]], dtype=np.float32)
            actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(2)}
            observations, rewards, terminations, truncations, infos = env.step(actions)

            if step < max_steps - 1:
                assert not any(terminations.values()), "Should not terminate early"
                assert not any(truncations.values()), "Should not truncate before max_steps"
            else:
                assert not any(terminations.values()), "Should not terminate (threshold not met)"
                assert all(truncations.values()), "Should truncate at max_steps"

    def test_reset_after_termination(self):
        """Test that environment can be reset after termination."""
        env = RendezvousEnv(
            num_agents=2,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
            break_distance_threshold=0.5,
        )

        # First episode
        observations1, _ = env.reset(seed=42)
        env.agent_handler.positions = np.array([[5.0, 5.0], [5.0, 5.0]], dtype=np.float32)
        actions = {f"agent_{i}": np.array([0.0, 0.0], dtype=np.float32) for i in range(2)}
        _, _, terminations, _, _ = env.step(actions)
        assert all(terminations.values())

        # Reset and second episode
        observations2, _ = env.reset(seed=43)

        # Observations should be different (different random seed)
        # and environment should be ready for new episode
        assert env.current_step == 0, "Step counter should be reset"
        assert len(observations2) == 2, "Should have observations for all agents"


class TestTorusWrapMovement:
    """Test that agents correctly wrap around in torus mode."""

    def test_position_wrapping_x_axis(self):
        """Test that positions wrap correctly on x-axis."""
        env = RendezvousEnv(
            num_agents=1,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
            torus=True,
        )
        env.reset(seed=42)

        # Place agent near right edge and move it right
        env.agent_handler.positions = np.array([[9.5, 5.0]], dtype=np.float32)
        env.agent_handler.orientations = np.array([0.0], dtype=np.float32)  # Facing right
        env.agent_handler.linear_vels = np.array([1.0], dtype=np.float32)

        # Move agent
        actions = {"agent_0": np.array([1.0, 0.0], dtype=np.float32)}
        env.step(actions)

        # Position should wrap to near left edge
        expected_x = (9.5 + 1.0) % 10.0  # 0.5
        assert np.isclose(env.agent_handler.positions[0, 0], expected_x, atol=1e-5), (
            "Position should wrap around x-axis in torus mode"
        )

    def test_position_wrapping_y_axis(self):
        """Test that positions wrap correctly on y-axis."""
        env = RendezvousEnv(
            num_agents=1,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
            torus=True,
        )
        env.reset(seed=42)

        # Place agent near top edge and move it up
        env.agent_handler.positions = np.array([[5.0, 9.5]], dtype=np.float32)
        env.agent_handler.orientations = np.array([np.pi / 2], dtype=np.float32)  # Facing up
        env.agent_handler.linear_vels = np.array([1.0], dtype=np.float32)

        # Move agent
        actions = {"agent_0": np.array([1.0, 0.0], dtype=np.float32)}
        env.step(actions)

        # Position should wrap to near bottom edge
        expected_y = (9.5 + 1.0) % 10.0  # 0.5
        assert np.isclose(env.agent_handler.positions[0, 1], expected_y, atol=1e-5), (
            "Position should wrap around y-axis in torus mode"
        )

    def test_no_wrapping_without_torus(self):
        """Test that positions don't wrap when torus=False."""
        env = RendezvousEnv(
            num_agents=1,
            world_size=10.0,
            max_steps=100,
            obs_model="global_basic",
            torus=False,
        )
        env.reset(seed=42)

        # Place agent near edge and move it past boundary
        env.agent_handler.positions = np.array([[9.5, 5.0]], dtype=np.float32)
        env.agent_handler.orientations = np.array([0.0], dtype=np.float32)  # Facing right
        env.agent_handler.linear_vels = np.array([1.0], dtype=np.float32)

        # Move agent
        actions = {"agent_0": np.array([1.0, 0.0], dtype=np.float32)}
        env.step(actions)

        # Position should go past boundary (no wrapping)
        expected_x = 9.5 + 1.0  # 10.5 (outside world)
        assert np.isclose(env.agent_handler.positions[0, 0], expected_x, atol=1e-5), (
            "Position should not wrap when torus=False"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
