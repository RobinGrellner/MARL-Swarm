import pytest
import numpy as np
import math
from environments.base.agent_handler import AgentHandler


class TestAgentHandlerInitialization:
    """Test AgentHandler initialization and validation."""

    def test_valid_initialization(self):
        """Test that AgentHandler initializes correctly with valid parameters."""
        handler = AgentHandler(
            num_agents=3,
            kinematics="single",
            v_max=1.0,
            omega_max=1.0,
            acc_v_max=2.0,
            acc_omega_max=2.0,
        )

        assert handler.num_agents == 3
        assert handler.kinematics == "single"
        assert handler.v_max == 1.0
        assert handler.omega_max == 1.0
        assert len(handler.agents) == 3
        assert handler.agents == ["agent_0", "agent_1", "agent_2"]

    def test_invalid_num_agents(self):
        """Test that invalid number of agents raises ValueError."""
        with pytest.raises(ValueError, match="There must be at least one Agent present"):
            AgentHandler(
                num_agents=0,
                kinematics="single",
                v_max=1.0,
                omega_max=1.0,
                acc_v_max=2.0,
                acc_omega_max=2.0,
            )

        with pytest.raises(ValueError):
            AgentHandler(
                num_agents=-1,
                kinematics="single",
                v_max=1.0,
                omega_max=1.0,
                acc_v_max=2.0,
                acc_omega_max=2.0,
            )

    def test_invalid_kinematics(self):
        """Test that invalid kinematics raises ValueError."""
        with pytest.raises(ValueError, match="kinematics must be 'single' or 'double'"):
            AgentHandler(
                num_agents=2,
                kinematics="triple",
                v_max=1.0,
                omega_max=1.0,
                acc_v_max=2.0,
                acc_omega_max=2.0,
            )

    @pytest.mark.parametrize("kinematics", ["single", "double"])
    def test_both_kinematics_modes(self, kinematics):
        """Test that both kinematics modes initialize correctly."""
        handler = AgentHandler(
            num_agents=2,
            kinematics=kinematics,
            v_max=1.0,
            omega_max=1.0,
            acc_v_max=2.0,
            acc_omega_max=2.0,
        )
        assert handler.kinematics == kinematics


class TestAgentHandlerActions:
    """Test action processing and movement."""

    @pytest.fixture
    def handler(self):
        """Create a basic AgentHandler for testing."""
        handler = AgentHandler(
            num_agents=3,
            kinematics="single",
            v_max=1.0,
            omega_max=1.0,
            acc_v_max=2.0,
            acc_omega_max=2.0,
        )
        # Initialize states
        handler.positions = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=np.float32)
        handler.linear_vels = np.zeros(3, dtype=np.float32)
        handler.angular_vels = np.zeros(3, dtype=np.float32)
        handler.orientations = np.zeros(3, dtype=np.float32)
        return handler

    @pytest.fixture
    def sample_actions(self):
        """Sample actions for testing."""
        return {
            "agent_0": np.array([0.5, 0.1], dtype=np.float32),
            "agent_1": np.array([0.3, -0.2], dtype=np.float32),
            "agent_2": np.array([0.8, 0.5], dtype=np.float32),
        }

    def test_clean_actions_valid_input(self, handler, sample_actions):
        """Test that valid actions are processed correctly."""
        cleaned = handler._clean_actions(sample_actions)

        assert cleaned.shape == (3, 2)
        assert cleaned.dtype == np.float32

        # Should match the order of agents
        np.testing.assert_array_equal(cleaned[0], np.array([0.5, 0.1], dtype=np.float32))
        np.testing.assert_array_equal(cleaned[1], np.array([0.3, -0.2], dtype=np.float32))
        np.testing.assert_array_equal(cleaned[2], np.array([0.8, 0.5], dtype=np.float32))

    def test_clean_actions_missing_agent(self, handler):
        """Test that missing agent actions raise RuntimeError."""
        incomplete_actions = {
            "agent_0": np.array([0.5, 0.1]),
            "agent_1": np.array([0.3, -0.2]),
            # Missing agent_2
        }

        with pytest.raises(RuntimeError, match="Actions must be provided for all agents"):
            handler._clean_actions(incomplete_actions)

    def test_clean_actions_extra_agent(self, handler, sample_actions):
        """Test that extra agent actions raise RuntimeError."""
        extra_actions = {
            **sample_actions,
            "agent_3": np.array([0.1, 0.1]),  # Extra agent
        }

        with pytest.raises(RuntimeError, match="Actions must be provided for all agents"):
            handler._clean_actions(extra_actions)

    def test_action_clipping(self, handler):
        """Test that actions are clipped to valid ranges."""
        extreme_actions = {
            "agent_0": np.array([10.0, -10.0], dtype=np.float32),  # Beyond limits
            "agent_1": np.array([-5.0, 3.0], dtype=np.float32),
            "agent_2": np.array([2.0, -2.0], dtype=np.float32),
        }

        cleaned = handler._clean_actions(extreme_actions)

        # Should be clipped to [-v_max, v_max] and [-omega_max, omega_max]
        assert np.all(cleaned[:, 0] <= handler.v_max)
        assert np.all(cleaned[:, 0] >= -handler.v_max)
        assert np.all(cleaned[:, 1] <= handler.omega_max)
        assert np.all(cleaned[:, 1] >= -handler.omega_max)

        # Specific checks
        np.testing.assert_array_equal(cleaned[0], [1.0, -1.0])  # Clipped
        np.testing.assert_array_equal(cleaned[1], [-1.0, 1.0])  # Clipped
        np.testing.assert_array_equal(cleaned[2], [1.0, -1.0])  # Clipped


class TestAgentMovement:
    """Test agent movement mechanics."""

    @pytest.fixture
    def single_handler(self):
        """Handler with single integrator kinematics."""
        handler = AgentHandler(
            num_agents=2,
            kinematics="single",
            v_max=5.0,
            omega_max=1.0,
            acc_v_max=2.0,
            acc_omega_max=2.0,
        )
        # Initialize at origin with zero velocities
        handler.positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        handler.linear_vels = np.zeros(2, dtype=np.float32)
        handler.angular_vels = np.zeros(2, dtype=np.float32)
        handler.orientations = np.array([0.0, math.pi / 2], dtype=np.float32)  # 0° and 90°
        return handler

    @pytest.fixture
    def double_handler(self):
        """Handler with double integrator kinematics."""
        handler = AgentHandler(
            num_agents=2,
            kinematics="double",
            v_max=5.0,
            omega_max=1.0,
            acc_v_max=2.0,
            acc_omega_max=2.0,
        )
        handler.positions = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32)
        handler.linear_vels = np.zeros(2, dtype=np.float32)
        handler.angular_vels = np.zeros(2, dtype=np.float32)
        handler.orientations = np.array([0.0, math.pi / 2], dtype=np.float32)
        return handler

    def test_single_integrator_movement(self, single_handler):
        """Test movement with single integrator kinematics."""
        actions = {
            "agent_0": np.array([1.0, 0.0], dtype=np.float32),  # Move forward
            "agent_1": np.array([0.5, 0.0], dtype=np.float32),  # Move forward slower
        }

        initial_positions = single_handler.positions.copy()
        single_handler.move(actions)

        # Velocities should be set directly to action values
        np.testing.assert_array_equal(single_handler.linear_vels, [1.0, 0.5])
        np.testing.assert_array_equal(single_handler.angular_vels, [0.0, 0.0])

        # Positions should change based on velocity and orientation
        # Agent 0: moves in x-direction (orientation = 0)
        # Agent 1: moves in y-direction (orientation = π/2)
        expected_pos = initial_positions + np.array(
            [
                [1.0, 0.0],  # Agent 0: v=1.0, θ=0 → dx=1.0, dy=0.0
                [0.0, 0.5],  # Agent 1: v=0.5, θ=π/2 → dx=0.0, dy=0.5
            ]
        )
        np.testing.assert_array_almost_equal(single_handler.positions, expected_pos, decimal=5)

    def test_double_integrator_movement(self, double_handler):
        """Test movement with double integrator kinematics."""
        actions = {
            "agent_0": np.array([2.0, 0.0], dtype=np.float32),  # Accelerate forward
            "agent_1": np.array([1.0, 0.0], dtype=np.float32),
        }

        initial_velocities = double_handler.linear_vels.copy()

        double_handler.move(actions)

        # Velocities should be updated by acceleration (not set directly)
        expected_vels = initial_velocities + np.array([2.0, 1.0])
        np.testing.assert_array_equal(double_handler.linear_vels, expected_vels)

    def test_angular_movement(self, single_handler):
        """Test angular movement updates orientation correctly."""
        actions = {
            "agent_0": np.array([0.0, 0.5], dtype=np.float32),  # Turn
            "agent_1": np.array([0.0, -0.3], dtype=np.float32),  # Turn opposite
        }

        initial_orientations = single_handler.orientations.copy()
        single_handler.move(actions)

        expected_orientations = initial_orientations + np.array([0.5, -0.3])
        np.testing.assert_array_almost_equal(single_handler.orientations, expected_orientations, decimal=5)

    def test_orientation_wrapping(self, single_handler):
        """Test that orientations are wrapped to [-π, π]."""
        # Set orientations close to boundaries
        single_handler.orientations = np.array([math.pi - 0.1, -math.pi + 0.1], dtype=np.float32)

        actions = {
            "agent_0": np.array([0.0, 0.5], dtype=np.float32),  # Should wrap to negative
            "agent_1": np.array([0.0, -0.5], dtype=np.float32),  # Should wrap to positive
        }

        single_handler.move(actions)

        # Check that orientations are within [-π, π]
        assert np.all(single_handler.orientations <= math.pi)
        assert np.all(single_handler.orientations >= -math.pi)

    def test_velocity_clipping_double_integrator(self, double_handler):
        """Test that velocities are clipped in double integrator mode."""
        # Start with high velocity
        double_handler.linear_vels = np.array([4.8, -4.9], dtype=np.float32)

        # Apply acceleration that would exceed limits
        actions = {
            "agent_0": np.array([0.5, 0.0], dtype=np.float32),  # Would go to 1.3
            "agent_1": np.array([-0.5, 0.0], dtype=np.float32),  # Would go to -1.4
        }

        double_handler.move(actions)

        # Should be clipped to [-v_max, v_max]
        assert np.all(double_handler.linear_vels <= double_handler.v_max)
        assert np.all(double_handler.linear_vels >= -double_handler.v_max)

        np.testing.assert_array_equal(double_handler.linear_vels, [5.0, -5.0])  # Clipped


class TestPositionInitialization:
    """Test position initialization methods."""

    @pytest.fixture
    def handler(self):
        return AgentHandler(
            num_agents=5,
            kinematics="single",
            v_max=1.0,
            omega_max=1.0,
            acc_v_max=2.0,
            acc_omega_max=2.0,
        )

    def test_initialize_random_positions(self, handler):
        """Test random position initialization."""
        world_size = 10.0

        # Set seed for reproducible test
        np.random.seed(42)
        handler.initialize_random_positions(world_size)

        # Check shape and bounds
        assert handler.positions.shape == (5, 2)
        assert handler.positions.dtype == np.float32
        assert np.all(handler.positions >= 0.0)
        assert np.all(handler.positions <= world_size)

    def test_initialize_random_positions_reproducible(self, handler):
        """Test that same seed produces same positions."""
        world_size = 5.0

        # First initialization
        np.random.seed(123)
        handler.initialize_random_positions(world_size)
        positions1 = handler.positions.copy()

        # Second initialization with same seed
        np.random.seed(123)
        handler.initialize_random_positions(world_size)
        positions2 = handler.positions.copy()

        np.testing.assert_array_equal(positions1, positions2)

    def test_initialize_different_world_sizes(self, handler):
        """Test initialization with different world sizes."""
        world_sizes = [1.0, 5.0, 100.0]

        for world_size in world_sizes:
            handler.initialize_random_positions(world_size)

            assert np.all(handler.positions >= 0.0)
            assert np.all(handler.positions <= world_size)


class TestAgentHandlerProperties:
    """Test computed properties and state consistency."""

    @pytest.fixture
    def handler(self):
        handler = AgentHandler(
            num_agents=2,
            kinematics="single",
            v_max=1.0,
            omega_max=1.0,
            acc_v_max=2.0,
            acc_omega_max=2.0,
        )
        # Initialize with known values
        handler.positions = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        handler.linear_vels = np.array([0.5, 1.0], dtype=np.float32)
        handler.angular_vels = np.array([0.1, -0.2], dtype=np.float32)
        handler.orientations = np.array([0.0, math.pi / 2], dtype=np.float32)
        return handler

    def test_state_consistency_after_move(self, handler):
        """Test that all states are updated consistently during movement."""
        actions = {
            "agent_0": np.array([0.8, 0.2], dtype=np.float32),
            "agent_1": np.array([0.6, -0.1], dtype=np.float32),
        }

        initial_positions = handler.positions.copy()
        handler.move(actions)

        # Positions should have changed
        assert not np.array_equal(handler.positions, initial_positions)

        # All arrays should have correct shapes
        assert handler.positions.shape == (2, 2)
        assert handler.linear_vels.shape == (2,)
        assert handler.angular_vels.shape == (2,)
        assert handler.orientations.shape == (2,)

    def test_no_state_arrays_share_memory(self, handler):
        """Test that state arrays are independent."""
        original_positions = handler.positions.copy()

        # Modify positions
        handler.positions[0, 0] = 999.0

        # Original should be unchanged
        assert original_positions[0, 0] != 999.0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_actions(self):
        """Test handling of zero actions."""
        handler = AgentHandler(
            num_agents=2,
            kinematics="single",
            v_max=1.0,
            omega_max=1.0,
            acc_v_max=2.0,
            acc_omega_max=2.0,
        )

        handler.positions = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        handler.linear_vels = np.array([0.5, 0.3], dtype=np.float32)
        handler.angular_vels = np.zeros(2, dtype=np.float32)
        handler.orientations = np.zeros(2, dtype=np.float32)

        actions = {
            "agent_0": np.array([0.0, 0.0], dtype=np.float32),
            "agent_1": np.array([0.0, 0.0], dtype=np.float32),
        }

        initial_positions = handler.positions.copy()
        handler.move(actions)

        # Single integrator: zero actions should result in zero velocities
        np.testing.assert_array_equal(handler.linear_vels, [0.0, 0.0])
        np.testing.assert_array_equal(handler.angular_vels, [0.0, 0.0])

        # Position should remain unchanged (no velocity)
        np.testing.assert_array_equal(handler.positions, initial_positions)

    def test_single_agent(self):
        """Test handler works with single agent."""
        handler = AgentHandler(
            num_agents=1,
            kinematics="single",
            v_max=1.0,
            omega_max=1.0,
            acc_v_max=2.0,
            acc_omega_max=2.0,
        )

        handler.initialize_random_positions(5.0)
        assert handler.positions.shape == (1, 2)
        assert len(handler.agents) == 1
        assert handler.agents[0] == "agent_0"

        actions = {"agent_0": np.array([0.5, 0.1], dtype=np.float32)}
        handler.move(actions)  # Should not crash

    @pytest.mark.parametrize("num_agents", [1, 2, 5, 10, 100])
    def test_scaling_with_agent_count(self, num_agents):
        """Test that handler scales properly with different agent counts."""
        handler = AgentHandler(
            num_agents=num_agents,
            kinematics="single",
            v_max=1.0,
            omega_max=1.0,
            acc_v_max=2.0,
            acc_omega_max=2.0,
        )

        handler.initialize_random_positions(10.0)

        # Create actions for all agents
        actions = {f"agent_{i}": np.random.uniform(-0.5, 0.5, 2).astype(np.float32) for i in range(num_agents)}

        handler.move(actions)

        # Verify all arrays have correct shapes
        assert handler.positions.shape == (num_agents, 2)
        assert handler.linear_vels.shape == (num_agents,)
        assert handler.angular_vels.shape == (num_agents,)
        assert handler.orientations.shape == (num_agents,)


# Fixture for running specific test groups
@pytest.fixture(scope="session")
def performance_config():
    """Configuration for performance tests."""
    return {
        "num_agents": 1000,
        "kinematics": "single",
        "v_max": 1.0,
        "omega_max": 1.0,
        "acc_v_max": 2.0,
        "acc_omega_max": 2.0,
    }


class TestPerformance:
    """Performance tests for AgentHandler."""

    def test_large_scale_movement(self, performance_config):
        """Test performance with many agents."""
        import time

        handler = AgentHandler(**performance_config)
        handler.initialize_random_positions(100.0)

        # Create random actions
        actions = {
            f"agent_{i}": np.random.uniform(-1, 1, 2).astype(np.float32)
            for i in range(performance_config["num_agents"])
        }

        # Time the movement
        start_time = time.time()
        for _ in range(100):  # 100 steps
            handler.move(actions)
        end_time = time.time()

        steps_per_second = 100 / (end_time - start_time)
        print(f"Performance: {steps_per_second:.1f} steps/second with {performance_config['num_agents']} agents")

        # Should be reasonably fast
        assert steps_per_second > 10


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__])
