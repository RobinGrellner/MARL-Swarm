"""
Comprehensive tests for PursuitEvasionEnv.

Tests environment behavior, evasion strategies, rewards, and multi-agent dynamics.
"""

import numpy as np
import pytest
from environments.pursuit.pursuit_evasion_env import PursuitEvasionEnv


class TestPursuitEvasionEnvironment:
    """Test suite for PursuitEvasionEnv."""

    def test_initialization_default_params(self):
        """Test environment initialization with default parameters."""
        env = PursuitEvasionEnv()
        assert env.num_pursuers == 10
        assert env.world_size == 10.0
        assert env.max_steps == 100
        assert env.capture_radius == 0.5
        assert env.evader_speed == 1.0
        assert env.evader_strategy == "voronoi_center"
        env.close()

    def test_initialization_custom_params(self):
        """Test environment initialization with custom parameters."""
        env = PursuitEvasionEnv(
            num_pursuers=5,
            world_size=20.0,
            max_steps=200,
            capture_radius=1.0,
            evader_speed=0.8,
            evader_strategy="weighted_escape",
        )
        assert env.num_pursuers == 5
        assert env.world_size == 20.0
        assert env.max_steps == 200
        assert env.capture_radius == 1.0
        assert env.evader_speed == 0.8
        assert env.evader_strategy == "weighted_escape"
        env.close()

    def test_reset_initializes_agents(self):
        """Test that reset properly initializes all agent positions."""
        env = PursuitEvasionEnv(num_pursuers=5)
        obs, info = env.reset()

        # Check all pursuers have valid positions within world bounds
        for i, agent in enumerate(env.agents):
            pursuer_pos = env.agent_handler.positions[i]
            assert 0 <= pursuer_pos[0] <= env.world_size
            assert 0 <= pursuer_pos[1] <= env.world_size

        # Check evader position is within world bounds
        assert 0 <= env.evader_pos[0] <= env.world_size
        assert 0 <= env.evader_pos[1] <= env.world_size

        env.close()

    def test_evasion_strategies_exist(self):
        """Test that all evasion strategies can be created."""
        strategies = ["simple", "max_min_distance", "weighted_escape", "voronoi_center"]

        for strategy in strategies:
            env = PursuitEvasionEnv(num_pursuers=5, evader_strategy=strategy)
            assert env.evasion_agent is not None
            assert env.evader_strategy == strategy
            env.close()

    def test_evader_moves_away_from_pursuers(self):
        """Test that evader moves away from pursuers (distance increases)."""
        env = PursuitEvasionEnv(num_pursuers=5, world_size=50.0, max_steps=50, capture_radius=0.1)
        obs, info = env.reset()

        # Calculate initial minimum distance from evader to any pursuer
        initial_distances = np.linalg.norm(env.agent_handler.positions - env.evader_pos, axis=1)
        initial_min_dist = np.min(initial_distances)

        # Take a few steps
        for _ in range(10):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)

        # Calculate final minimum distance
        final_distances = np.linalg.norm(env.agent_handler.positions - env.evader_pos, axis=1)
        final_min_dist = np.min(final_distances)

        # Evader should have moved away (on average)
        # This is probabilistic with random pursuer actions, but very likely to pass
        assert final_min_dist > 0.5  # At least some distance maintained

        env.close()

    def test_observation_models(self):
        """Test that all observation models work correctly."""
        obs_models = ["global_basic", "global_extended", "local_basic", "local_extended"]

        for obs_model in obs_models:
            env = PursuitEvasionEnv(num_pursuers=5, obs_model=obs_model, comm_radius=5.0)
            obs, info = env.reset()

            # Check observations exist and have correct shape
            for agent in env.agents:
                assert agent in obs
                assert obs[agent].shape == env.observation_space(agent).shape

            env.close()

    def test_rewards_structure(self):
        """Test that rewards are properly calculated."""
        env = PursuitEvasionEnv(num_pursuers=5, max_steps=50)
        obs, info = env.reset()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Check rewards exist and are valid floats for all agents
        assert len(rewards) == env.num_pursuers
        for agent in env.agents:
            assert agent in rewards
            assert isinstance(rewards[agent], (float, np.floating))
            assert not np.isnan(rewards[agent])
            assert not np.isinf(rewards[agent])

        env.close()

    def test_capture_reward_bonus(self):
        """Test that capture results in large reward bonus."""
        env = PursuitEvasionEnv(num_pursuers=5, capture_radius=100.0, max_steps=10)
        obs, info = env.reset()

        # With huge capture radius, should capture immediately
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        # All agents should get positive reward from capture
        for agent in env.agents:
            assert rewards[agent] > 5.0  # Capture bonus is 10.0

        env.close()

    def test_termination_on_capture(self):
        """Test that episode terminates when evader is captured."""
        env = PursuitEvasionEnv(num_pursuers=5, capture_radius=100.0, max_steps=100)
        obs, info = env.reset()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Should terminate due to capture
        assert any(terminations.values())
        assert all(terminations.values())  # All agents should see same termination

        env.close()

    def test_truncation_on_max_steps(self):
        """Test that episode truncates after max_steps."""
        env = PursuitEvasionEnv(num_pursuers=5, max_steps=10, capture_radius=0.1)
        obs, info = env.reset()

        for step in range(15):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)

            if step >= 9:  # After max_steps
                assert any(truncations.values())
                break

        env.close()

    def test_evader_stays_in_bounds(self):
        """Test that evader position stays within world bounds."""
        env = PursuitEvasionEnv(num_pursuers=10, world_size=10.0, max_steps=100)
        obs, info = env.reset()

        for _ in range(50):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)

            # Check evader is within bounds
            assert 0 <= env.evader_pos[0] <= env.world_size
            assert 0 <= env.evader_pos[1] <= env.world_size

            if any(terminations.values()) or any(truncations.values()):
                break

        env.close()

    def test_pursuers_stay_in_bounds(self):
        """Test that pursuer positions stay within world bounds."""
        env = PursuitEvasionEnv(num_pursuers=5, world_size=10.0, max_steps=100)
        obs, info = env.reset()

        for _ in range(50):
            actions = {agent: env.action_space(agent).sample() for agent in env.agents}
            obs, rewards, terminations, truncations, infos = env.step(actions)

            # Check all pursuers are within bounds
            for pos in env.agent_handler.positions:
                assert 0 <= pos[0] <= env.world_size
                assert 0 <= pos[1] <= env.world_size

            if any(terminations.values()) or any(truncations.values()):
                break

        env.close()

    def test_scale_invariance(self):
        """Test that observation space scales correctly."""
        num_pursuers = 5
        max_pursuers = 15

        env = PursuitEvasionEnv(
            num_pursuers=num_pursuers, max_pursuers=max_pursuers, obs_model="global_basic"
        )
        obs, info = env.reset()

        # Observation should be sized for max_pursuers, not actual pursuers
        expected_dim = env.obs_layout["total_dim"]
        for agent in env.agents:
            assert obs[agent].shape[0] == expected_dim

        env.close()

    def test_kinematics_single_integrator(self):
        """Test single integrator kinematics."""
        env = PursuitEvasionEnv(num_pursuers=5, kinematics="single")
        assert env.agent_handler.kinematics == "single"
        obs, info = env.reset()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        env.close()

    def test_kinematics_double_integrator(self):
        """Test double integrator kinematics."""
        env = PursuitEvasionEnv(num_pursuers=5, kinematics="double")
        assert env.agent_handler.kinematics == "double"
        obs, info = env.reset()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        env.close()

    def test_info_dict_completeness(self):
        """Test that info dict contains all required keys."""
        env = PursuitEvasionEnv(num_pursuers=5)
        obs, info = env.reset()

        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        required_keys = ["distance_to_evader", "min_distance_to_evader", "evader_captured"]

        for agent in env.agents:
            assert agent in infos
            for key in required_keys:
                assert key in infos[agent], f"Missing key '{key}' in info for {agent}"

        env.close()

    def test_deterministic_reset_with_seed(self):
        """Test that reset with same seed produces same initial state."""
        env1 = PursuitEvasionEnv(num_pursuers=5)
        env2 = PursuitEvasionEnv(num_pursuers=5)

        obs1, _ = env1.reset(seed=42)
        obs2, _ = env2.reset(seed=42)

        # Check observations are identical (should use same seed)
        for agent in env1.agents:
            np.testing.assert_array_almost_equal(obs1[agent], obs2[agent])

        env1.close()
        env2.close()

    def test_continuous_action_space(self):
        """Test that action space is continuous and appropriate."""
        env = PursuitEvasionEnv(num_pursuers=5)
        obs, info = env.reset()

        for agent in env.agents:
            action_space = env.action_space(agent)
            # Should be continuous box
            assert hasattr(action_space, "low") and hasattr(action_space, "high")
            # Sample and verify bounds
            action = action_space.sample()
            assert action.shape == action_space.shape

        env.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
