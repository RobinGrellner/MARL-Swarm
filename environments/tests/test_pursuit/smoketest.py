"""
Smoke test for PursuitEvasionEnv.

Tests basic functionality without training.
"""

from environments.pursuit.pursuit_evasion_env import PursuitEvasionEnv


def test_environment_creation():
    """Test that the environment can be created."""
    env = PursuitEvasionEnv(num_pursuers=5, world_size=10.0, max_steps=50)
    assert len(env.agents) == 5
    assert env.world_size == 10.0
    assert env.max_steps == 50
    env.close()


def test_reset():
    """Test that the environment can be reset."""
    env = PursuitEvasionEnv(num_pursuers=5)
    obs, info = env.reset()

    assert len(obs) == 5
    assert all(agent in obs for agent in env.agents)
    assert all(agent in info for agent in env.agents)

    env.close()


def test_step():
    """Test that the environment can take steps."""
    env = PursuitEvasionEnv(num_pursuers=5, max_steps=10)
    obs, info = env.reset()

    for _ in range(5):
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        obs, rewards, terminations, truncations, infos = env.step(actions)

        assert len(obs) == len(env.agents)
        assert len(rewards) == len(env.agents)
        assert len(terminations) == len(env.agents)
        assert len(truncations) == len(env.agents)
        assert len(infos) == len(env.agents)

    env.close()


def test_observation_shapes():
    """Test that observations have the correct shape."""
    env = PursuitEvasionEnv(num_pursuers=5, obs_model="global_basic")
    obs, _ = env.reset()

    # Check observation shape matches space
    for agent in env.agents:
        assert obs[agent].shape == env.observation_space(agent).shape
        assert obs[agent].shape == (env.obs_layout["total_dim"],)

    env.close()


def test_capture_termination():
    """Test that episode terminates when evader is captured."""
    env = PursuitEvasionEnv(num_pursuers=5, capture_radius=100.0, max_steps=100)
    obs, _ = env.reset()

    # With large capture radius, should terminate immediately
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    obs, rewards, terminations, truncations, infos = env.step(actions)

    # Should be captured since capture radius is huge
    assert any(terminations.values())
    assert any(infos[agent]["evader_captured"] for agent in env.agents)

    env.close()


def test_scale_invariance():
    """Test scale-invariant observation model."""
    # Create env with 5 pursuers but max_pursuers=10
    env = PursuitEvasionEnv(num_pursuers=5, max_pursuers=10, obs_model="global_basic")
    obs, _ = env.reset()

    # Observation should be sized for 10 pursuers
    assert env.obs_layout["max_neighbours"] == 9  # 10 - 1

    # All agents should have same observation shape
    for agent in env.agents:
        assert obs[agent].shape == env.observation_space(agent).shape

    env.close()


if __name__ == "__main__":
    print("Running pursuit-evasion environment tests...")
    test_environment_creation()
    print("✓ Environment creation test passed")

    test_reset()
    print("✓ Reset test passed")

    test_step()
    print("✓ Step test passed")

    test_observation_shapes()
    print("✓ Observation shape test passed")

    test_capture_termination()
    print("✓ Capture termination test passed")

    test_scale_invariance()
    print("✓ Scale-invariance test passed")

    print("\n✓ All tests passed!")
