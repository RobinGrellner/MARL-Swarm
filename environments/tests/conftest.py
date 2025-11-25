import pytest
import numpy as np
from environments.rendezvous.rendezvous_env import RendezvousEnv
from environments.pursuit.pursuit_env import PursuitEnv


@pytest.fixture
def seed():
    """Fixed seed for reproducible tests."""
    return 42


@pytest.fixture
def basic_config():
    """Basic configuration for testing the AgentHandler"""
    return {
        "num_agents": 10,
        "kinematics": "single",
        "v_max": 1.0,
        "omega_max": np.deg2rad(30),
        "acc_v_max": 0.1,
        "acc_omega_max": np.deg2rad(3),
    }


@pytest.fixture
def rendezvous_env(basic_config):
    """Create a rendezvous environment for testing."""
    return RendezvousEnv(**basic_config)  # TODO: Add rendezvous specific config


@pytest.fixture
def pursuit_env(basic_config):
    """Create a pursuit environment for testing."""
    return PursuitEnv(**basic_config)  # TODO: Add pursuit specific config


@pytest.fixture
def sample_actions(basic_config):
    """Sample actions for all agents."""
    num_agents = basic_config["num_agents"]
    return {f"agent_{i}": np.array([0.5, 0.1], dtype=np.float32) for i in range(num_agents)}
