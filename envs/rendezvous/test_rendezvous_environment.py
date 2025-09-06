import envs.rendezvous.rendezvous_environment as r_env
from pettingzoo.test import parallel_api_test, parallel_seed_test
import unittest
import numpy as np


class TestRendezvousEnvironment(unittest.TestCase):

    def test_pettingzoo_tests(self):
        env_fn = r_env.parallel_env
        parallel_api_test(env_fn(), num_cycles=10_000)
        parallel_seed_test(env_fn)

    def test_agent_num(self):
        env_fn = r_env.parallel_env
        env = env_fn(num_agents=10)
        self.assertEqual(len(env.possible_agents), 10)

        env_2 = env_fn(num_agents=2)
        self.assertEqual(len(env_2.possible_agents), 2)

    def test_init(self):
        env_fn = r_env.parallel_env
        with self.assertRaises(ValueError):
            env_fn(num_agents=-10)
        with self.assertRaises(ValueError):
            env_fn(num_agents=1.5)

        with self.assertRaises(ValueError):
            env_fn(world_size=(-2, 2))
        with self.assertRaises(ValueError):
            env_fn(world_size=(2, -2))
        with self.assertRaises(ValueError):
            env_fn(world_size=(-2, -2))

        with self.assertRaises(ValueError):
            env_fn(max_cycles=0)
        with self.assertRaises(ValueError):
            env_fn(max_cycles=-2)

    def test_agent_movement_simple(self):
        env = r_env.parallel_env(
            max_cycles=5,
            world_size=(100, 100),
            num_agents=2,
        )
        np.random.seed(100)
        env.reset()
        env.agent_map["agent_0"].set_coords(50, 50)
        env.agent_map["agent_0"].phi = 0
        env.agent_map["agent_1"].set_coords(100, 100)
        env.agent_map["agent_1"].phi = 0

        # Val1 = velocity, val2 = steering angle
        actions = {"agent_0": [5, np.deg2rad(0)], "agent_1": [0, np.deg2rad(10)]}

        agent_0_coord = env.agent_map["agent_0"].get_coords()
        agent_1_coord = env.agent_map["agent_1"].get_coords()

        env.step(actions)

        agent_0_coord_after = env.agent_map["agent_0"].get_coords()
        agent_1_coord_after = env.agent_map["agent_1"].get_coords()

        self.assertEqual((agent_0_coord[0] + 5, agent_0_coord[1]), agent_0_coord_after)
        self.assertEqual(agent_1_coord, agent_1_coord_after)

    def test_agent_movement_rotation_only(self):
        env = r_env.parallel_env(
            max_cycles=5,
            world_size=(100, 100),
            num_agents=2,
        )
        np.random.seed(100)
        env.reset()

        env.agent_map["agent_0"].set_coords(50, 50)
        env.agent_map["agent_0"].phi = 0
        env.agent_map["agent_1"].set_coords(100, 100)
        env.agent_map["agent_1"].phi = 0

        # val1 = velocity, val2 = steering angle
        actions = {"agent_0": [0, np.deg2rad(10)]}
        coord = env.agent_map["agent_0"].get_coords()
        vel_phi = env.agent_map["agent_0"].get_vel_phi()

        env.step(actions)

        coord_after = env.agent_map["agent_0"].get_coords()
        vel_phi_after = env.agent_map["agent_0"].get_vel_phi()

        self.assertEqual(coord, coord_after)

        self.assertEqual((vel_phi[0], vel_phi[1] + np.deg2rad(10)), vel_phi_after)

    def test_observation(self):
        env = r_env.parallel_env(
            max_cycles=5,
            world_size=(100, 100),
            num_agents=2,
        )
        np.random.seed(100)
        env.reset()

        env.agent_map["agent_0"].set_coords(20, 10)
        env.agent_map["agent_0"].phi = 0
        env.agent_map["agent_1"].set_coords(70, 10)
        env.agent_map["agent_1"].phi = np.deg2rad(90)

        # val1 = velocity, val2 = steering angle
        actions = {"agent_0": [0, 0], "agent_1": [0, 0]}

        obs, rewards, terminations, truncations, infos = env.step(actions)

        expected_distance = 50
        expected_bearing = np.deg2rad(90)
        a0_expected_distance_to_wall = 10
        a1_expected_distance_to_wall = 10
        a0_expected_bearing_to_wall = -np.deg2rad(90)
        a1_expected_bearing_to_wall = 0

        expected_obs = {
            "agent_0": [
                a0_expected_distance_to_wall,
                a0_expected_bearing_to_wall,
                expected_distance,
                0,
            ],
            "agent_1": [
                a1_expected_distance_to_wall,
                a1_expected_bearing_to_wall,
                expected_distance,
                -expected_bearing,
            ],
        }

        self.assertEqual(obs, expected_obs)

    def test_reward(self):
        env = r_env.parallel_env(
            max_cycles=5,
            world_size=(100, 100),
            num_agents=2,
        )
        np.random.seed(100)
        env.reset()

        env.agent_map["agent_0"].set_coords(20, 10)
        env.agent_map["agent_0"].phi = 0
        env.agent_map["agent_1"].set_coords(70, 10)
        env.agent_map["agent_1"].phi = np.deg2rad(90)

        # val1 = velocity, val2 = steering angle
        actions = {"agent_0": [0, 0], "agent_1": [0, 0]}
        dc = np.linalg.norm(env.world_size)
        alpha = -1 / ((2 * (2 - 1)) / 2 * dc)
        distance = 2 * 50
        beta = -(1e-3)
        act_pen_norm = np.linalg.norm(
            list(actions.values())
        )  # Shape [N, action length]

        exp_rew = alpha * distance + beta * act_pen_norm

        obs, rewards, terminations, truncations, infos = env.step(actions)
        expected_rewards = {"agent_0": exp_rew, "agent_1": exp_rew}
        self.assertEqual(rewards, expected_rewards)


if __name__ == "__main__":
    unittest.main()
