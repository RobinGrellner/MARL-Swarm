from envs.evader.evader_environment import EvaderEnvironment
from pettingzoo.test import parallel_api_test, parallel_seed_test
import unittest
import numpy as np


class TestEvaderEnvironment(unittest.TestCase):

    def test_pettingzoo_tests(self):
        env_fn = EvaderEnvironment
        parallel_api_test(env_fn(), num_cycles=1_000_000)
        parallel_seed_test(env_fn)
        # Die restlichen pettingzoo-Tests funktionieren nur bei AEC Environments

    def test_agent_num(self):
        env_fn = EvaderEnvironment
        env = env_fn(num_pursuers=10)
        self.assertEqual(len(env.possible_agents), 11)

        env_2 = env_fn(num_pursuers=1)
        self.assertEqual(len(env_2.possible_agents), 2)

    def test_init(self):
        env_fn = EvaderEnvironment
        with self.assertRaises(ValueError):
            env_fn(num_pursuers=-10)
        with self.assertRaises(ValueError):
            env_fn(num_pursuers=1.5)

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
        env = EvaderEnvironment(
            max_cycles=5,
            world_size=(100, 100),
            num_pursuers=2,
        )
        np.random.seed(100)
        env.reset()
        # Val1 = velocity, val2 = steering angle
        actions = {"pursuer_0": [5, 0], "pursuer_1": [0, 10]}

        purs_0_coord = env.agent_map["pursuer_0"].get_coords()
        purs_1_coord = env.agent_map["pursuer_1"].get_coords()

        env.step(actions)

        purs_0_coord_after = env.agent_map["pursuer_0"].get_coords()
        purs_1_coord_after = env.agent_map["pursuer_1"].get_coords()

        self.assertEqual((purs_0_coord[0] + 5, purs_0_coord[1]), purs_0_coord_after)
        self.assertEqual(purs_1_coord, purs_1_coord_after)

    def test_agent_movement_rotation_only(self):
        env = EvaderEnvironment(
            max_cycles=5,
            world_size=(100, 100),
            num_pursuers=2,
        )
        np.random.seed(100)
        env.reset()
        # val1 = velocity, val2 = steering angle
        actions = {"pursuer_0": [0, 10]}
        p_coord = env.agent_map["pursuer_0"].get_coords()
        p_vel_phi = env.agent_map["pursuer_0"].get_vel_phi()

        env.step(actions)

        p_coord_after = env.agent_map["pursuer_0"].get_coords()
        p_vel_phi_after = env.agent_map["pursuer_0"].get_vel_phi()

        self.assertEqual(
            p_coord,
            p_coord_after
        )

        self.assertEqual(
            (p_vel_phi[0], p_vel_phi[1] + 10),
            p_vel_phi_after
        )

if __name__ == "__main__":
    unittest.main()
