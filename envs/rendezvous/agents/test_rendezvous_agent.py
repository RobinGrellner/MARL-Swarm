from envs.evader.agents.pursuer_agent import Pursuer
import unittest
import numpy as np


class TestRendezvous(unittest.TestCase):
    def test_move(self):
        p = Pursuer("p", 10, 10, 0, 0, 10, 10)

        p_coord = p.get_coords()
        p.move()
        p_coord_new = p.get_coords()

        self.assertEqual(p_coord, p_coord_new, "Agent should not Move")

        vel = 4
        phi = 90  # Turning the agent to face to the (true) right
        p.set_vel_w(vel, phi)

        p_coord = p.get_coords()
        p.move()
        p_coord_new = p.get_coords()

        self.assertEqual(
            (
                p_coord[0],
                p_coord[1] + 4,
            ),
            p_coord_new,
            "Agent should have moved by the correct amount"
        )
        ##TODO Test negative velocity and angle


if __name__ == "__main__":
    unittest.main()
