from pursuer_agent import Pursuer
import unittest
import numpy as np


class TestPursuer(unittest.TestCase):
    def test_move(self):
        p = Pursuer("p", 10, 10, 0, 0, 10, 10)

        p_coord = p.get_coords()
        p.move()
        p_coord_new = p.get_coords()

        self.assertEqual(p_coord, p_coord_new)

        vel = 4
        phi = 12
        p.set_vel_phi(vel, phi)

        p_coord = p.get_coords()
        p.move()
        p_coord_new = p.get_coords()

        self.assertEqual(
            (
                p_coord[0] + vel * np.cos(np.deg2rad(phi)),
                p_coord[1] + vel * np.sin(np.deg2rad(phi)),
            ),
            p_coord_new,
        )


if __name__ == "__main__":
    unittest.main()
