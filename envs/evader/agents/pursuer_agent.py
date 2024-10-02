from gymnasium import spaces
import numpy as np

class Pursuer:
    def __init__(
        self,
        name,
        x_coord,
        y_coord,
        vel,
        phi,
        comm_radius,
        obs_radius,
    ):
        self.name = name
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.vel = vel
        self.phi = phi
        self.comm_radius = comm_radius
        self.obs_radius = obs_radius

    def action_space(self):
        """
        Action space definition
            [vel_delta, phi_delta]
        with    -3 < vel_delta < 3
                -20 < phi_delta < -20
        """
        return spaces.Box(
            low=np.array(-3, -20), high=np.array(3, 20), shape=(2,), dtype=np.float64
        )

    def observation_space(self):
        """
        Observation space definition
            [
                [self_pos_x, self_pos_y],
                [evader]
            ]
        """
        pass

    def move(self):
        steer_deg = np.deg2rad(self.phi)
        d_x = self.vel * np.cos(steer_deg)
        d_y = self.vel * np.sin(steer_deg)

        self.x_coord = self.x_coord + d_x
        self.y_coord = self.y_coord + d_y

    def set_coords(self, x, y):
        self.x_coord = x
        self.y_coord = y

    def get_coords(self):
        return (self.x_coord, self.y_coord)
    
    def set_vel_phi(self, vel, phi):
        self.vel = vel
        self.phi = phi
    
    def get_vel_phi(self):
        return (self.vel, self.phi)