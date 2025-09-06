import numpy as np
from gymnasium import spaces


# TODO Implement double integrator/single integrator case
class Agent:
    def __init__(
        self,
        name,
        x_coord,
        y_coord,
        vel,
        phi,
        comm_radius,
        obs_radius,
        movement_mode="single_integrator",
        obs_mode="global",
    ):
        # Sanity checks
        if movement_mode not in ["single_integrator", "double_integrator"]:
            print(f"Movement type for Agent: {name} unknown.")
            return
        self.name = name
        self.x_coord = x_coord
        self.y_coord = y_coord
        self.vel = vel
        self.phi = phi
        self.w = 0
        self.comm_radius = comm_radius
        self.obs_radius = obs_radius
        self.movement_mode = movement_mode
        self.obs_mode = obs_mode

    def observe(
        self,
        active_agents,
        agent_map,
        world_size,
    ):
        obs = {}
        obs_local = [
            self.__calc_dist_to_closest_wall(world_size),
            self.__bearing_to_closest_wall(world_size=world_size),
        ]
        if self.obs_mode == "global":
            obs_other_agents = []
            for agent in active_agents:
                if agent == self.name:
                    continue
                obs_other_agents.append(self.__calc_distance(agent_map[agent]))
                obs_other_agents.append(self.__calc_bearing(agent_map[agent]))
            obs = obs_local + obs_other_agents
        elif self.obs_mode == "global_extended":
            pass
        elif self.obs_mode == "local":
            pass
        elif self.obs_mode == "local_extended":
            pass
        elif self.obs_mode == "local_comm":
            pass
        return obs

    def __calc_distance(self, other_agent):
        return np.linalg.norm(np.subtract(self.get_coords(), other_agent.get_coords()))

    def __calc_bearing(self, other_agent):
        # If Arctan(x) with x -> inf: 90 DEG
        if other_agent.x_coord == self.x_coord:
            return np.deg2rad(90) - self.phi
        return (
            np.arctan(
                np.divide(
                    other_agent.y_coord - self.y_coord,
                    other_agent.x_coord - self.x_coord,
                )
            )
            - self.phi
        )

    def __calc_rel_orientation(self, coord_1, coord_2, phi_2):
        # If Arctan(x) with x -> inf: 90 DEG
        if coord_2[0] == coord_1[0]:
            return np.deg2rad(90) - phi_2
        return (
            np.arctan(np.divide(coord_1[1] - coord_2[1], coord_1[0] - coord_2[0]))
            - phi_2
        )

    def __calc_rel_vel(self, vel_1, vel_2, phi_1, phi_2):
        # Transformation into carthesian coords and magnitude calculation

        return np.sqrt(
            np.square(vel_1 * np.cos(phi_1) - vel_2 * np.cos(phi_2))
            + np.square(vel_1 * np.sin(phi_1) - vel_2 * np.sin(phi_2))
        )

    def __calc_dist_to_closest_wall(self, world_size):
        return np.min(
            [
                self.x_coord,
                self.y_coord,
                world_size[0] - self.x_coord,
                world_size[1] - self.y_coord,
            ]
        )

    def __bearing_to_closest_wall(self, world_size):
        if (  # Left side is closest
            self.x_coord < world_size[0] - self.x_coord
            and self.x_coord < self.y_coord
            and self.x_coord < world_size[1] - self.y_coord
        ):
            return self.phi
        elif (  # Right side is closest
            world_size[0] - self.x_coord < self.x_coord
            and world_size[0] - self.x_coord < self.y_coord
            and world_size[0] - self.x_coord < world_size[1] - self.y_coord
        ):
            return self.phi - np.deg2rad(180)
        elif (  # Top side is closest
            self.y_coord < self.x_coord
            and self.y_coord < world_size[0] - self.x_coord
            and self.y_coord < world_size[1] - self.y_coord
        ):
            return self.phi - np.deg2rad(90)
        elif (  # Bottom side is closest
            world_size[1] - self.y_coord < self.x_coord
            and world_size[0] - self.y_coord < self.y_coord
            and world_size[0] - self.y_coord < world_size[0] - self.x_coord
        ):
            return self.phi - np.deg2rad(270)
        return 0

    def move(self):
        # Calculate deltas
        d_x = self.vel * np.cos(self.phi)
        d_y = self.vel * np.sin(self.phi)
        d_phi = self.w

        # Apply movestep
        self.x_coord += d_x
        self.y_coord += d_y
        self.phi += d_phi

        # Keep phi within [0, 2π)
        self.phi = self.phi % (2 * np.pi)

    def set_coords(self, x, y):
        self.x_coord = x
        self.y_coord = y

    def get_coords(self):
        return (self.x_coord, self.y_coord)

    def set_vel_w(self, vel, w):
        self.vel = vel
        self.w = w

    def get_vel_phi(self):
        return (self.vel, self.phi)
