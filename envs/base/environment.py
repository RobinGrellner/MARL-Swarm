from pettingzoo import ParallelEnv
import numpy as np

class Environment(ParallelEnv):
    def _get_positions(self):
        return {
            self.agent_map[agent].name: self.agent_map[agent].get_coords()
            for agent in self.agents
        }

    def _calc_distance(self, coord_1, coord_2):
        return np.linalg.norm(np.subtract(coord_1, coord_2))

    def _calc_bearing(self, coord_1, coord_2, phi_1):
        # If Arctan(x) with x -> inf: 90 DEG
        if coord_2[0] == coord_1[0]:
            return 90 - phi_1
        return (
            np.rad2deg(
                np.arctan(np.divide(coord_2[1] - coord_1[1], coord_2[0] - coord_1[0]))
            )
            - phi_1
        )

    def _calc_rel_orientation(self, coord_1, coord_2, phi_2):
        # If Arctan(x) with x -> inf: 90 DEG
        if coord_2[0] == coord_1[0]:
            return 90 - phi_2
        return (
            np.rad2deg(
                np.arctan(np.divide(coord_1[1] - coord_2[1], coord_1[0] - coord_2[0]))
            )
            - phi_2
        )

    def _calc_rel_vel(self, vel_1, vel_2, phi_1, phi_2):
        # Transformation into carthesian coords and magnitude calculation
        deg_phi_1 = np.deg2rad(phi_1)
        deg_phi_2 = np.deg2rad(phi_2)

        return np.sqrt(
            np.square(vel_1 * np.cos(deg_phi_1) - vel_2 * np.cos(deg_phi_2))
            + np.square(vel_1 * np.sin(deg_phi_1) - vel_2 * np.sin(deg_phi_2))
        )

    def _calc_dist_to_closest_wall(self, coord):
        return np.min(
            [
                coord[0],
                coord[1],
                self.world_size[0] - coord[0],
                self.world_size[1] - coord[1],
            ]
        )

    def _bearing_to_closest_wall(self, coords, phi):
        x = coords[0]
        y = coords[1]
        if (  # Left side is closest
            x < self.world_size[0] - x and x < y and x < self.world_size[1] - y
        ):
            return phi
        elif (  # Right side is closest
            self.world_size[0] - x < x
            and self.world_size[0] - x < y
            and self.world_size[0] - x < self.world_size[1] - y
        ):
            return phi - 180
        elif (  # Top side is closest
            y < x and y < self.world_size[0] - x and y < self.world_size[1] - y
        ):
            return phi - 90
        elif (  # Bottom side is closest
            self.world_size[1] - y < x
            and self.world_size[0] - y < y
            and self.world_size[0] - y < self.world_size[0] - x
        ):
            return phi - 270
        return

    def _observe(self):
        obs = {agent: [] for agent in self.agents}

        evader_pos = self.evader.get_coords()
        for agent1 in self.agents:
            a_1 = self.agent_map[agent1]
            obs_values = []
            # observations
            obs_values.append(a_1.get_vel_phi()[0])  # velocity
            obs_values.append(a_1.get_vel_phi()[1])  # orientation
            obs_values.append(
                self._calc_dist_to_closest_wall(a_1.get_coords())
            )  # distance to closest wall
            obs_values.append(
                self._bearing_to_closest_wall(a_1.get_coords(), a_1.get_vel_phi()[1])
            )  # absolute bearing to nearest wall
            obs_values.append(
                self._calc_distance(a_1.get_coords(), evader_pos)
            )  # distance to evader
            obs_values.append(
                self._calc_bearing(
                    a_1.get_coords(),
                    evader_pos,
                    a_1.get_vel_phi()[1],
                )
            )  # bearing to evader

            # observing other agents
            for agent2 in self.agents:
                if agent1 == agent2:
                    continue
                a_2 = self.agent_map[agent2]

                agent_dist = self._calc_distance(
                    a_1.get_coords(),
                    a_2.get_coords(),
                )

                agent_bearing = self._calc_bearing(
                    a_1.get_coords(),
                    a_2.get_coords(),
                    a_1.get_vel_phi()[1],
                )

                agent_rel_orientation = self._calc_rel_orientation(
                    a_1.get_coords(),
                    a_2.get_coords(),
                    a_2.get_vel_phi()[1],
                )

                agent_rel_vel = self._calc_rel_vel(
                    a_1.get_vel_phi()[0],
                    a_2.get_vel_phi()[0],
                    a_1.get_vel_phi()[1],
                    a_2.get_vel_phi()[1],
                )
                obs_values.extend(
                    [agent_dist, agent_bearing, agent_rel_orientation, agent_rel_vel]
                )
                obs[agent1] = np.array(obs_values, dtype=np.float32)
        return obs