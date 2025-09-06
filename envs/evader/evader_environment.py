import functools

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
import pygame
from envs.evader.agents.pursuer_agent import Pursuer
from copy import copy
from pettingzoo.utils import parallel_to_aec, wrappers

FPS = 20


def env(
    self,
    num_pursuers=10,
    max_cycles=100,
    num_evaders=1,
    world_size=(10, 10),
    render_mode=None,
    logging=False,
):
    """
    Wrap into default wrappers
    """
    env = parallel_env(
        num_pursuers=num_pursuers,
        num_evaders=num_evaders,
        max_cycles=max_cycles,
        world_size=world_size,
        render_mode=render_mode,
        logging=logging,
    )
    return env


def raw_env(
    self,
    num_pursuers=10,
    max_cycles=100,
    num_evaders=1,
    world_size=(10, 10),
    render_mode=None,
    logging=False,
):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(
        num_pursuers=num_pursuers,
        max_cycles=max_cycles,
        world_size=world_size,
        num_evaders=num_evaders,
        render_mode=render_mode,
        logging=logging,
    )
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"name": "evader_env", "render_modes": ["human"]}

    def __init__(
        self,
        num_pursuers=10,
        num_evaders=1,
        max_cycles=100,
        world_size=(10, 10),
        render_mode=None,
        logging=False,
    ):
        """
        Initialization of the environment
        :param num_pursuers: Number of Catch-Agents
        :num_evaders: Number of Evade-Agents
        :param max_cycles: Maximum Cycles per episode
        :param world_size: Size od the 2D-World
        """
        # Sanity checks
        if num_pursuers <= 0 or not isinstance(num_pursuers, int):
            raise ValueError("Number of Pursuers must be > 0 and an Integer")
        if world_size[0] <= 0 or world_size[1] <= 0:
            raise ValueError("World size must be > 0")
        if max_cycles <= 0:
            raise ValueError("Max Cycles must be > 0")

        self.logging = logging
        self.num_pursuers = num_pursuers
        self.max_cycles = max_cycles
        self.world_size = world_size
        self.timesteps = 0
        self.render_mode = render_mode

        # Defining the agents [Evaders are not agents, but part of the environment]
        self.evader = Pursuer(f"evader", 0, 0, 0, 0, 10, 10)
        self.pursuers = [
            Pursuer(f"pursuer_{i}", 0, 0, 0, 0, 0, 0) for i in range(self.num_pursuers)
        ]
        self.agent_map = {agent.name: agent for agent in self.pursuers}
        self.possible_agents = [agent.name for agent in self.pursuers]

        # Pygame-Initialisation
        self.window_size = 600  # Fenstergröße für die Pygame-Darstellung
        self.screen = None
        self.clock = None

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

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Defines the observationspace for each agent:
        {agent_name:
            [own_velocity,
             own_orientation,
             own_closest_wall_dist,
             own_closest_wall_bearing,
             own_evader_distance,
             own_evader_bearing,
             for each other agent:
                [distance_to_agent,
                 bearing_to_agent,
                 relative_orientation,
                 relative_speed,
                ]
             ]
        } #TODO: Speed ist aktuell noch uncapped - Evtl. für den Agent cappen!
        """
        return spaces.Box(low=-np.inf, high=np.inf, shape=(46,))
        #     low=np.array([[-np.inf, -np.inf, 0, -np.inf, 0, -np.inf]
        #         + [0, -np.inf, -np.inf, -np.inf] * (self.num_pursuers)for i in range(self.num_pursuers + 1)]

        #     ).astype('float32'),
        #     high=np.array(
        #         [[
        #             np.inf,
        #             np.inf,
        #             np.sqrt(self.world_size[0] * self.world_size[1]),
        #             np.inf,
        #             np.sqrt(self.world_size[0] * self.world_size[1]),
        #             np.inf,
        #         ]
        #         + [
        #             np.sqrt(self.world_size[0] * self.world_size[1]),
        #             np.inf,
        #             np.inf,
        #             np.inf,
        #         ]
        #         * (self.num_pursuers) for i in range(self.num_pursuers + 1)]
        #     ).astype('float32'),
        #     shape=[self.num_pursuers + 1, 6 + (self.num_pursuers) * 4],
        # )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Defines the actionspace
            [vel_delta, phi_delta]
        with    -3 < vel_delta < 3
                -20 < phi_delta < -20
        """
        return spaces.Box(
            low=np.array([-3, -20]).astype("float32"),
            high=np.array([3, 20]).astype("float32"),
            shape=(2,),
        )

    def reset(self, seed=100, options=None):
        """
        Resets all parameters to starting position
        :return: Starting Positions of the agents
        """
        np.random.seed(seed)
        self.agents = copy(self.possible_agents)
        self.timesteps = 0

        for agent in self.possible_agents:
            self.agent_map[agent].set_coords(
                *np.random.uniform(low=(0, 0), high=self.world_size, size=(2))
            )

        self.evader.set_coords(
            *np.random.uniform(low=(0, 0), high=self.world_size, size=(2))
        )

        obs = self._observe()
        infos = {a: {} for a in self.possible_agents}
        return obs, infos

    def step(self, actions):
        """
        Executes one Step in time
        :param actions: Dictionary with actions for every agent
        :return: observations, rewards, Done-Signals and infos
        """
        self.timesteps += 1
        truncations = {agent: False for agent in self.agents}
        obs = {agent: [] for agent in self.agents}
        rewards = {agent: 0 for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        self._move_agents(actions)
        self._move_evaders()

        obs = self._observe()
        terminations = self._check_caught(catch_distance=5)

        # Überprüfe, ob die maximale Anzahl von Schritten erreicht wurde
        if self.timesteps >= self.max_cycles:
            truncations = {agent: True for agent in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def _move_agents(self, actions):
        for agent, action in actions.items():
            self.agent_map[agent].set_vel_w(action[0], action[1])
            self.agent_map[agent].move()
            self.agent_map[agent].set_coords(
                *np.clip(self.agent_map[agent].get_coords(), 0, self.world_size)
            )

    def _check_caught(self, catch_distance=5):
        terminations = {agent: False for agent in self.agents}
        evader_pos = self.evader.get_coords()

        catch_distance = 5
        for agent in self.agents:
            distance = self._calc_distance(
                self.agent_map[agent].get_coords(), evader_pos
            )

            if distance < catch_distance:
                # pursuer hat Evader gefangen
                rewards[self.evader.name] = -1  # Bestrafung für Evader
                rewards[agent] = 1  # Belohnung für den pursuer
                terminations = {agent: True for agent in self.agents}  # Episode endet
                continue
        return terminations

    def _move_evaders(self):
        # TODO Implement
        pass

    def render(self):
        """
        Renders environment with pygame.
        """
        if self.render_mode == None:
            return
        elif self.render_mode == "human":
            self.__render_setup()
            self.__render_pursuers()
            self.__render_evaders()

            pygame.display.flip()
            self.clock.tick(FPS)

    def __render_setup(self):
        """ "
        Sets up the pygame screen if not happened before.
        """
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((self.window_size, self.window_size))
            pygame.display.set_caption("Tag Environment")
            self.clock = pygame.time.Clock()
        self.screen.fill((255, 255, 255))

    def __render_pursuers(self):
        """
        Puts pursuer agents to the screen. Also shows their orientation.
        """
        scale = self.window_size / max(self.world_size)
        for agent in list(filter(lambda a: a != self.evader.name, self.agents)):
            pursuer_pos = (
                int(self.agent_map[agent].get_coords()[0] * scale),
                int(self.agent_map[agent].get_coords()[1] * scale),
            )
            bearing_rad = np.deg2rad(self.agent_map[agent].get_vel_phi()[1])
            pygame.draw.circle(self.screen, (255, 0, 0), pursuer_pos, 5)
            pygame.draw.line(
                self.screen,
                (255, 0, 0),
                pursuer_pos,
                (
                    pursuer_pos[0] + int(10 * np.cos(bearing_rad)),
                    pursuer_pos[1] + int(10 * np.sin(bearing_rad)),
                ),
            )

    def __render_evaders(self):
        """
        Puts evader agents to the screen. Also shows their orientation.
        """
        scale = self.window_size / max(self.world_size)
        evader_pos = (
            int(self.evader.get_coords()[0] * scale),
            int(self.evader.get_coords()[1] * scale),
        )
        pygame.draw.circle(self.screen, (0, 0, 255), evader_pos, 5)

    def close(self):
        pass


if __name__ == "__main__":
    env = parallel_env(
        num_pursuers=20, max_cycles=100, world_size=(200, 200), render_mode="human"
    )

    observations = env.reset()
    action = np.array([1, 9])

    for _ in range(env.max_cycles):
        actions = {
            agent: env.action_space(env.agents[4]).sample() for agent in env.agents
        }
        env.render()
        obs, rewards, terminations, truncations, infos = env.step(actions)
        if all(terminations.values()):
            break
    env.close()
