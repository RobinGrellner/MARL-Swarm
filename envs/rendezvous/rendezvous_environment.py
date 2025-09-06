import functools

import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
import pygame
from envs.rendezvous.agents.rendezvous_agent import RendezvousAgent
from pettingzoo.utils import parallel_to_aec

FPS = 10


def env(
    max_cycles=100,
    world_size=(10, 10),
    render_mode=None,
    logging=False,
    num_agents=20,
):
    """
    Wrap into default wrappers
    """
    env = parallel_env(
        max_cycles=max_cycles,
        world_size=world_size,
        render_mode=render_mode,
        logging=logging,
        num_agents=num_agents,
    )
    return env


def raw_env(
    max_cycles=100,
    world_size=(10, 10),
    num_agents=20,
    render_mode=None,
    logging=False,
):
    """
    To support the AEC API, the raw_env() function just uses the from_parallel
    function to convert from a ParallelEnv to an AEC env
    """
    env = parallel_env(
        max_cycles=max_cycles,
        world_size=world_size,
        render_mode=render_mode,
        logging=logging,
        num_agents=num_agents,
    )
    env = parallel_to_aec(env)
    return env


class parallel_env(ParallelEnv):
    metadata = {"name": "rendezvous_env", "render_modes": ["human"]}

    def __init__(
        self,
        num_agents=20,
        observation_radius=50,
        observation_mode="global",
        max_cycles=100,
        world_size=(10, 10),
        render_mode=None,
        logging=False,
    ):
        """
        Initialization of the environment
        :param num_agents: Number of agents
        :param observation_radius: Radius in which the agents sense other agents
        :param observation_mode: Determines how much info agents observe about other agents
        :param max_cycles: Maximum cycles per episode
        :param world_size: Size od the 2D-World
        :param render_mode: Determines if and how the environment should be rendered
        :param logging: logging for agent positions
        """
        # Sanity checks
        if num_agents < 0 or not isinstance(num_agents, int):
            raise ValueError("Number of Agents must be > 0 and an Integer")
        if world_size[0] <= 0 or world_size[1] <= 0:
            raise ValueError("World size must be > 0")
        if max_cycles <= 0:
            raise ValueError("Max Cycles must be > 0")

        self.logging = logging
        self.num_r_agents = num_agents
        self.max_cycles = max_cycles
        self.world_size = world_size
        self.timesteps = 0
        self.render_mode = render_mode
        self.observation_radius = observation_radius
        self.observation_mode = observation_mode

        # Initializing Agents
        self.agents = [f"agent_{i}" for i in range(self.num_r_agents)]
        self.agent_map = {
            agent: RendezvousAgent(agent, 0, 0, 0, 0, 0, 0, obs_mode=observation_mode)
            for agent in self.agents
        }
        self.possible_agents = [agent for agent in self.agents]

        # Pygame-Initialisation
        self.window_size = 600
        self.screen = None
        self.clock = None

    def __observe(self):
        obs = {agent: [] for agent in self.agents}
        if self.observation_mode == "global":
            obs = {
                agent: self.agent_map[agent].observe(
                    self.agents, self.agent_map, self.world_size
                )
                for agent in self.agents
            }
        elif self.observation_mode == "global_extended":
            pass
        elif self.observation_mode == "local":
            pass
        elif self.observation_mode == "local_extended":
            pass
        elif self.observation_mode == "local_comm":
            pass
        return obs

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        """
        Defines the observation space for each agent. If there is agents not in view, zero padding is applied.
        Observation spaces depend on the observaition mode:
        :global: Per neighbor [distance, bearing],
                local [distance to nearest wall, bearing to nearest wall]
        :global_extended: Per neighbor [distance, bearing, relative orientation, relative velocity],
                local [distance to nearest wall, bearing to nearest wall]
        :local: Per neighbor [distance, bearing],
                local [distance to nearest wall, bearing to nearest wall]
        :local_extended: Per neighbor [distance, bearing, relative orientation],
                local [distance to nearest wall, bearing to nearest wall]
        :local_comm: Per neighbor [distance, bearing, relative orientation, neighborhood size],
                local [distance to nearest wall, bearing to nearest wall, own neighborhood size]
        """
        return self.agent_map[agent].observation_space(self.possible_agents)

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Defines the actionspace
            [vel, angular_momentum]
        with    -3 < vel < 3
                -20 < angular_momentum < 360
        """
        return spaces.Box(
            low=np.array([-3, np.deg2rad(-40)]).astype("float32"),
            high=np.array([3, np.deg2rad(40)]).astype("float32"),
            shape=(2,),
        )

    def reset(self, seed=100, options=None):
        """
        Resets all parameters to starting position
        :return: Starting Positions of the agents
        """
        np.random.seed(seed)
        self.agents = [f"agent_{i}" for i in range(self.num_r_agents)]
        self.timesteps = 0

        for agent in self.possible_agents:
            self.agent_map[agent].set_coords(
                *np.random.uniform(low=(0, 0), high=self.world_size, size=(2))
            )
            orientation = np.random.uniform(low=(0), high=(np.pi * 2))
            self.agent_map[agent].phi = orientation

        obs = self.__observe()
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

        self.__move_agents(actions)

        obs = self.__observe()
        terminations = {agent: False for agent in self.agents}
        reward = self.__calc_reward(actions)
        rewards = {agent: reward for agent in self.agents}

        if self.timesteps >= self.max_cycles:
            truncations = {agent: True for agent in self.agents}

        distance_sum = self.__calc_distance_sum(self.agents[0])
        if distance_sum <= 2:
            terminations = {agent: True for agent in self.agents}

        if any(terminations.values()) or all(truncations.values()):
            self.agents = []
        return obs, rewards, terminations, truncations, infos

    def __calc_distance_sum(self, agent_self):
        sum = 0
        for agent in self.agents:
            if agent == agent_self:
                continue
            sum += np.linalg.norm(
                np.subtract(
                    self.agent_map[agent_self].get_coords(),
                    self.agent_map[agent].get_coords(),
                )
            )
        return sum

    def __calc_reward(self, actions):
        dc = np.linalg.norm(self.world_size)
        n = len(self.agents)
        alpha = -1 / ((n * (n - 1)) / 2 * dc)
        beta = -(1e-3)
        act_pen_norm = np.linalg.norm(
            list(actions.values())
        )  # Shape [N, action length]
        sum = 0
        for agent in self.agents:
            sum += self.__calc_distance_sum(agent)
        return alpha * sum + beta * act_pen_norm

    def __move_agents(self, actions):
        for agent, action in actions.items():
            self.agent_map[agent].set_vel_w(action[0], action[1])
            self.agent_map[agent].move()
            self.agent_map[agent].set_coords(
                *np.clip(self.agent_map[agent].get_coords(), 0, self.world_size)
            )

    def render(self):
        """
        Renders environment with pygame.
        """
        if self.render_mode == None:
            return
        elif self.render_mode == "human":
            self.__render_setup()
            self.__render_pursuers()
            text_surface = self.my_font.render(
                f"Steps: {self.timesteps}", False, (0, 0, 0)
            )
            self.screen.blit(text_surface, (0, 0))
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
            self.my_font = pygame.font.SysFont("Arial", 30)
        self.screen.fill((255, 255, 255))

    def __render_pursuers(self):
        """
        Puts pursuer agents to the screen. Also shows their orientation.
        """
        scale = self.window_size / max(self.world_size)
        for agent in self.agents:
            agent_pos = (
                int(self.agent_map[agent].get_coords()[0] * scale),
                int(self.agent_map[agent].get_coords()[1] * scale),
            )
            bearing_rad = self.agent_map[agent].get_vel_phi()[1]
            pygame.draw.circle(self.screen, (100, 100, 255), agent_pos, 5)
            if self.observation_mode not in ["global", "global_extended"]:
                pygame.draw.circle(
                    self.screen,
                    (190, 190, 255, 0.5),
                    agent_pos,
                    self.observation_radius,
                    width=1,
                )
            pygame.draw.line(
                self.screen,
                (255, 0, 0),
                agent_pos,
                (
                    agent_pos[0] + int(10 * np.cos(bearing_rad)),
                    agent_pos[1] + int(10 * np.sin(bearing_rad)),
                ),
            )

    def close(self):
        pass


if __name__ == "__main__":
    env = parallel_env(
        max_cycles=100, world_size=(200, 200), render_mode="human", num_agents=3
    )

    observations = env.reset()

    for _ in range(env.max_cycles):
        actions = {agent: [-1.4369668, -0.6981317] for agent in env.agents}
        env.render()
        obs, rewards, terminations, truncations, infos = env.step(actions)
        print("Observations: ", obs)
        if all(terminations.values()):
            break
    env.close()

    # env = raw_env(max_cycles=100, world_size=(200, 200), render_mode="human")
    # #env.render = parallel_env.render
    # observations = env.reset()

    # print("Start")
    # for agent in env.agent_iter():
    #     observation, reward, termination, truncation, info = env.last()

    #     if termination or truncation:
    #         action = None
    #     else:
    #         # invalid action masking is optional and environment-dependent
    #         if "action_mask" in info:
    #             mask = info["action_mask"]
    #         elif isinstance(observation, dict) and "action_mask" in observation:
    #             mask = observation["action_mask"]
    #         else:
    #             mask = None
    #         action = env.action_space(agent).sample(mask) # this is where you would insert your policy
    #     env.render()
    #     env.step(action)
    # print("End")
    # env.close()
