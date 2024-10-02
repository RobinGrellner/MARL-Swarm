import functools
import numpy as np
from pettingzoo import ParallelEnv
from gymnasium import spaces
import pygame
import time
from agents.pursuer_agent import Pursuer
from copy import copy


class EvaderEnvironment(ParallelEnv):
    metadata = {"name": "evader_env", "render_modes": ["human", "text"]}

    def __init__(
        self,
        num_pursuers=3,
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
        #Sanity checks
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

        # Defining the agents
        self.evader = Pursuer(f"evader", 0, 0, 0, 0, 10, 10)
        self.pursuers = [
            Pursuer(f"pursuer_{i}", 0, 0, 0, 0, 0, 0)
            for i in range(self.num_pursuers)
        ]
        self.agent_map = {agent.name: agent for agent in self.pursuers}
        self.agent_map[self.evader.name] = self.evader
        self.possible_agents = [self.evader.name] + [agent.name for agent in self.pursuers]

        # Pygame-Initialisation
        self.window_size = 600  # Fenstergröße für die Pygame-Darstellung
        self.screen = None
        self.clock = None

    def _get_positions(self):
        return {self.agent_map[agent].name: self.agent_map[agent].get_coords() for agent in self.agents}

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        # TODO richtigen Observationspace definieren (evtl. mit Sequence)
        """
        Defines the observationspace
        Observationspace = Positions of all Agents
        """
        return spaces.Box(
            low=np.array([0, 0]), high=np.array(self.world_size), dtype=np.float32
        )

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        """
        Defines the actionspace
            [vel_delta, phi_delta]
        with    -3 < vel_delta < 3
                -20 < phi_delta < -20
        """
        return spaces.Box(
            low=np.array([-3, -20]),
            high=np.array([3, 20]),
            shape=(2,),
            dtype=np.float64,
        )

    def reset(self, seed=100, options=None):
        """
        Resets all parameters to starting position
        :return: Starting Positions of the agents
        """
        np.random.seed(seed)
        self.agents = self.possible_agents[:]
        self.timesteps = 0

        for agent in self.possible_agents:
            self.agent_map[agent].set_coords(
                *np.random.uniform(low=(0, 0), high=self.world_size, size=(2))
            )

        obs = self._get_positions()
        infos = {a: {} for a in self.possible_agents}
        return obs, infos

    def step(self, actions):
        """
        Executes one Step in time
        :param actions: Dictionary with actions for every agent
        :return: observations, rewards, Done-Signals and infos
        """
        self.timesteps += 1
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: False for agent in self.agents}
        rewards = {agent: 0 for agent in self.agents}
        infos = {agent: {} for agent in self.agents}

        # Bewegung der Agenten basierend auf den Aktionen (evader und pursuer)
        for agent, action in actions.items():
            # Aktualisiere Position basierend auf der Aktion (Richtung + Geschwindigkeit)
            self.agent_map[agent].set_vel_phi(action[0], action[1])
            self.agent_map[agent].move()

            # Stelle sicher, dass sich die Agenten innerhalb der Grenzen des 2D-Raums bewegen
            self.agent_map[agent].set_coords(*np.clip(self.agent_map[agent].get_coords(), 0, self.world_size))

        # Berechne den Abstand zwischen dem Evader und allen pursuern
        evader_pos = self.agent_map[self.evader.name].get_coords()

        # Wenn ein pursuer den Evader fängt (d.h. der Abstand ist kleiner als ein bestimmter Wert)
        catch_distance = 5
        obs = self._get_positions()
        for agent in self.agents:
            if agent == self.evader.name:
                continue
            distance = np.linalg.norm(
                np.subtract(self.agent_map[agent].get_coords(), evader_pos)
            )
            if distance < catch_distance:
                # pursuer hat Evader gefangen
                rewards[self.evader.name] = -1  # Bestrafung für Evader
                rewards[agent] = 1  # Belohnung für den pursuer
                terminations = {agent: True for agent in self.agents}  # Episode endet
                self.agents = []
                continue
        # Überprüfe, ob die maximale Anzahl von Schritten erreicht wurde
        if self.timesteps >= self.max_cycles:
            truncations = {agent: True for agent in self.agents}
            self.agents = []

        return obs, rewards, terminations, truncations, infos

    def render(self):
        """
        Rendert die Umgebung mit Pygame.
        """
        if self.render_mode == None:
            return
        elif self.render_mode == "human":
            if self.screen is None:
                pygame.init()
                self.screen = pygame.display.set_mode(
                    (self.window_size, self.window_size)
                )
                pygame.display.set_caption("Tag Environment")
                self.clock = pygame.time.Clock()

            # Färbe den Hintergrund weiß
            self.screen.fill((255, 255, 255))

            # Skaliere die Positionen der Agenten auf das Fenster
            scale = self.window_size / max(self.world_size)

            # Render den Evader (blauer Kreis)
            evader_pos = (
                int(self.agent_map[self.evader.name].get_coords()[0] * scale),
                int(self.agent_map[self.evader.name].get_coords()[1] * scale),
            )

            pygame.draw.circle(self.screen, (0, 0, 255), evader_pos, 5)

            # Render die pursuers (rote Kreise)
            for agent in list(filter(lambda a : a != self.evader.name, self.agents)):

                pursuer_pos = (
                    int(self.agent_map[agent].get_coords()[0] * scale),
                    int(self.agent_map[agent].get_coords()[1] * scale),
                )
                pygame.draw.circle(self.screen, (255, 0, 0), pursuer_pos, 5)

            # Aktualisiere das Display
            pygame.display.flip()

            # Begrenze die Framerate
            self.clock.tick(60)


if __name__ == "__main__":
    # Beispiel, wie die Umgebung verwendet werden kann
    env = EvaderEnvironment(
        num_pursuers=10, max_cycles=100, world_size=(200, 200), render_mode="human"
    )
    observations = env.reset()
    action = np.array([1, 9])

    for _ in range(env.max_cycles):
        # Zufällige Aktionen für jeden Agenten
        #action = env.action_space(env.agents[4]).sample()

        actions = {agent: action for agent in env.agents}
        env.render()
        observations, rewards, terminations, truncations, infos = env.step(actions)
        #print(observations)
        time.sleep(0.1)
        

        # Beende die Schleife, wenn die Episode vorbei ist
        if all(terminations.values()):
            break

    env.close()
