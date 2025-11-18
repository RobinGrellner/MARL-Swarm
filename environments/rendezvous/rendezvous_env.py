from environments.base.base_environment import BaseEnv
from environments.rendezvous.observations_vectorized import compute_observations_vectorized
from typing import Dict, List, Optional
import numpy as np
import math
import pygame
from gymnasium import spaces


class RendezvousEnv(BaseEnv):
    """Rendezvous task for homogeneous swarms.

    Each agent receives observations about the nearest wall, optionally
    information about its neighbours (distance, bearing, relative
    orientation, relative velocity, neighbour count) and a binary mask
    indicating which neighbour slots contain valid data.  The reward is
    shared among all agents and penalises the sum of pairwise distances
    as well as the magnitude of the control signals.
    """
    metadata = {"name": "rendezvous_env", "render_modes": ["human"]}

    def __init__(
        self,
        *,
        num_agents: int,
        world_size: float,
        max_steps: int = 1000,
        csv_log_path: Optional[str] = None,
        torus: bool = False,
        kinematics: str = "single",
        v_max: float = 1.0,
        omega_max: float = 1.0,
        acc_v_max: float = 1.0,
        acc_omega_max: float = 1.0,
        obs_model: str = "classic",
        comm_radius: Optional[float] = None,
        render_mode: Optional[str] = "",
        fps: int = 60,
        break_distance_threshold: Optional[float] = None,
        max_agents: Optional[int] = None,
    ):
        # Store rendezvous-specific parameters needed before base init
        self.csv_log_path = csv_log_path
        self.obs_model = obs_model.lower() if obs_model is not None else "global_basic"
        self.comm_radius = comm_radius
        self.break_distance_threshold = break_distance_threshold
        self.max_agents = max_agents if max_agents is not None else num_agents

        # Initialize base environment
        super().__init__(
            num_agents=num_agents,
            world_size=world_size,
            torus=torus,
            kinematics=kinematics,
            v_max=v_max,
            omega_max=omega_max,
            acc_v_max=acc_v_max,
            acc_omega_max=acc_omega_max,
            max_steps=max_steps,
            render_mode=render_mode,
        )

        # Precompute constants for reward
        n = num_agents
        self.dc: float = world_size
        self.alpha: float = -1.0 / ((n * (n - 1) / 2.0) * self.dc)
        self.beta: float = -1e-3

        # Default comm_radius == world size
        if self.comm_radius is None:
            self.comm_radius = world_size

        # Precompute an array of indices for neighbour iteration.
        self._neighbour_indices: List[List[int]] = [
            [j for j in range(self.agent_handler.num_agents) if j != i]
            for i in range(self.agent_handler.num_agents)
        ]

        # Pygame-Initialisation
        self.window_size = 600
        self.screen = None
        self.clock = None
        self.fps = fps

    # Methods for setup
    def _get_observation_space(self):
        """Define an observation space for each agent.
        Depending on the obs_model, the observation contains a block of
        local features, followed by zero or more neighbour feature blocks and
        a mask.
        """
        obs_spaces: Dict[str, spaces.Box] = {}
        num_agents = self.agent_handler.num_agents
        world_size = self.world_size
        kin = self.agent_handler.kinematics

        if self.obs_model == "classic":
            # own position (x,y), mean position (x,y), linear velocity, angular velocity (optional), orientation
            base_dim = 6 if kin == "single" else 7
            pos_low = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
            pos_high = np.array(
                [world_size, world_size, world_size, world_size], dtype=np.float32
            )
            v_low = np.array([-self.agent_handler.v_max], dtype=np.float32)
            v_high = np.array([self.agent_handler.v_max], dtype=np.float32)
            if kin == "single":
                w_low = np.array([], dtype=np.float32)
                w_high = np.array([], dtype=np.float32)
            else:
                w_low = np.array([-self.agent_handler.omega_max], dtype=np.float32)
                w_high = np.array([self.agent_handler.omega_max], dtype=np.float32)
            ori_low = np.array([-math.pi], dtype=np.float32)
            ori_high = np.array([math.pi], dtype=np.float32)
            obs_low = np.concatenate([pos_low, v_low, w_low, ori_low], dtype=np.float32)
            obs_high = np.concatenate(
                [pos_high, v_high, w_high, ori_high], dtype=np.float32
            )

            # "Save" dimensions
            self._local_feature_dim = base_dim
            self._neighbour_feature_dim = 0
            self._max_neighbours = 0
            self.obs_total_dim = base_dim
            for name in self.agent_names:
                obs_spaces[name] = spaces.Box(
                    low=obs_low, high=obs_high, shape=(base_dim,), dtype=np.float32
                )
            return obs_spaces

        # Determine local and neighbour feature dimensions
        if self.obs_model in {"global_basic", "local_basic"}:
            neighbour_feature_dim = 2  # distance, bearing
            local_feature_dim = 2  # distance to wall, bearing to wall
        elif self.obs_model == "global_extended":
            neighbour_feature_dim = (
                5  # dist, bearing, relative orientation, rel vel x, rel vel y
            )
            local_feature_dim = 2
        elif self.obs_model == "local_extended":
            neighbour_feature_dim = 3  # dist, bearing, relative orientation
            local_feature_dim = 2
        elif self.obs_model == "local_comm":
            neighbour_feature_dim = (
                4  # dist, bearing, relative orientation, neighbour size
            )
            local_feature_dim = (
                3  # distance to wall, bearing to wall, own neighbourhood size
            )
        else:
            raise ValueError(f"Unknown observation model: {self.obs_model}")

        # Save dimensions
        self._neighbour_feature_dim = neighbour_feature_dim
        self._local_feature_dim = local_feature_dim
        # Use max_agents to size observations for scale-invariant learning
        self._max_neighbours = self.max_agents - 1

        # Total observation length: local + neighbour*max_neighbours + mask
        self.obs_total_dim = (
            local_feature_dim
            + self._max_neighbours * neighbour_feature_dim
            + self._max_neighbours
        )

        # Build bounds: mask entries in [0,1], others unbounded
        low = -np.inf * np.ones(self.obs_total_dim, dtype=np.float32)
        high = np.inf * np.ones(self.obs_total_dim, dtype=np.float32)
        mask_start = local_feature_dim + self._max_neighbours * neighbour_feature_dim
        low[mask_start:] = 0.0
        high[mask_start:] = 1.0
        for name in self.agent_names:
            obs_spaces[name] = spaces.Box(
                low=low, high=high, shape=(self.obs_total_dim,), dtype=np.float32
            )
        # Expose a layout descriptor used by custom feature extractors.  This
        # dictionary encodes the positions of the local features, neighbour
        # feature block and mask within each agent's observation vector.  It
        # also records the number of neighbours and dimensionality of each
        # feature block.  Training-Skripts can use this attribute to slice observations 
        # without hard‑coding the sizes here.
        self.obs_layout = {
            "local_dim": self._local_feature_dim,
            "neigh_dim": self._neighbour_feature_dim,
            "max_neighbours": self._max_neighbours,
            "total_dim": self.obs_total_dim,
        }
        return obs_spaces

    # ------------------------------------------------------------------
    # Reset and helper methods
    # ------------------------------------------------------------------
    def _reset_agents(self) -> None:
        """Reset agents to random positions and zero velocities."""
        self.agent_handler.initialize_random_positions(self.world_size)

    def delta(self, pos_i: np.ndarray, pos_j: np.ndarray) -> np.ndarray:
        """Compute minimal displacement from pos_i to pos_j (torus aware)."""
        diff = pos_j - pos_i
        if self.torus:
            half = self.world_size / 2.0
            if diff[0] > half:
                diff[0] -= self.world_size
            elif diff[0] < -half:
                diff[0] += self.world_size
            if diff[1] > half:
                diff[1] -= self.world_size
            elif diff[1] < -half:
                diff[1] += self.world_size
        return diff

    # ------------------------------------------------------------------
    # Observations (Vectorized)
    # ------------------------------------------------------------------
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Construct observations for all agents using vectorized implementation."""
        return compute_observations_vectorized(
            positions=self.agent_handler.positions,
            orientations=self.agent_handler.orientations,
            linear_vels=self.agent_handler.linear_vels,
            angular_vels=self.agent_handler.angular_vels,
            agent_names=self.agent_names,
            obs_model=self.obs_model,
            kinematics=self.agent_handler.kinematics,
            world_size=self.world_size,
            torus=self.torus,
            comm_radius=self.comm_radius,
            max_neighbours=self._max_neighbours,
            neighbour_feature_dim=self._neighbour_feature_dim,
        )

    # ------------------------------------------------------------------
    # Reward computation
    # ------------------------------------------------------------------
    def _calculate_rewards(
        self, actions: Optional[Dict[str, np.ndarray]]
    ) -> Dict[str, float]:
        """Compute global reward shared by all agents.

        Reward = α * sum_{i<j} min(d_ij, dc) + β * sum_i ||a_i||
        (Appendix E.1).  The normalisation ensures that the return
        magnitude is approximately in [−1, 0].  A small negative
        contribution proportional to the magnitude of the agents' actions
        discourages unnecessarily large control inputs.
        """
        # Sum of clipped distances (vectorized for performance)
        positions = self.agent_handler.positions
        n = len(positions)

        # Compute all pairwise displacements: shape (n, n, 2)
        diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

        # Apply torus wrapping if enabled
        if self.torus:
            half = self.world_size / 2.0
            diff = np.where(diff > half, diff - self.world_size, diff)
            diff = np.where(diff < -half, diff + self.world_size, diff)

        # Compute all pairwise distances: shape (n, n)
        distances = np.linalg.norm(diff, axis=2)

        # Extract upper triangle (avoid double-counting and self-distances)
        upper_tri_indices = np.triu_indices(n, k=1)
        pairwise_dists = distances[upper_tri_indices]

        # Clip and sum
        clipped_dists = np.minimum(pairwise_dists, self.dc)
        total_distance = np.sum(clipped_dists)
        reward_distance = self.alpha * total_distance

        # Action penalty
        reward_action = 0.0
        if actions is not None:
            for act in actions.values():
                arr = np.asarray(act, dtype=np.float32)
                reward_action += np.linalg.norm(arr)
            reward_action *= self.beta

        total_reward = reward_distance + reward_action
        return {name: float(total_reward) for name in self.agent_names}

    # ------------------------------------------------------------------
    # Termination and truncation
    # ------------------------------------------------------------------
    def _check_terminations(self) -> Dict[str, bool]:
        """Return early termination flags.

        If break_distance_threshold is set, terminate when all pairwise
        distances are below the threshold (successful rendezvous).
        """
        terminations = {name: False for name in self.agent_names}

        if self.break_distance_threshold is not None:
            # Vectorized computation of maximum pairwise distance
            positions = self.agent_handler.positions
            n = len(positions)

            if self.torus:
                # Compute all pairwise displacements (torus-aware)
                # Shape: (n, n, 2)
                diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
                # Apply torus wrapping
                half = self.world_size / 2.0
                diff = np.where(diff > half, diff - self.world_size, diff)
                diff = np.where(diff < -half, diff + self.world_size, diff)
            else:
                # Simple Euclidean displacement
                diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]

            # Compute all pairwise distances: shape (n, n)
            distances = np.linalg.norm(diff, axis=2)
            # Maximum pairwise distance (ignoring diagonal self-distances)
            max_pairwise = np.max(distances)

            # If all agents are within threshold, terminate successfully
            if max_pairwise < self.break_distance_threshold:
                terminations = {name: True for name in self.agent_names}

        return terminations

    def _check_truncations(self) -> Dict[str, bool]:
        """Truncations are handled via step_count in BaseEnv."""
        return {name: False for name in self.agent_names}

    # ------------------------------------------------------------------
    # Info and intermediate steps
    # ------------------------------------------------------------------
    def _get_infos(self) -> Dict[str, dict]:
        """Provide diagnostic information: distance to COM and max pairwise distance."""
        infos: Dict[str, dict] = {}
        positions = self.agent_handler.positions
        mean_pos = np.mean(positions, axis=0)
        dists = np.linalg.norm(positions - mean_pos, axis=1)
        # Compute maximum pairwise distance
        max_pairwise = 0.0
        for i in range(len(self.agent_names)):
            for j in range(i + 1, len(self.agent_names)):
                d = np.linalg.norm(self.delta(positions[i], positions[j]))
                if d > max_pairwise:
                    max_pairwise = d
        for idx, name in enumerate(self.agent_names):
            infos[name] = {
                "distance_to_com": float(dists[idx]),
                "max_pairwise_distance": float(max_pairwise),
            }
        return infos

    def _intermediate_steps(self) -> None:
        """No intermediate processing required."""
        return None

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render(self) -> None:
        """Render the environment using pygame if in human mode."""
        if self.render_mode != "human":
            return
        self.__render_setup()
        self.__render_agents()
        # Display step count
        if self._font is not None:
            text_surface = self._font.render(
                f"Steps: {self.step_count}", True, (0, 0, 0)
            )
            assert self.screen is not None
            self.screen.blit(text_surface, (10, 10))
        pygame.display.flip()
        # Process pygame events to keep window responsive
        pygame.event.pump()
        assert self.clock is not None
        self.clock.tick(self.fps)

    def __render_setup(self) -> None:
        """Initialise the pygame display if not already set up."""
        if self.screen is not None:
            return
        pygame.init()
        self.screen = pygame.display.set_mode((self.window_size, self.window_size))
        pygame.display.set_caption("Rendezvous Environment")
        self.clock = pygame.time.Clock()
        self._font = pygame.font.SysFont("Arial", 18)

    def __render_agents(self) -> None:
        """Draw agents as circles with orientation arrows and communication radius."""
        assert self.screen is not None
        self.screen.fill((255, 255, 255))
        scale = self.window_size / float(self.world_size)
        positions = self.agent_handler.positions
        orientations = self.agent_handler.orientations
        for i, name in enumerate(self.agent_names):
            pos_world = positions[i]
            orient = orientations[i]
            x_pix = int(pos_world[0] * scale)
            y_pix = int(pos_world[1] * scale)
            # Draw agent body
            pygame.draw.circle(self.screen, (100, 100, 255), (x_pix, y_pix), 5)
            # Draw orientation as a red arrow
            arrow_len = 10
            end_x = x_pix + int(arrow_len * math.cos(orient))
            end_y = y_pix + int(arrow_len * math.sin(orient))
            pygame.draw.line(
                self.screen, (255, 0, 0), (x_pix, y_pix), (end_x, end_y), 2
            )
            # Draw communication radius for local models
            if self.obs_model.startswith("local"):
                pygame.draw.circle(
                    self.screen,
                    (190, 190, 255),
                    (x_pix, y_pix),
                    int(self.comm_radius * scale),
                    1,
                )

    def _close(self) -> None:
        """Close the pygame window if open."""
        if self.screen is not None:
            pygame.quit()
        self.screen = None
        self.clock = None
        self._font = None


if __name__ == "__main__":
    """Run a simple random rollout when this module is executed directly.

    This example instantiates a rendezvous environment with three agents
    and executes a single episode using random actions.  The built-in
    `render()` method is used to display the simulation if the
    environment was created with `render_mode="human"`.  The loop
    terminates either when a termination condition occurs or once
    ``max_steps`` have been executed.
    """
    # Create a small test environment.  Adjust parameters as needed.
    env = RendezvousEnv(
        num_agents=3,
        world_size=500.0,
        obs_model="global_basic",
        torus=False,
        kinematics="single",
        render_mode="human",
        fps=60,
        comm_radius=50.0,
        v_max=10,
        omega_max=1.0,
    )
    # Reset to obtain the initial observations
    observations, infos = env.reset()
    try:
        # Execute at most `max_steps` steps
        for _ in range(env.max_steps):
            # Sample a random action for each agent
            actions = {agent: env.action_spaces[agent].sample() for agent in env.agents}
            # Render the current state (a no‑op when render_mode != 'human')
            env.render()
            # Apply actions and collect feedback
            observations, rewards, terminations, truncations, infos = env.step(actions)
            # Stop early if any termination or truncation condition is met
            if any(terminations.values()) or any(truncations.values()):
                break
    finally:
        env.close()
