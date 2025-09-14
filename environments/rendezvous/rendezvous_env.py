from environments.base.base_environment import BaseEnv
from environments.base.agent_handler import AgentHandler
import environments.rendezvous.observations as observation_helpers
from typing import Dict, List, Tuple, Optional, Any
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
    ):
        # Store rendezvous-specific parameters
        self.csv_log_path = csv_log_path
        self.obs_model = obs_model.lower() if obs_model is not None else "global_basic"
        self.comm_radius = comm_radius
        self.render_mode = render_mode

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
        self._max_neighbours = num_agents - 1

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
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """Construct observations for all agents."""
        observations: Dict[str, np.ndarray] = {}
        positions = self.agent_handler.positions
        orientations = self.agent_handler.orientations
        linear_vels = self.agent_handler.linear_vels
        angular_vels = self.agent_handler.angular_vels
        kin = self.agent_handler.kinematics
        num_agents = self.agent_handler.num_agents

        # Precompute velocity vectors when needed
        velocity_vectors: Optional[np.ndarray] = None
        if self.obs_model == "global_extended":
            velocity_vectors = np.stack(
                [
                    linear_vels * np.cos(orientations),
                    linear_vels * np.sin(orientations),
                ],
                axis=1,
            )

        for idx, name in enumerate(self.agent_names):
            pos_i = positions[idx]
            theta_i = orientations[idx]

            # Classic observation
            if self.obs_model == "classic":
                mean_pos = np.mean(positions, axis=0)
                if kin == "single":
                    obs_vec = np.array(
                        [
                            pos_i[0],
                            pos_i[1],
                            mean_pos[0],
                            mean_pos[1],
                            linear_vels[idx],
                            theta_i,
                        ],
                        dtype=np.float32,
                    )
                else:
                    obs_vec = np.array(
                        [
                            pos_i[0],
                            pos_i[1],
                            mean_pos[0],
                            mean_pos[1],
                            linear_vels[idx],
                            angular_vels[idx],
                            theta_i,
                        ],
                        dtype=np.float32,
                    )
                observations[name] = obs_vec
                continue

            # Distance and bearing to nearest wall
            if self.torus:
                d_wall = float(self.world_size)
                phi_wall = 0.0
            else:
                dx_left = pos_i[0]
                dx_right = self.world_size - pos_i[0]
                dy_bottom = pos_i[1]
                dy_top = self.world_size - pos_i[1]
                dists = [dx_left, dx_right, dy_bottom, dy_top]
                d_wall = float(min(dists))
                min_idx = int(np.argmin(dists))
                if min_idx == 0:
                    target = np.array([0.0, pos_i[1]], dtype=np.float32)
                elif min_idx == 1:
                    target = np.array([self.world_size, pos_i[1]], dtype=np.float32)
                elif min_idx == 2:
                    target = np.array([pos_i[0], 0.0], dtype=np.float32)
                else:
                    target = np.array([pos_i[0], self.world_size], dtype=np.float32)
                delta_w = target - pos_i
                phi_wall = float(math.atan2(delta_w[1], delta_w[0]) - theta_i)
                phi_wall = (phi_wall + math.pi) % (2 * math.pi) - math.pi

            local_feats: List[float] = [d_wall, phi_wall]
            if self.obs_model == "local_comm":
                # count neighbours within comm radius
                count_i = 0
                for j in range(num_agents):
                    if j == idx:
                        continue
                    if (
                        np.linalg.norm(self.delta(pos_i, positions[j]))
                        <= self.comm_radius
                    ):
                        count_i += 1
                local_feats.append(float(count_i))

            # Build neighbour feature vector and mask
            neighbour_features: List[float] = []
            mask: List[float] = []
            # Collect all neighbours and sort by distance for deterministic ordering
            neighbour_info: List[Tuple[float, int]] = []
            for j in range(num_agents):
                if j == idx:
                    continue
                dvec = self.delta(pos_i, positions[j])
                distance = np.linalg.norm(dvec)
                neighbour_info.append((distance, j))
            neighbour_info.sort(key=lambda x: x[0])

            for _, j in neighbour_info:
                pos_j = positions[j]
                dvec = self.delta(pos_i, pos_j)
                d_ij = float(np.linalg.norm(dvec))
                valid = True
                if self.obs_model.startswith("local") and d_ij > self.comm_radius:
                    valid = False
                # Bearing relative to i's orientation
                if valid:
                    bearing = float(math.atan2(dvec[1], dvec[0]) - theta_i)
                    bearing = (bearing + math.pi) % (2 * math.pi) - math.pi
                else:
                    bearing = 0.0
                features_j: List[float] = []
                if valid:
                    features_j.append(d_ij)
                    features_j.append(bearing)
                    # Relative orientation
                    if self.obs_model in {
                        "global_extended",
                        "local_extended",
                        "local_comm",
                    }:
                        psi = float(orientations[j] - theta_i)
                        psi = (psi + math.pi) % (2 * math.pi) - math.pi
                        features_j.append(psi)
                    # Relative velocity
                    if self.obs_model == "global_extended":
                        assert velocity_vectors is not None
                        rel_v = velocity_vectors[idx] - velocity_vectors[j]
                        features_j.append(float(rel_v[0]))
                        features_j.append(float(rel_v[1]))
                    # Neighbourhood size of j for local_comm
                    if self.obs_model == "local_comm":
                        count_j = 0
                        for k in range(num_agents):
                            if k == j:
                                continue
                            if (
                                np.linalg.norm(self.delta(pos_j, positions[k]))
                                <= self.comm_radius
                            ):
                                count_j += 1
                        features_j.append(float(count_j))
                else:
                    # Invalid neighbour: pad with zeros
                    features_j = [0.0] * self._neighbour_feature_dim
                neighbour_features.extend(features_j)
                mask.append(1.0 if valid else 0.0)
            # Convert to fixed length arrays
            obs_vec = np.concatenate(
                [
                    np.asarray(local_feats, dtype=np.float32),
                    np.asarray(neighbour_features, dtype=np.float32),
                    np.asarray(mask, dtype=np.float32),
                ]
            )
            observations[name] = obs_vec
        return observations

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
        # Sum of clipped distances
        total_distance = 0.0
        for i in range(len(self.agent_names)):
            for j in range(i + 1, len(self.agent_names)):
                d = np.linalg.norm(
                    self.delta(
                        self.agent_handler.positions[i], self.agent_handler.positions[j]
                    )
                )
                total_distance += min(d, self.dc)
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
        """Return early termination flags."""
        terminations = {name: False for name in self.agent_names}
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
    and executes a single episode using random actions.  The built‑in
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
    observations = env.reset()
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
