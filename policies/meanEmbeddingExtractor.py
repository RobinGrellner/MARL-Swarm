from typing import List
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class MeanEmbeddingExtractor(BaseFeaturesExtractor):
    """
    Per-Agent extractor for observations: [local | neighbors | mask]

    :param phi_hidden: List for the hidden layers of the Mean-Embedding NN
    :param embed_dim: Dimension of the embedding per neighbor.
    :return: The mean embedding + local observation.
    """
    def __init__(
        self,
        observation_space: spaces.Box,
        local_dim: int,
        neigh_dim: int,
        max_neigh: int,
        embed_dim: int = 64,
        phi_hidden: List[int] = (128,)
    ) -> None:
        self.local_dim = local_dim
        self.neigh_dim = neigh_dim
        self.max_neigh = max_neigh
        self.embed_dim = int(embed_dim) if neigh_dim > 0 else 0

        # Output-dimension
        output_dimension = self.local_dim + self.embed_dim
        super().__init__(observation_space, features_dim=output_dimension)

        # Agent observation
        self.agent_obs_dim = self.local_dim + self.max_neigh * self.neigh_dim + self.max_neigh

        # Embedding-MLP: Observations -> MLP -> output_dimension (Embedding)
        self.network = None
        if self.neigh_dim > 0 and self.embed_dim > 0:
            layers: List[nn.Module] = []
            curr_layer_dim = self.neigh_dim
            for h in phi_hidden:
                layers += [nn.Linear(curr_layer_dim, h), nn.ReLU()]
                curr_layer_dim = h
            layers += [nn.Linear(curr_layer_dim, self.embed_dim), nn.ReLU()]
            self.network = nn.Sequential(*layers)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Observations: (B, agent_obs_dim)
        batch_size = observations.shape[0]
        assert observations.shape[1] == self.agent_obs_dim, \
            f"Expected {self.agent_obs_dim}, got {observations.shape[1]}"

        local = observations[:, : self.local_dim]

        if self.neigh_dim == 0 or self.embed_dim == 0 or self.network is None:
            return local

        start = self.local_dim
        end = start + self.max_neigh * self.neigh_dim
        neigh = observations[:, start:end].view(batch_size, self.max_neigh, self.neigh_dim)      # (batchsize, max_neighbors, neighbor_dim)
        mask = observations[:, end : end + self.max_neigh].unsqueeze(-1)                 # (batchsize, max_neighbors, 1)

        # φ auf jeden Nachbar
        phi_out = self.network(neigh)                                                        # (batchsize, max_neighbors, neighbor_dim)
        # gemasktes Mittel
        sum_mask = mask.sum(dim=1).clamp_min(1e-6)                                       # (batchsize, 1, 1) -> Broadcast
        phi_mean = (phi_out * mask).sum(dim=1) / sum_mask.squeeze(1)                     # (batchsize, neighbor_dim)

        return torch.cat([phi_mean, local], dim=1)