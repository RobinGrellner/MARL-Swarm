from typing import List, Optional
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


class MeanEmbeddingExtractor(BaseFeaturesExtractor):
    """Permutation- and scale-invariant feature-extractor.

    Takes a flattened observation from a single agent and produces a fixed-dimensional
    representation by averaging a neural embedding across all neighbors. The NN for the
    Mean-Embedding is trained jointly with the RL-agent. Result is concatenated with the
    Agents local features to form the final feature-vector. For neighbor dimension == 0
    the Mean-Embedding is skipped -> only local features.
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        local_dim: int,
        neigh_dim: int,
        max_neigh: int,
        embed_dim: int = 64,
        phi_hidden: Optional[List[int]] = None,
        num_agents: Optional[int] = None,
        eps: float = 1e-6,
    ) -> None:
        self.local_dim = local_dim
        self.neigh_dim = neigh_dim
        self.max_neigh = max_neigh
        self.embed_dim = embed_dim if neigh_dim > 0 else 0
        self.num_agents = int(num_agents) if num_agents is not None else None
        self.eps = float(eps)

        # Feature dimension is per-agent (for parameter sharing across all agents)
        # Output: local features + mean embedding (if neighbors exist)
        features_dim = (self.embed_dim if self.neigh_dim > 0 and self.embed_dim > 0 else 0) + self.local_dim

        super().__init__(observation_space, features_dim=features_dim)

        # Build network: a sequence of linear layers with ReLU activations
        layers: List[nn.Module] = []
        if self.neigh_dim > 0 and self.embed_dim > 0:
            phi_hidden = phi_hidden or [64]
            curr_layer_size = self.neigh_dim
            for tmp_layer_size in phi_hidden:
                layers.append(nn.Linear(curr_layer_size, tmp_layer_size))
                layers.append(nn.ReLU())
                curr_layer_size = tmp_layer_size
            # Final layer maps to embed_dim
            layers.append(nn.Linear(curr_layer_size, self.embed_dim))
            layers.append(nn.ReLU())
            self.phi = nn.Sequential(*layers)
        else:
            self.phi = None
                
        # Offsets:
        self._agent_obs_dim = self.local_dim + self.max_neigh * self.neigh_dim + self.max_neigh
        self._neigh_block_len = self.max_neigh * self.neigh_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Each observation corresponds to a single agent: shape (batch_size, agent_obs_dim)
        batch_size = observations.shape[0]
        # Split observation into local part, neighbour features and mask
        local = observations[:, : self.local_dim]
        if self.neigh_dim > 0 and self.embed_dim > 0 and self.phi is not None:
            start = self.local_dim
            end_feats = start + self.max_neigh * self.neigh_dim
            neigh_block = observations[:, start:end_feats]
            mask = observations[:, end_feats : end_feats + self.max_neigh]
            # Reshape neighbour block to (batch, max_neigh, neigh_dim)
            neigh = neigh_block.view(batch_size, self.max_neigh, self.neigh_dim)
            # Compute φ for each neighbour
            phi_out = self.phi(neigh)
            mask_exp = mask.unsqueeze(-1)
            # Sum and normalise
            sum_mask = mask_exp.sum(dim=1, keepdim=True).clamp_min(1e-6)
            phi_mean = (phi_out * mask_exp).sum(dim=1) / sum_mask.squeeze(1)
            # Concatenate local features and mean embedding
            return torch.cat([local, phi_mean], dim=1)
        else:
            # No neighbour features present
            return local
