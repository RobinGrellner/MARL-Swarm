from typing import List, Optional, Dict, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces


# Mapping from string names to activation function constructors
ACTIVATION_FUNCTIONS: Dict[str, Callable[[], nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "elu": nn.ELU,
}


class AttentionAggregation(nn.Module):
    """Learnable attention-based aggregation over neighbor embeddings.

    This module computes attention weights for each neighbor embedding and
    produces a weighted sum. The attention mechanism learns which neighbors
    are most relevant for decision-making.

    Args:
        embed_dim: Dimension of the neighbor embeddings to aggregate.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable attention projection: maps embedding to scalar attention score
        self.attention_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )

    def forward(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute attention-weighted aggregation of neighbor embeddings.

        Args:
            embeddings: Neighbor embeddings of shape (batch, max_neigh, embed_dim).
            mask: Binary mask of shape (batch, max_neigh) indicating valid neighbors.

        Returns:
            Aggregated embedding of shape (batch, embed_dim).
        """
        # Compute raw attention scores: (batch, max_neigh, 1)
        attention_scores = self.attention_proj(embeddings)
        # Squeeze last dimension: (batch, max_neigh)
        attention_scores = attention_scores.squeeze(-1)

        # Check for all-masked case BEFORE softmax to avoid NaN gradients
        has_valid_neighbors = mask.sum(dim=-1, keepdim=True) > 0  # (batch, 1)

        # Mask out invalid neighbors with large negative value before softmax
        # This ensures they get ~0 weight after softmax
        mask_value = torch.finfo(attention_scores.dtype).min
        attention_scores = attention_scores.masked_fill(mask == 0, mask_value)

        # Apply softmax to get attention weights: (batch, max_neigh)
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Clean zero-padding for all-masked cases to avoid NaN gradients
        # Use torch.where to ensure clean zeros instead of NaN
        attention_weights = torch.where(
            has_valid_neighbors,
            attention_weights,
            torch.zeros_like(attention_weights)
        )

        # Weighted sum of embeddings: (batch, embed_dim)
        # attention_weights: (batch, max_neigh) -> unsqueeze to (batch, max_neigh, 1)
        weighted_embeddings = embeddings * attention_weights.unsqueeze(-1)
        aggregated = weighted_embeddings.sum(dim=1)

        return aggregated


class MeanEmbeddingExtractor(BaseFeaturesExtractor):
    """Permutation- and scale-invariant feature-extractor.

    Takes a flattened observation from a single agent and produces a fixed-dimensional
    representation by aggregating a neural embedding across all neighbors. The NN for the
    embedding is trained jointly with the RL-agent. Result is concatenated with the
    agent's local features to form the final feature-vector. For neighbor dimension == 0
    the embedding is skipped -> only local features.

    Args:
        observation_space: The observation space of a single agent.
        local_dim: Number of local features (agent's own state).
        neigh_dim: Number of features per neighbor.
        max_neigh: Maximum number of neighbors.
        embed_dim: Dimension of the learned neighbor embedding (default: 64).
        phi_hidden: List of hidden layer sizes for the phi network (default: [64]).
        num_agents: Optional number of agents (for compatibility).
        eps: Small epsilon for numerical stability (default: 1e-6).
        activation: Activation function name. One of "relu", "tanh", "gelu",
                   "leaky_relu", "elu" (default: "relu").
        aggregation: Aggregation operation for neighbor embeddings. One of "mean",
                    "max", "sum", "attention" (default: "mean").
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
        activation: str = "relu",
        aggregation: str = "mean",
    ) -> None:
        self.local_dim = local_dim
        self.neigh_dim = neigh_dim
        self.max_neigh = max_neigh
        self.embed_dim = embed_dim if neigh_dim > 0 else 0
        self.num_agents = int(num_agents) if num_agents is not None else None
        self.eps = float(eps)
        self.activation_name = activation.lower()
        self.aggregation_name = aggregation.lower()

        # Validate activation function
        if self.activation_name not in ACTIVATION_FUNCTIONS:
            raise ValueError(
                f"Unknown activation function: '{activation}'. "
                f"Supported: {list(ACTIVATION_FUNCTIONS.keys())}"
            )

        # Validate aggregation operation
        valid_aggregations = {"mean", "max", "sum", "attention"}
        if self.aggregation_name not in valid_aggregations:
            raise ValueError(
                f"Unknown aggregation operation: '{aggregation}'. "
                f"Supported: {list(valid_aggregations)}"
            )

        # Feature dimension is per-agent (for parameter sharing across all agents)
        # Output: local features + aggregated embedding (if neighbors exist)
        features_dim = (self.embed_dim if self.neigh_dim > 0 and self.embed_dim > 0 else 0) + self.local_dim

        super().__init__(observation_space, features_dim=features_dim)

        # Get the activation function constructor
        activation_fn = ACTIVATION_FUNCTIONS[self.activation_name]

        # Build phi network: a sequence of linear layers with configurable activations
        layers: List[nn.Module] = []
        if self.neigh_dim > 0 and self.embed_dim > 0:
            phi_hidden = phi_hidden or [64]
            curr_layer_size = self.neigh_dim
            for tmp_layer_size in phi_hidden:
                layers.append(nn.Linear(curr_layer_size, tmp_layer_size))
                layers.append(activation_fn())
                curr_layer_size = tmp_layer_size
            # Final layer maps to embed_dim
            layers.append(nn.Linear(curr_layer_size, self.embed_dim))
            layers.append(activation_fn())
            self.phi = nn.Sequential(*layers)

            # Create attention module if using attention aggregation
            if self.aggregation_name == "attention":
                self.attention = AttentionAggregation(self.embed_dim)
            else:
                self.attention = None
        else:
            self.phi = None
            self.attention = None

        # Offsets for parsing the observation vector
        self._agent_obs_dim = self.local_dim + self.max_neigh * self.neigh_dim + self.max_neigh
        self._neigh_block_len = self.max_neigh * self.neigh_dim

    def _aggregate(self, phi_out: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Aggregate neighbor embeddings using the configured aggregation operation.

        Args:
            phi_out: Neighbor embeddings of shape (batch, max_neigh, embed_dim).
            mask: Binary mask of shape (batch, max_neigh) indicating valid neighbors.

        Returns:
            Aggregated embedding of shape (batch, embed_dim).
        """
        mask_exp = mask.unsqueeze(-1)  # (batch, max_neigh, 1)

        if self.aggregation_name == "mean":
            # Mean pooling over valid neighbors
            sum_mask = mask_exp.sum(dim=1, keepdim=True).clamp_min(self.eps)
            aggregated = (phi_out * mask_exp).sum(dim=1) / sum_mask.squeeze(1)

        elif self.aggregation_name == "sum":
            # Sum pooling over valid neighbors
            aggregated = (phi_out * mask_exp).sum(dim=1)

        elif self.aggregation_name == "max":
            # Max pooling over valid neighbors
            # Set masked positions to very negative values so they don't affect max
            masked_phi = phi_out.clone()
            mask_value = torch.finfo(masked_phi.dtype).min
            masked_phi = masked_phi.masked_fill(mask_exp == 0, mask_value)
            aggregated, _ = masked_phi.max(dim=1)
            # Handle edge case: if all neighbors are masked, return zeros
            all_masked = (mask.sum(dim=-1, keepdim=True) == 0)
            aggregated = aggregated.masked_fill(all_masked, 0.0)

        elif self.aggregation_name == "attention":
            # Attention-weighted aggregation
            aggregated = self.attention(phi_out, mask)

        else:
            # This should never happen due to validation in __init__
            raise ValueError(f"Unknown aggregation: {self.aggregation_name}")

        return aggregated

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Process observations and produce feature vectors.

        Args:
            observations: Flattened observations of shape (batch_size, agent_obs_dim).

        Returns:
            Feature vectors of shape (batch_size, features_dim).
        """
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
            # Compute phi for each neighbour
            phi_out = self.phi(neigh)
            # Aggregate using configured method
            phi_aggregated = self._aggregate(phi_out, mask)
            # Concatenate local features and aggregated embedding
            return torch.cat([local, phi_aggregated], dim=1)
        else:
            # No neighbour features present
            return local
