"""GATv2 Graph Neural Network policy for ggSwarm formation control.

Custom SKRL policy that uses GATv2 K-hop message passing for spatial
awareness. The environment publishes KNN sparse edges each step via
set_knn_edges(). The policy reads these during collection (group-aligned
batches) and falls back to self-loops during PPO mini-batch updates.

K-hop depth is controlled by num_gnn_layers (default 2).
Edge sparsity is controlled by num_neighbors in env config (default K=2).
This architecture scales from 8 to 20+ agents without changes.

Compatible with SKRL's PPO via GaussianMixin + DeterministicMixin.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class GgswarmGNNPolicy(GaussianMixin, DeterministicMixin, Model):
    """GATv2 policy for swarm formation control with K-hop message passing.

    Obs layout: [local_obs(12), rel_pos_n0(3), rel_pos_n1(3)] = 18D
    The policy uses 12D local obs as node features. KNN sparse edges are
    published by the environment via set_knn_edges() each step.

    Args:
        observation_space: Observation space (18D for K=2 neighbors).
        action_space: Action space (4D: thrust + 3 moments).
        device: Torch device.
        hidden_channels: GATv2 hidden dimension.
        num_heads: Number of attention heads.
        num_gnn_layers: Number of GATv2 layers (K-hop depth).
        num_neighbors: K-nearest neighbors (must match env config).
        num_agents: Drones per swarm group (kept for backward compat).
        local_obs_dim: Dimension of local observation (12D).
    """

    # --- Shared KNN edge buffer (set by environment each step) ---
    _latest_knn_edges: torch.Tensor | None = None
    _latest_knn_batch_size: int = -1

    @classmethod
    def set_knn_edges(cls, edge_index: torch.Tensor, batch_size: int) -> None:
        """Publish KNN sparse edges from the environment.

        Called by the environment in _get_observations() every step.
        The policy reads these in _prepare_graph() during collection.

        Args:
            edge_index: [2, num_edges] — bidirectional KNN edges
            batch_size: number of drones in the batch (for freshness check)
        """
        cls._latest_knn_edges = edge_index
        cls._latest_knn_batch_size = batch_size

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_channels: int = 64,
        num_heads: int = 2,
        num_gnn_layers: int = 2,
        num_neighbors: int = 2,
        num_agents: int = 8,
        local_obs_dim: int = 12,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self,
            clip_actions=False,
            clip_log_std=True,
            min_log_std=-20.0,
            max_log_std=2.0,
            reduction="sum",
            role="policy",
        )
        DeterministicMixin.__init__(self, clip_actions=False, role="value")

        self._num_neighbors = num_neighbors
        self._num_agents = num_agents
        self._local_obs_dim = local_obs_dim
        self._hidden = hidden_channels

        # Node embedding: local obs -> hidden
        self.node_encoder = nn.Sequential(
            nn.Linear(local_obs_dim, hidden_channels),
            nn.ELU(),
        )

        # GATv2 layers
        self.gnn_layers = nn.ModuleList()
        for _ in range(num_gnn_layers):
            self.gnn_layers.append(
                GATv2Conv(hidden_channels, hidden_channels // num_heads,
                          heads=num_heads, concat=True, add_self_loops=True)
            )
        self.gnn_norms = nn.ModuleList([
            nn.LayerNorm(hidden_channels) for _ in range(num_gnn_layers)
        ])

        # Policy head (actions)
        self.policy_head = nn.Linear(hidden_channels, self.num_actions)
        self.log_std_parameter = nn.Parameter(
            torch.zeros(self.num_actions), requires_grad=True
        )

        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, 1),
        )

        # Cache for shared computation between policy and value
        self._shared_output = None

        # Empty edge fallback (for mini-batch / single-agent)
        self.register_buffer(
            "_empty_edge_index",
            torch.zeros(2, 0, dtype=torch.long),
        )

    def act(self, inputs, role=""):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role=""):
        # Get full obs: [batch, 18] = [batch, 12 + K*3]
        states = inputs.get("states")

        if role == "policy":
            node_features, edge_index = self._prepare_graph(states)
            h = self._gnn_forward(node_features, edge_index)
            self._shared_output = h
            actions = self.policy_head(h)
            return actions, self.log_std_parameter, {}

        elif role == "value":
            if self._shared_output is None:
                node_features, edge_index = self._prepare_graph(states)
                h = self._gnn_forward(node_features, edge_index)
            else:
                h = self._shared_output
            self._shared_output = None
            value = self.value_head(h)
            return value, {}

    def _prepare_graph(self, obs: torch.Tensor):
        """Split obs into node features and retrieve KNN sparse edges.

        Uses edges published by the environment via set_knn_edges().
        Falls back to empty edges (self-loops only) when edges are stale
        or batch size doesn't match (PPO mini-batch update).

        Args:
            obs: shape [batch, 18] — [local_obs(12), rel_n0(3), rel_n1(3)]

        Returns:
            node_features: [batch, 12] — local obs per drone
            edge_index: [2, num_edges] — KNN sparse graph connectivity
        """
        node_features = obs[:, :self._local_obs_dim]  # [B, 12]
        B = obs.shape[0]

        # Use KNN edges if fresh (published by env this step, batch size matches)
        if (
            GgswarmGNNPolicy._latest_knn_edges is not None
            and GgswarmGNNPolicy._latest_knn_batch_size == B
        ):
            edge_index = GgswarmGNNPolicy._latest_knn_edges
        else:
            # Mini-batch during PPO update or single-agent: no edges
            edge_index = self._empty_edge_index

        return node_features, edge_index

    def _gnn_forward(self, node_features: torch.Tensor,
                     edge_index: torch.Tensor) -> torch.Tensor:
        """Run GATv2 message passing.

        Args:
            node_features: [batch, 12]
            edge_index: [2, num_edges]

        Returns:
            h: [batch, hidden_channels]
        """
        h = self.node_encoder(node_features)  # [B, hidden]

        for gnn_layer, norm in zip(self.gnn_layers, self.gnn_norms):
            h_res = h
            h = gnn_layer(h, edge_index)  # [B, hidden]
            h = norm(h + h_res)  # residual + layer norm
            h = torch.nn.functional.elu(h)

        return h
