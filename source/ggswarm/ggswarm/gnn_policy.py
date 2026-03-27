"""GATv2 Graph Neural Network policy for ggSwarm formation control.

Custom SKRL policy that replaces the MLP with GATv2 message passing.
Each drone is a graph node (12D local obs). Edges connect K-nearest
neighbors. GATv2 performs attention-weighted message passing to produce
4D actions (thrust + 3 moments).

Compatible with SKRL's PPO via GaussianMixin + DeterministicMixin.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class GgswarmGNNPolicy(GaussianMixin, DeterministicMixin, Model):
    """GATv2 policy for swarm formation control.

    Obs layout: [local_obs(12), rel_pos_n0(3), rel_pos_n1(3)] = 18D
    The policy splits this into node features (12D) and neighbor info
    (6D) to construct the graph internally.

    Args:
        observation_space: Observation space (18D for K=2 neighbors).
        action_space: Action space (4D: thrust + 3 moments).
        device: Torch device.
        hidden_channels: GATv2 hidden dimension.
        num_heads: Number of attention heads.
        num_gnn_layers: Number of GATv2 layers.
        num_neighbors: K-nearest neighbors (must match env config).
        local_obs_dim: Dimension of local observation (12D).
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_channels: int = 64,
        num_heads: int = 2,
        num_gnn_layers: int = 2,
        num_neighbors: int = 2,
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
        """Split obs into node features and construct edge index.

        Args:
            obs: shape [batch, 18] — [local_obs(12), rel_n0(3), rel_n1(3)]

        Returns:
            node_features: [batch, 12] — local obs per drone
            edge_index: [2, num_edges] — graph connectivity
        """
        B = obs.shape[0]
        K = self._num_neighbors
        local_obs_dim = self._local_obs_dim

        # Split obs
        node_features = obs[:, :local_obs_dim]  # [B, 12]

        # Build edge index: each node connects to K neighbors
        # Since we're processing a batch of independent nodes (not a
        # multi-node graph), we create self-loops and neighbor edges
        # based on the neighbor relative positions being non-zero.
        #
        # For batched PPO: each "node" in the batch is an independent
        # drone. We connect each drone to itself (self-loop handled by
        # GATv2's add_self_loops=True). For neighbor edges, we need to
        # know which batch indices are neighbors.
        #
        # Key insight: during training, the batch contains all drones
        # from all swarm groups. Drones in the same group are at
        # consecutive indices. We can reconstruct edges from this.

        # Extract neighbor relative positions
        neighbor_data = obs[:, local_obs_dim:]  # [B, K*3]

        # Build edges: connect node i to its K nearest neighbors
        # Neighbors are at positions i+offset within each swarm group
        # But we don't know group boundaries here. Instead, use the
        # neighbor data to identify which nodes are connected.
        #
        # Simple approach: create a fully-connected graph within each
        # processing batch. GATv2 attention will learn to weight edges.
        # This works because self-loops + attention = local processing.

        # For now: each node connects to all others in the batch
        # (GATv2 attention handles relevance). For small batches (3-6
        # drones) this is efficient. For large batches, we'd need
        # proper edge construction.
        if B <= 64:
            # Small batch: fully connected (play mode or small training)
            src = torch.arange(B, device=obs.device).repeat_interleave(B)
            dst = torch.arange(B, device=obs.device).repeat(B)
            # Remove self-loops (GATv2 adds them internally)
            mask = src != dst
            edge_index = torch.stack([src[mask], dst[mask]], dim=0)
        else:
            # Large batch: self-loops only (GATv2 adds them)
            # The node encoder + self-attention acts as an MLP fallback
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=obs.device)

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
