"""GATv2 Graph Neural Network policy for ggSwarm formation control.

Custom SKRL policy that replaces the MLP with GATv2 message passing.
Each drone is a graph node (12D local obs). Edges connect all drones
within each swarm group (fully-connected within-group). GATv2 performs
attention-weighted message passing to produce 4D actions.

During PPO mini-batch updates (shuffled samples), falls back to
self-loops only since group structure is destroyed. This is standard
practice in GNN-RL (DGN, CommNet, TarMAC).

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
    The policy splits this into node features (12D) and uses the group
    structure to build within-group edges for GATv2 message passing.

    Args:
        observation_space: Observation space (18D for K=2 neighbors).
        action_space: Action space (4D: thrust + 3 moments).
        device: Torch device.
        hidden_channels: GATv2 hidden dimension.
        num_heads: Number of attention heads.
        num_gnn_layers: Number of GATv2 layers (K-hop depth).
        num_neighbors: K-nearest neighbors (must match env config).
        num_agents: Drones per swarm group (for edge construction).
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

        # --- Pre-compute within-group edge template ---
        # Fully-connected edges within a single group of A agents
        # (excludes self-loops — GATv2 adds those via add_self_loops=True)
        A = num_agents
        if A > 1:
            src, dst = [], []
            for i in range(A):
                for j in range(A):
                    if i != j:
                        src.append(i)
                        dst.append(j)
            self.register_buffer(
                "_group_edge_template",
                torch.tensor([src, dst], dtype=torch.long),
            )  # shape: [2, A*(A-1)]
        else:
            self.register_buffer(
                "_group_edge_template",
                torch.zeros(2, 0, dtype=torch.long),
            )

        # Empty edge fallback (for mini-batch / single-agent)
        self.register_buffer(
            "_empty_edge_index",
            torch.zeros(2, 0, dtype=torch.long),
        )

        # Cache for the expanded edge_index (avoid recomputation)
        self._cached_edge_index: torch.Tensor | None = None
        self._cached_num_groups: int = -1

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
        """Split obs into node features and construct within-group edges.

        When the batch is group-aligned (batch_size divisible by num_agents),
        builds fully-connected edges within each swarm group for proper
        GATv2 message passing. Otherwise falls back to empty edges
        (self-loops only via add_self_loops=True).

        Args:
            obs: shape [batch, 18] — [local_obs(12), rel_n0(3), rel_n1(3)]

        Returns:
            node_features: [batch, 12] — local obs per drone
            edge_index: [2, num_edges] — graph connectivity
        """
        node_features = obs[:, :self._local_obs_dim]  # [B, 12]

        B = obs.shape[0]
        A = self._num_agents

        # Build within-group edges when batch structure is intact
        # (collection phase: full env batch with consecutive group indices)
        if A > 1 and B >= A and B % A == 0:
            G = B // A
            # Rebuild edge_index only when group count changes
            if G != self._cached_num_groups:
                offsets = torch.arange(G, device=obs.device) * A  # [G]
                # Broadcast template [2, E] across G groups with offsets
                # template.unsqueeze(2): [2, E, 1] + offsets [1, 1, G] → [2, E, G]
                self._cached_edge_index = (
                    self._group_edge_template.unsqueeze(2)
                    + offsets.reshape(1, 1, G)
                ).reshape(2, -1)
                self._cached_num_groups = G
            edge_index = self._cached_edge_index
        else:
            # Mini-batch during PPO update or single-agent: no group structure
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
