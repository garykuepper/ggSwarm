"""GATv2 Graph Neural Network policy for ggSwarm formation control.

Custom SKRL policy that uses GATv2 K-hop message passing for spatial
awareness. The environment publishes KNN sparse edges each step via
set_knn_edges(). The policy caches these edges and replays them during
PPO mini-batch updates to maintain graph structure.

K-hop depth is controlled by num_gnn_layers (default 2).
Edge sparsity is controlled by num_neighbors in env config (default K=2).
This architecture scales from 8 to 20+ agents without changes.

Compatible with SKRL's PPO via GaussianMixin + DeterministicMixin.
"""

from __future__ import annotations

from collections import deque

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv

from skrl.models.torch import DeterministicMixin, GaussianMixin, Model


class GgswarmGNNPolicy(GaussianMixin, DeterministicMixin, Model):
    """GATv2 policy for swarm formation control with K-hop message passing.

    Obs layout: [local_obs(12), rel_pos_n0(3), rel_pos_n1(3)] = 18D
    The policy uses 12D local obs as node features. KNN sparse edges are
    published by the environment via set_knn_edges() each step and cached
    in a deque ring buffer for replay during PPO updates.

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

    # --- Shared KNN edge state (set by environment, read by policy) ---
    _latest_knn_edges: torch.Tensor | None = None
    _latest_knn_batch_size: int = -1
    _edge_cache: deque = deque()
    _num_envs: int = 0

    @classmethod
    def init_edge_cache(cls, memory_size: int, num_envs: int) -> None:
        """Initialize the edge cache ring buffer.

        Must be called after policy creation and before training.

        Args:
            memory_size: number of rollout steps (matches SKRL memory_size)
            num_envs: total number of environments (= total drone count)
        """
        cls._edge_cache = deque(maxlen=memory_size)
        cls._num_envs = num_envs

    @classmethod
    def set_knn_edges(cls, edge_index: torch.Tensor, batch_size: int) -> None:
        """Publish KNN sparse edges from the environment.

        Called by the environment in _get_observations() every step.
        Caches the edges for replay during PPO mini-batch updates.

        Args:
            edge_index: [2, num_edges] — bidirectional KNN edges
            batch_size: number of drones in the batch (for freshness check)
        """
        cls._latest_knn_edges = edge_index
        cls._latest_knn_batch_size = batch_size
        cls._num_envs = batch_size
        # Cache for PPO update replay (deque auto-drops oldest at maxlen)
        cls._edge_cache.append(edge_index.clone())

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

        # Empty edge fallback (for initialization / single-agent)
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
        """Split obs into node features and retrieve/reconstruct KNN edges.

        Three modes:
        1. Collection (B == num_envs): use fresh edges from env
        2. PPO update (B is multiple of num_envs): reconstruct from cache
        3. Fallback: empty edges (self-loops only via add_self_loops=True)

        Args:
            obs: shape [batch, 18] — [local_obs(12), rel_n0(3), rel_n1(3)]

        Returns:
            node_features: [batch, 12] — local obs per drone
            edge_index: [2, num_edges] — KNN sparse graph connectivity
        """
        node_features = obs[:, :self._local_obs_dim]  # [B, 12]
        B = obs.shape[0]
        N = GgswarmGNNPolicy._num_envs

        if B == N and GgswarmGNNPolicy._latest_knn_edges is not None:
            # Collection phase: single timestep, use fresh edges
            edge_index = GgswarmGNNPolicy._latest_knn_edges

        elif (
            N > 0
            and B > N
            and B % N == 0
            and len(GgswarmGNNPolicy._edge_cache) > 0
        ):
            # PPO update phase: mini-batch contains multiple timestep blocks
            # Each block of N samples has group structure intact (SKRL sample_all
            # returns sequential contiguous slices, no shuffling)
            num_blocks = B // N
            edges = []
            for block_idx in range(num_blocks):
                cache_idx = block_idx % len(GgswarmGNNPolicy._edge_cache)
                # Offset edge indices by block position in the mega-batch
                block_edges = (
                    GgswarmGNNPolicy._edge_cache[cache_idx] + block_idx * N
                )
                edges.append(block_edges)
            edge_index = torch.cat(edges, dim=1)

        else:
            # Fallback: no edges (initialization, single-agent, or misaligned batch)
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


class GgswarmCentralizedValue(DeterministicMixin, Model):
    """Centralized critic for MAPPO (Phase 1+ shared-scene multi-drone).

    Takes the concatenation of all A drone obs in an env as the shared
    state and produces a scalar value. Decentralized actors (`GgswarmGNNPolicy`)
    consume per-drone obs at execution time; this critic is training-only.

    With `state_space=-1` on the env cfg, DirectMARLEnv auto-concatenates
    obs_dict[drone_0..drone_{A-1}] along the feature dim, giving the
    shared state shape `[num_envs, A * obs_per_agent]` (e.g., 8*18=144 for
    K=2 neighbors).

    All A agents share the same value model instance for parameter sharing.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        hidden_channels: int = 128,
    ):
        Model.__init__(self, observation_space, action_space, device)
        DeterministicMixin.__init__(self, clip_actions=False, role="value")

        self.net = nn.Sequential(
            nn.Linear(self.num_observations, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ELU(),
            nn.Linear(hidden_channels, 1),
        )

    def compute(self, inputs, role=""):  # noqa: ARG002
        states = inputs.get("states")
        return self.net(states), {}
