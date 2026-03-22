# Copyright (c) 2022-2026, ggSwarm Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""GNN policy wrapper bridging PyTorch Geometric and SKRL."""

from typing import Any, cast

import torch
import torch.nn as nn
from skrl.models.torch import GaussianMixin, Model
from torch_geometric.nn import GATv2Conv

from ..contract_logic import adjacency_to_edge_index


class GGSwarmGNNPolicy(GaussianMixin, Model):
    """GATv2-based policy network for decentralized swarm coordination.

    This model converts the dense 3D adjacency matrix from the environment
    into a sparse 2D edge_index for PyTorch Geometric, handling batched
    environments as disconnected components in a single graph.
    """

    def __init__(
        self,
        observation_space,
        action_space,
        device,
        clip_actions=False,
        clip_log_std=True,
        min_log_std=-20,
        max_log_std=2,
        reduction="sum",
        *,
        hidden_channels: int = 128,
        num_heads: int = 2,
        initial_log_std: float = -0.5,
        **kwargs,
    ):
        """Build GATv2 policy; architecture widths are cfg-driven (Rule 14).

        Args:
            hidden_channels: Output width of the second GAT layer (and action MLP input).
            num_heads: Attention heads on the first GAT layer (second layer uses 1 head).
            initial_log_std: Initial diagonal log-std for the Gaussian policy head.
            kwargs: Absorbs extra keys from skrl's model instantiator.
        """
        _ = kwargs  # skrl may pass unused keys from YAML
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction
        )

        # Node features (observation space per agent)
        # shape: [obs_dim]
        in_channels = observation_space.shape[0]
        out_channels = action_space.shape[0]
        # First layer: multi-head concat → mid_dim channels per node
        mid_dim = (hidden_channels // 2) * num_heads

        self.conv1 = GATv2Conv(
            in_channels, hidden_channels // 2, heads=num_heads, concat=True
        )
        self.conv2 = GATv2Conv(mid_dim, hidden_channels, heads=1, concat=False)

        # Action Head
        self.action_head = nn.Linear(hidden_channels, out_channels)
        self.log_std_parameter = nn.Parameter(torch.ones(out_channels) * initial_log_std)

    def compute(self, inputs, role):
        # shape: [num_envs * num_agents, obs_dim]
        if isinstance(inputs, dict):
            # SKRL provides tensors in a dict-like structure; cast for static typing.
            obs = cast(torch.Tensor, inputs["states"])
            extras: Any = inputs.get("extras", {})
            if isinstance(extras, dict):
                adj_matrix = cast(torch.Tensor | None, extras.get("adj_matrix", None))
            else:
                adj_matrix = None
        else:
            # If SKRL ever passes states directly, treat `inputs` as `obs`.
            obs = cast(torch.Tensor, inputs)
            adj_matrix = None

        # Enforce 2D shape for torch_geometric
        if obs.dim() > 2:
            obs = obs.reshape(-1, obs.shape[-1])

        # Fetch the adjacency matrix passed from the environment extras
        # shape expected: [num_envs, num_agents, num_agents]
        if adj_matrix is not None:
            # Flatten the batched 3D adjacency matrix to a 2D sparse edge_index
            edge_index = adjacency_to_edge_index(adj_matrix)
        else:
            # Fallback if no adj_matrix is provided (e.g., self-loops only)
            num_nodes = obs.shape[0]
            edge_index = torch.arange(num_nodes, device=self.device).repeat(2, 1)

        # Forward pass through GNN
        # shape: [num_envs * num_agents, hidden_channels]
        x = torch.relu(self.conv1(obs, edge_index))
        x = torch.relu(self.conv2(x, edge_index))

        # Output actions
        # shape: [num_envs * num_agents, action_dim]
        action_mean = self.action_head(x)

        return action_mean, self.log_std_parameter, {}
