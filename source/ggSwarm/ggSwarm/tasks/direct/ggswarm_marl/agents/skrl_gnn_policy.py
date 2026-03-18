# Copyright (c) 2022-2026, ggSwarm Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""GNN policy wrapper bridging PyTorch Geometric and SKRL."""

import torch
import torch.nn as nn
from skrl.models.torch import GaussianMixin, Model
from torch_geometric.nn import GATv2Conv


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
        **kwargs,
    ):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(
            self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction
        )

        # Safely extract initial_log_std if provided in kwargs from skrl config
        initial_log_std = kwargs.get("initial_log_std", 0.0)

        # Node features (observation space per agent)
        # shape: [obs_dim]
        in_channels = observation_space.shape[0]
        hidden_channels = 128
        out_channels = action_space.shape[0]

        # GNN Layers (Limit to 2 heads to prevent over-smoothing - Rule 11/Proposal)
        self.conv1 = GATv2Conv(in_channels, hidden_channels // 2, heads=2, concat=True)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=1, concat=False)

        # Action Head
        self.action_head = nn.Linear(hidden_channels, out_channels)
        self.log_std_parameter = nn.Parameter(torch.ones(out_channels) * initial_log_std)

    def compute(self, inputs, role):
        # shape: [num_envs * num_agents, obs_dim]
        obs = inputs["states"]

        # Enforce 2D shape for torch_geometric
        if obs.dim() > 2:
            obs = obs.reshape(-1, obs.shape[-1])

        # Fetch the adjacency matrix passed from the environment extras
        # shape expected: [num_envs, num_agents, num_agents]
        adj_matrix = inputs.get("extras", {}).get("adj_matrix", None)

        if adj_matrix is not None:
            num_envs, num_agents, _ = adj_matrix.shape

            # Flatten the batched 3D adjacency matrix to a 2D sparse edge_index
            # Find all non-zero elements (edges)
            # indices shape: [num_edges, 3] -> (env_idx, agent_i, agent_j)
            indices = adj_matrix.nonzero(as_tuple=False)

            env_idx = indices[:, 0]

            # Shift node indices so each environment's graph is disconnected but in the same batch
            src = indices[:, 1] + (env_idx * num_agents)
            dst = indices[:, 2] + (env_idx * num_agents)

            # shape: [2, num_edges]
            edge_index = torch.stack([src, dst], dim=0)
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
