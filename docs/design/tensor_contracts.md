# Tensor Shape Contracts

All functions manipulating swarm tensors must add a shape comment on first access
(CLAUDE.md Rule 7). This document is the reference for standard shapes.

## Core State Tensors

| Tensor | Shape | Description |
| :--- | :--- | :--- |
| `pos_w` | `[num_envs, num_agents, 3]` | World-frame positions |
| `quat_w` | `[num_envs, num_agents, 4]` | World-frame quaternions |
| `lin_vel_b` | `[num_envs, num_agents, 3]` | Body-frame linear velocity |
| `ang_vel_b` | `[num_envs, num_agents, 3]` | Body-frame angular velocity |
| `proj_grav_b` | `[num_envs, num_agents, 3]` | Projected gravity in body frame (z = -1 when level) |
| `rel_pos_b` | `[num_envs, num_agents, 3]` | Body-frame relative position to goal |

## Action Tensors

| Tensor | Shape | Description |
| :--- | :--- | :--- |
| `actions` (per-agent) | `[num_envs, num_agents, 4]` | `[thrust_cmd, moment_x, moment_y, moment_z]` in `[-1, 1]` |
| `actions` (flat) | `[num_instances, 4]` | Flattened: `num_instances = num_envs * num_agents` |
| `_thrust` | `[num_instances, 1, 3]` | Body-frame thrust (Z only) for `permanent_wrench_composer` |
| `_moment` | `[num_instances, 1, 3]` | Body-frame moments for `permanent_wrench_composer` |

## Graph Tensors

| Tensor | Shape | Description |
| :--- | :--- | :--- |
| `adj_matrix` | `[num_envs, num_agents, num_agents]` | Distance-based adjacency (diagonal zeros, radius-controlled) |
| `edge_index` | `[2, num_edges]` | Sparse graph for PyG (batched envs as disconnected components) |

## Reward Tensors

| Tensor | Shape | Description |
| :--- | :--- | :--- |
| `total_rewards` | `[num_envs, num_agents]` | Per-agent reward signal |
| `terms_dict[key]` | `[num_envs, num_agents]` | Individual reward components for logging |

## GNN Policy (Centralized Forward)

When using `patch_mappo_gnn_batched_act`, observations are batched across all agents:

| Tensor | Shape | Description |
| :--- | :--- | :--- |
| `batched_obs` | `[num_envs * num_agents, obs_dim]` | All agents' preprocessed obs (row-major) |
| `mean_actions` | `[num_envs * num_agents, action_dim]` | GNN output before per-agent splitting |

Row ordering: flat index `i * num_agents + j` = env `i`, agent `j`. This matches
`adjacency_to_edge_index` node indexing.

---

## See Also

- [Architecture](architecture.md) — GNSC 5-layer model and data flow
- CLAUDE.md Rule 7 — mandatory shape comments
- CLAUDE.md Rule 13 — per-step allocation ban
