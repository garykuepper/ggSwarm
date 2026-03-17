# Phase 2: Brain Development (Weeks 7–8)

Phase 2 trains the GATv2 coordination policy using MAPPO so agents learn basic formation keeping in open space. This phase covers GNSC Layer 2 (GNN Message Passing) and Layer 3 groundwork.

---

## Objectives

| ID | Objective | Success Criteria |
| :--- | :--- | :--- |
| P2.1 | Train a shared MAPPO policy that keeps agents in a stable formation | Mean formation error < 0.5m within 50k timesteps |
| P2.2 | Integrate GATv2 as the policy backbone | Policy consumes adjacency matrix from `extras["adj_matrix"]` |
| P2.3 | Evolve rewards from hover-in-place to formation-aware | Agents maintain target inter-agent spacing |
| P2.4 | Validate training pipeline end-to-end | TensorBoard logs show converging reward curve |

Aligns with proposal **Milestone M1 (Week 8):** "GNN policy training."

---

## Architecture Changes

### Custom GATv2 Policy Network

Replace the current MLP policy (`[32, 32]` layers) with a GATv2-based network:

```
Input: agent_obs (12-dim) + neighbor_obs (via adjacency matrix)
  │
  ▼
GATv2Conv (K-hop, max 3 hops)
  │
  ▼
MLP Head → Actions (4-dim)
```

**Integration points:**

- `drone_swarm_env.py` already stores `self.extras["adj_matrix"]` with shape `[num_envs, num_agents, num_agents]`.
- The custom model must implement SKRL's `GaussianMixin` interface for PPO compatibility.
- Edge construction: use the adjacency matrix to define which agents exchange messages.

### Formation-Aware Rewards

Extend the current reward function with formation components:

| Component | Scale | Formula |
| :--- | :--- | :--- |
| **Formation error** | `+2.0` | `exp(-mean_spacing_error / 0.3)` where spacing error = `|actual_dist - target_dist|` |
| **Cohesion** | `+0.5` | Penalize if any neighbor distance > `connectivity_threshold` |
| Position (existing) | `+1.0` | `exp(-dist_to_goal / 0.5)` |
| Velocity penalty (existing) | `-0.05` | `‖lin_vel_b‖` |
| Angular velocity penalty (existing) | `-0.01` | `‖ang_vel_b‖` |
| Alive bonus (existing) | `+0.1` | Constant |

---

## SKRL Configuration Tuning

Current `skrl_mappo_cfg.yaml` needs these adjustments:

| Parameter | Current | Target | Rationale |
| :--- | :--- | :--- | :--- |
| `network.layers` | `[32, 32]` | `[128, 64]` | Larger capacity for multi-agent coordination |
| `trainer.timesteps` | `4800` | `100000+` | Sufficient training for convergence |
| `experiment.directory` | `cart_double_pendulum_direct` | `ggswarm_marl` | Fix template leftover |
| `agent.rollouts` | `16` | `32` | More experience per update |
| `agent.learning_rate` | `3.0e-04` | `1.0e-04` | Slower, more stable learning for MARL |
| `agent.entropy_loss_scale` | `0.0` | `0.01` | Encourage exploration in early training |

---

## Implementation Plan

### Step 1: Fix Training Pipeline Basics

1. Update `skrl_mappo_cfg.yaml` with corrected parameters (table above).
2. Run a baseline MLP training to confirm the pipeline works end-to-end.
3. Verify TensorBoard logging and checkpoint saving.

### Step 2: Formation Rewards

1. Define target formation geometry (e.g., uniform spacing at 1.0m).
2. Add `_compute_formation_reward()` to `drone_swarm_env.py`.
3. Integrate formation reward into `_get_rewards()`.
4. Validate reward signals make physical sense by inspecting logged values.

### Step 3: GATv2 Policy

1. Create a custom SKRL model class that wraps `torch_geometric.nn.GATv2Conv`.
2. Build edge index from the adjacency matrix each step.
3. Register the custom model in the SKRL config.
4. Train and compare against the MLP baseline.

### Step 4: Evaluation

1. Run trained policy with `play.py` for visual inspection.
2. Log formation error metrics over episodes.
3. Compare MLP vs GATv2 convergence speed and final performance.

---

## Training Workflow

```powershell
# Train with MAPPO (default)
..\IsaacLab\isaaclab.bat -p scripts\skrl\train.py --task=Template-Ggswarm-Marl-Direct-v0 --algorithm=MAPPO

# Train headless (no GUI, faster)
..\IsaacLab\isaaclab.bat -p scripts\skrl\train.py --task=Template-Ggswarm-Marl-Direct-v0 --algorithm=MAPPO --headless

# Evaluate a checkpoint
..\IsaacLab\isaaclab.bat -p scripts\skrl\play.py --task=Template-Ggswarm-Marl-Direct-v0 --checkpoint=<path>

# Monitor training
tensorboard --logdir=logs/skrl/ggswarm_marl
```

---

## Dependencies

- `torch_geometric` — for `GATv2Conv` implementation
- `skrl >= 1.4.3` — MAPPO multi-agent trainer
- `tensorboard` — training visualization

---

## Risks

| Risk | Mitigation |
| :--- | :--- |
| GATv2 over-smoothing with deep layers | Limit to 2–3 attention heads, max 3-hop neighborhood |
| VRAM saturation with 20+ agents | Use headless training; reduce `num_envs` if needed |
| Reward hacking (agents collapse to same point) | Add minimum separation penalty to reward |
| Training instability with formation rewards | Curriculum: start with hover rewards, gradually increase formation weight |
