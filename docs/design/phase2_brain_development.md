# Phase 2: Brain Development (Weeks 7–8)

Phase 2 trains the GATv2 coordination policy using MAPPO so agents learn basic formation keeping in open space. This phase covers GNSC Layer 2 (GNN Message Passing) and Layer 3 groundwork.

---

## Objectives

| ID | Objective | Success Criteria |
| :--- | :--- | :--- |
| P2.1 | Train a shared MAPPO policy that keeps agents in a stable formation | **Mean formation error < 0.5m** (defined below) after **≤ 50k environment steps** |
| P2.2 | Integrate GATv2 as the policy backbone | Policy handles batched graphs from `extras["adj_matrix"]` |
| P2.3 | Implement Curriculum Reward Shaping | Smooth transition from hover-in-place to formation-aware |
| P2.4 | Validate training pipeline end-to-end | TensorBoard logs show converging reward curve without collapsing |

Aligns with proposal **Milestone M1 (Week 8):** "GNN policy training."

> **Note on proposal targets:** The proposal's steady-state objective of **< 0.1m**
> formation error is a *project-level* target and is expected to be reached after
> later phases (e.g., safety constraints, trajectory smoothing, stress testing).
> Phase 2's goal is to establish a working GNN coordination policy and
> demonstrate basic formation keeping.

---

## Phase 2 Definition of "Mean Formation Error"

For an environment with \(N\) agents and desired spacing `target_formation_dist`, define the per-step spacing error as the mean absolute pairwise spacing error over unique pairs:

\[
e_t = \frac{2}{N(N-1)} \sum_{i < j} \left| d_{ij}(t) - d^* \right|
\]

Where \(d_{ij}(t)\) is the Euclidean distance between agents \(i\) and \(j\) at step \(t\), and \(d^* =\) `target_formation_dist`.

The **mean formation error** for an evaluation run is the average of \(e_t\) over all evaluation steps across all evaluation episodes.

### Evaluation Procedure (Pass/Fail)

- **Checkpoint**: evaluate `best_agent.pt` (or the newest `agent_*.pt`) under
  `logs/skrl/ggswarm_marl/**/checkpoints/`.
- **Episodes**: 10 evaluation episodes.
- **Episode length**: use the environment's configured episode length.
- **Pass condition**: mean formation error < 0.5m.
- **Secondary metrics (sanity)**:
  - separation event rate (fraction of steps where any pair violates minimum separation)
  - mean linear speed \(\|\mathrm{lin\_vel}\|\) (jitter proxy)

Implementation lives in `scripts/eval_phase2.py`.

---

## Architecture Changes

### Custom GATv2 Policy Network

Replace the current MLP policy (`[32, 32]` layers) with a GATv2-based network that bridges PyTorch Geometric and SKRL.

```text
Input: agent_obs (12-dim) + extras["adj_matrix"]
  │
  ▼ (Batched 3D Adjacency Flattened to 2D Sparse Edge Index)
  │
GATv2Conv (2 attention heads, max 2 hops)
  │
  ▼
MLP Head → Actions (4-dim)
```

**Curriculum-Based Rewards**
Rewards dynamically scale based on training progress to prevent early-stage training collapse.

| Component | Scale | Formula |
| :--- | :--- | :--- |
| **Separation Penalty** | `-5.0` | Applied if `dist < 2 * drone_radius` (prevents physical clipping/collapse) |
| **Formation error** | `+2.0 * α` | `exp(-mean_spacing_error / 0.3)` where spacing error = `\|actual_dist - target_dist\|` |
| **Cohesion** | `+0.5 * α` | `exp(-max_neighbor_dist / connectivity_threshold)` |
| Position (Hover) | `+1.0 * (1-α)` | `exp(-dist_to_goal / 0.5)` |
| Velocity penalty | `-0.05` | `‖lin_vel_b‖` |
| Alive bonus | `+0.1` | Constant |

*(Where `α` scales from 0.0 to 1.0 between `curriculum_start_step` and `curriculum_end_step` in `GGSwarmMarlEnvCfg`, currently 10k → 50k environment steps).*

---

## SKRL Configuration Tuning (`skrl_mappo_cfg.yaml`)

| Parameter | Current | Target | Rationale |
| :--- | :--- | :--- | :--- |
| `network.layers` | `[32, 32]` | `[128, 64]` | Larger capacity for multi-agent coordination |
| `trainer.timesteps` | `4800` | `100000+` | Sufficient training for convergence |
| `experiment.directory` | `cart_double_pendulum_direct` | `ggswarm_marl` | Fix template leftover |
| `agent.rollouts` | `16` | `32` | More experience per update |
| `agent.mini_batches` | `(default)` | `4` or `8` | Prevents memory spikes with larger rollouts |
| `agent.learning_rate` | `3.0e-04` | `1.0e-04` | Slower, more stable learning for MARL |
| `agent.entropy_loss_scale` | `0.0` | `0.01` | Encourage exploration in early training |

---

## Phase 2 Sub-phases

Phase 2 is split into three sequential sub-phases. Each sub-phase has its own
CLI family, cfg class, and gym task. Advance to the next sub-phase only after
the current assess gate passes (Rule 20).

| Sub-phase | CLI command | Task ID | Config class | Iterations | Gate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A: Hover-Stability | `hover-stability train` | `Template-GGSwarm-Marl-HoverStability-v0` | `GGSwarmMarlHoverStabilityCfg` | 80k | survival_steps > 500, airborne_ratio > 0.9, mean_roll < 15° |
| B: Formation | `phase2b train --checkpoint <Phase_A/best_agent.pt>` | `Template-GGSwarm-Marl-Formation-v0` | `GGSwarmMarlFormationCfg` | 120k | formation_error < 0.5m, stability metrics maintained |
| C: Perturbation | future — `# TODO (Phase C)` | TBD | TBD | TBD | TBD |

**Phase A → B handoff:** pass `best_agent.pt` from the Phase A run via `--checkpoint`.
Per Rule 21, `curriculum_start_step` is set to `0` in `GGSwarmMarlFormationCfg`
because `common_step_counter` resets to 0 on env re-init regardless of checkpoint.

**Phase B → C:** placeholder only. Do not implement until Phase B assess gate passes (Rule 2).

---

## Code Implementation

Here is the core logic needed to execute the riskiest parts of the plan: the PyG to SKRL bridge and the curriculum reward logic.

### 1. The GNN Policy Wrapper (`skrl_gnn_policy.py`)

This handles the critical task of converting the dense 3D adjacency matrix from your Isaac Lab environment into the 2D sparse graph format required by PyTorch Geometric, all while satisfying SKRL's `GaussianMixin` requirement.

```python
import torch
import torch.nn as nn
from skrl.models.torch import Model, GaussianMixin
from torch_geometric.nn import GATv2Conv

class GGSwarmGNNPolicy(GaussianMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)

        # Node features (observation space per agent)
        in_channels = observation_space.shape[0] 
        hidden_channels = 128
        out_channels = action_space.shape[0] # 4-dim action

        # GNN Layers (Limit to 2 heads to prevent over-smoothing)
        self.conv1 = GATv2Conv(in_channels, hidden_channels // 2, heads=2, concat=True)
        self.conv2 = GATv2Conv(hidden_channels, hidden_channels, heads=1, concat=False)
        
        # Action Head
        self.action_head = nn.Linear(hidden_channels, out_channels)
        self.log_std_parameter = nn.Parameter(torch.zeros(out_channels))

    def compute(self, inputs, role):
        obs = inputs["states"] # Shape: [num_envs * num_agents, obs_dim]
        
        # Fetch the adjacency matrix passed from the environment extras
        # Shape expected: [num_envs, num_agents, num_agents]
        adj_matrix = inputs.get("extras", {}).get("adj_matrix", None)
        
        if adj_matrix is not None:
            num_envs, num_agents, _ = adj_matrix.shape
            
            # Flatten the batched 3D adjacency matrix to a 2D sparse edge_index
            # Find all non-zero elements (edges)
            indices = adj_matrix.nonzero(as_tuple=False) # Shape: [num_edges, 3] -> (env_idx, agent_i, agent_j)
            
            env_idx = indices[:, 0]
            
            # Shift node indices so each environment's graph is disconnected but in the same batch
            src = indices[:, 1] + (env_idx * num_agents)
            dst = indices[:, 2] + (env_idx * num_agents)
            
            edge_index = torch.stack([src, dst], dim=0) # Shape: [2, num_edges]
        else:
            # Fallback if no adj_matrix is provided (e.g., self-loops only)
            num_nodes = obs.shape[0]
            edge_index = torch.arange(num_nodes, device=self.device).repeat(2, 1)

        # Forward pass through GNN
        x = torch.relu(self.conv1(obs, edge_index))
        x = torch.relu(self.conv2(x, edge_index))
        
        # Output actions
        action_mean = self.action_head(x)
        
        return action_mean, self.log_std_parameter, {}
```

### 2. Curriculum Reward Logic (`drone_swarm_env.py`)

This goes into your environment's reward computation block to ensure the agents learn to hover before they are penalized for formation errors.

```python
    def _get_rewards(self) -> torch.Tensor:
        # ... existing state extraction ...

        # 1. Define Curriculum Scale (alpha)
        # Assuming self.common_step_counter tracks total environment steps
        start_curriculum = 25000.0
        end_curriculum = 50000.0
        
        alpha = torch.clamp(
            (self.common_step_counter - start_curriculum) / (end_curriculum - start_curriculum),
            min=0.0, 
            max=1.0
        )

        # 2. Separation Penalty (ALWAYS ON)
        # Prevents physical clipping regardless of curriculum stage
        # Assuming inter_agent_distances is pre-calculated
        collision_mask = inter_agent_distances < (2 * self.drone_radius)
        separation_penalty = -5.0 * collision_mask.sum(dim=1)

        # 3. Position (Hover) Reward (Fades OUT)
        # Encourages staying near the global target
        hover_reward = 1.0 * torch.exp(-dist_to_goal / 0.5) * (1.0 - alpha)

        # 4. Formation Reward (Fades IN)
        # target_dist is your defined formation spacing
        spacing_error = torch.abs(inter_agent_distances - self.target_dist)
        mean_spacing_error = spacing_error.mean(dim=1)
        formation_reward = 2.0 * torch.exp(-mean_spacing_error / 0.3) * alpha

        # 5. Cohesion Reward (Fades IN)
        max_neighbor_dist, _ = torch.max(inter_agent_distances, dim=1)
        cohesion_reward = 0.5 * torch.exp(-max_neighbor_dist / self.connectivity_threshold) * alpha

        # ... compute velocity penalties and alive bonus ...

        total_reward = (
            separation_penalty + 
            hover_reward + 
            formation_reward + 
            cohesion_reward +
            alive_bonus +
            vel_penalties
        )

        return total_reward
```

---

## Training Workflow

```powershell
# Phase A: hover-stability (run on GCE via train_and_push.sh)
python scripts/run.py hover-stability train --headless --max_iterations 80000

# After GCS pull — assess locally (Rule 20):
python scripts/run.py hover-stability assess --run_dir logs/skrl/ggswarm_marl/<run>

# Phase B: formation resume (run on GCE, pass Phase A checkpoint)
python scripts/run.py phase2b train --headless --max_iterations 120000 --checkpoint logs/skrl/ggswarm_marl/<phase_a_run>/checkpoints/best_agent.pt

# After GCS pull — assess locally (Rule 20):
python scripts/run.py phase2b assess --run_dir logs/skrl/ggswarm_marl/<run>

# Eval and play are always local (GCE is training-only — see gce-training-ops rule):
python scripts/run.py phase2b eval --checkpoint logs/skrl/ggswarm_marl/<run>/checkpoints/best_agent.pt
python scripts/run.py phase2b play --checkpoint logs/skrl/ggswarm_marl/<run>/checkpoints/best_agent.pt
tensorboard --logdir logs/skrl/ggswarm_marl
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
