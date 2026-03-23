# Architecture: ggSwarm Decentralized Drone Coordination

## 1. Overview

ggSwarm is a decentralized formation control framework for large-scale Unmanned
Aerial Vehicle (UAV) swarms, built on the NVIDIA Isaac Lab simulation platform.
It follows the **Graph Neural Swarm Control (GNSC)** 5-Layer model with a
**Centralized Training, Decentralized Execution (CTDE)** workflow.

## 2. GNSC 5-Layer Model Mapping

| Layer | Responsibility | Implementation Component | Phase |
| :--- | :--- | :--- | :--- |
| **L1: Local Sensing** | LiDAR/IMU data collection | `GGSwarmMarlEnv` perception buffers (12-dim obs) | ✅ Phase 1 |
| **L2: GNN Messaging** | Spatial awareness / GNN | Distance-based adjacency matrix → GATv2 policy | ✅ Phase 2 |
| **L3: Consensus** | Formation alignment + fault recovery | `SwarmRaft` — heartbeat, leader election, redistribution | ✅ Phase 3 |
| **L4: Safety Shield** | Collision avoidance | Control Barrier Functions (CBF) — `cbf_safety.py` | ✅ Phase 3 |
| **L5: Execution** | Trajectory following + smoothing | Thrust/moment force application + MINCO EMA smoother | ✅ Phase 3 |

## 3. Data Flow

### Phase 2 (baseline)

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│  L1: Sensing │───▶│ L2: Adjacency│───▶│ L2: GATv2    │───▶│ L5: Force│
│  12-dim obs  │    │  Matrix      │    │  Policy      │    │  Control │
│  per agent   │    │  [N×N]       │    │  (MAPPO/PPO) │    │  4-dim   │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────┘
```

### Phase 3 (full GNSC stack)

```
┌──────────────┐   ┌──────────────┐   ┌──────────────┐
│  L1: Sensing  │──▶│ L2: GATv2    │──▶│ raw actions  │
│  12/14-dim    │   │  Policy      │   │  [N×4]       │
└──────────────┘   └──────────────┘   └──────┬───────┘
        ▲                                     │
        │                              ┌──────▼───────┐
        │                              │ L4: CBF      │
        │                              │  cbf_safety  │
        │                              └──────┬───────┘
        │                                     │ safe actions
        │                              ┌──────▼───────┐
        │                              │ L5: MINCO    │
        │                              │  EMA Smoother│
        │                              └──────┬───────┘
        │                                     │ smooth actions
┌───────┴──────┐                      ┌──────▼───────┐
│ L3: SwarmRaft│                      │ L5: Force    │
│  swarm_raft  │──▶ _desired_pos_w    │  Control     │
└──────────────┘                      └──────────────┘
```

1. **Perception:** Each agent gathers local state (lin_vel, ang_vel, gravity, rel_pos_to_goal).
   With `raft_enabled=True`, two consensus dims are appended (is_leader, num_alive_frac → 14-dim).
2. **Adjacency:** A distance-based graph (threshold 2.0m) defines message-passing edges.
3. **Policy:** GATv2 processes graph-structured state to output raw control actions.
4. **CBF:** Pairwise Control Barrier Functions project unsafe actions onto the safe half-space,
   guaranteeing zero inter-agent collisions when active.
5. **MINCO:** EMA action smoother reduces velocity jitter (≥ 20% reduction target).
6. **SwarmRaft:** On agent loss, the leader recomputes formation slots for surviving agents
   and updates `_desired_pos_w`; target is re-sync within 2.0 s.
7. **Control:** The PD attitude controller converts attitude commands to body-frame thrust +
   moments, applied to the main body via a single `permanent_wrench_composer` call.

**Action contract (Phase 2+):** The RL policy outputs 4-dim actions
`[thrust_cmd, desired_roll, desired_pitch, desired_yaw_rate]` intended in `[-1, 1]`.
`GGSwarmMarlEnv._pre_physics_step` **clamps** the stacked tensor to `[-1, 1]` before
CBF/MINCO and the PD loop, so Gaussian exploration cannot command out-of-range thrust
or tilt setpoints even when `clip_actions: False` in SKRL YAML.
An inner-loop PD attitude controller (`attitude_controller.py`) converts these to
body-frame thrust force and moments each physics step.
This matches real Crazyflie flight controller architecture (Bitcraze cascaded PID)
and the OmniDrones deployment pattern. The policy focuses on navigation/coordination;
raw flight dynamics are handled by the deterministic controller.

**Training telemetry (`extras["log"]` → TensorBoard):** SKRL’s MAPPO trainer logs
`infos["log"]` only when each value is a **0-dim `torch.Tensor`** on the training device
(`isinstance(v, torch.Tensor)` and `v.numel() == 1`). Python `float`s are **not** logged.
Populate per-step diagnostics as tensor scalars — e.g. `terms_dict["rew_pos"].mean()` — so
they appear as **`Info / rew_pos`**, **`Info / rew_vel`**, **`Info / rew_ang_vel`**,
**`Info / rew_low_clearance`**, **`Info / rew_terminated`**, **`Info / mean_world_z`**,
**`Info / low_clearance_frac`**, **`Info / curriculum_alpha`**, plus optional action telemetry
and CBF rates when enabled.

**Previous action contract (Runs 1–A1, now retired):** Policy output raw torques
(`moment_scale * action[1:4]`). This required the policy to simultaneously learn
flight dynamics and navigation, which proved unlearnable in 4 consecutive failed runs.

**Phase 2 prerequisite baseline:** `GGS-Hover-v0` trains a single drone to hold
its spawn pose before multi-agent formation tuning. This task keeps the same
observation/action interfaces but disables formation objectives.

**Phase 2 sub-phases:** Phase 2 is split into Phase A (hover-stability, formation OFF,
`Template-GGSwarm-Marl-HoverStability-v0`) and Phase B (formation resume,
`Template-GGSwarm-Marl-Formation-v0`). Both use the same 12-dim obs space and
GATv2 policy — reward path differs: Phase 2A sets `use_stable_hover_rewards=True`
(`compute_stable_hover_rewards` in `contract_logic.py`: tanh position, squared velocity,
`step_dt`-scaled, plus optional low-clearance). Phase B uses `compute_marl_rewards`
(Gaussian position, L2 velocity norms, curriculum/formation terms).
Phase 2A adds optional **low-clearance shaping** (`rew_scale_low_clearance`, `low_clearance_margin_m` in cfg) so training
penalizes time spent below the same altitude band used in eval (`min_height + margin`).
Optional **`rew_scale_terminated`** (hover-stability cfg) adds a **dense** penalty each step while `z < min_height`
(see `compute_stable_hover_rewards` in `contract_logic.py`).
Phase B resumes the Phase A `best_agent.pt` checkpoint via `--checkpoint`.
Phase C (perturbation) is a future placeholder — see `GGSwarmMarlFormationCfg`.

**Phase 3 backward compatibility:** All Phase 3 features are `False` by default in
`GGSwarmMarlEnvCfg`. The `Template-GGSwarm-Marl-Direct-v0` task (Phase 2) is
unchanged. Use `Template-GGSwarm-Marl-Phase3-v0` (or override config flags) to
enable Phase 3 features.

## 4. Training Pipeline

| Component | Technology |
| :--- | :--- |
| Framework | NVIDIA Isaac Lab 2.3 / Isaac Sim 5.1 |
| RL Library | SKRL (MAPPO agent) |
| Policy | GATv2Conv (PyTorch Geometric) |
| Optimizer | PPO with KL-adaptive learning rate |
| Compute | Local RTX 3070 (dev) / Cloud GPU (heavy training) |

Single-agent MARL with MAPPO (e.g. `GGS-Hover-v0`): SKRL's sequential trainer uses a
code path that does not populate `infos['shared_states']`; `scripts/skrl/train.py`
injects them from `DirectMARLEnv.state()` when `num_agents == 1` so the centralized
critic receives valid inputs.

## 5. Key Files

| File | Purpose |
| :--- | :--- |
| `drone_swarm_env.py` | MARL environment (scene, physics, obs, rewards, resets) |
| `drone_swarm_env_cfg.py` | Env configs: base `GGSwarmMarlEnvCfg`, Phase A `GGSwarmMarlHoverStabilityCfg`, Phase B `GGSwarmMarlFormationCfg`, Phase 3/4 variants |
| `attitude_controller.py` | Pure-torch PD attitude controller inner loop (no Isaac imports) |
| `contract_logic.py` | Pure-torch reward logic and adjacency matrix computation |
| `cbf_safety.py` | **Phase 3** L4: CBF pairwise safety projection |
| `swarm_raft.py` | **Phase 3** L3: SwarmRaft consensus and formation redistribution |
| `minco_trajectory.py` | **Phase 3** L5: EMA trajectory smoother |
| `drone_hover_env.py` | Hover-only baseline env (`GGS-Hover-v0`) with spawn-hold reward |
| `drone_hover_env_cfg.py` | Hover config (single-agent + ground-hit penalty params) |
| `agents/skrl_mappo_cfg.yaml` | SKRL MAPPO hyperparameters |
| `agents/skrl_mappo_hover_cfg.yaml` | SKRL MAPPO hyperparameters for hover baseline |
| `agents/skrl_gnn_policy.py` | GATv2 GNN policy wrapper (PyG bridge) |
| `scripts/skrl/train.py` | Training entry point |
| `scripts/skrl/play.py` | Evaluation / playback entry point |
| `scripts/eval_hover.py` | Hover baseline metrics and pass/fail evaluation |
| `scripts/eval_phase2.py` | Phase 2 formation metrics evaluation |
| `scripts/eval_phase3.py` | **Phase 3** CBF / SwarmRaft / MINCO evaluation (O1–O3) |
| `scripts/run.py` | Unified helper CLI: hover, hover-stability (A), phase2b (B), phase2, phase3, phase4, phase5, debug |

---

*Note: This document is maintained as a project rule. All structural changes must be reflected here.*
