# Phase 1: Foundation

Phase 1 of the ggSwarm project establishes the foundational Isaac Lab MARL environment: spawning multirotor assets, applying physics, collecting observations, and computing the graph connectivity layer (L2) for the swarm.

**Source Path:** `source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/`

---

## Core Files

| File | Purpose |
| :--- | :--- |
| `drone_swarm_env.py` | Main environment class (`GgswarmMarlEnv`). Scene setup, physics, observations, rewards, resets. |
| `drone_swarm_env_cfg.py` | Configuration dataclass (`GgswarmMarlEnvCfg`). Agent count, spaces, reward scales, physics params. |
| `__init__.py` | Gym registration under `Template-Ggswarm-Marl-Direct-v0`. |
| `agents/skrl_mappo_cfg.yaml` | SKRL MAPPO hyperparameters for training. |

---

## Environment Architecture

### Scene Setup (`_setup_scene`)

1. **Drone Spawning:** Iterates `num_agents` times, spawning each `CRAZYFLIE_CFG` asset at `/World/envs/env_0/drone_{i}`.
2. **Ground Plane:** Adds a `GroundPlaneCfg` at `/World/ground`.
3. **Environment Cloning:** Calls `scene.clone_environments()` to replicate the source drones across all `num_envs` parallel environments.
4. **Articulation Init:** Creates a single `Articulation` view using the regex prim path `/World/envs/env_.*/drone_.*`, which captures all drone instances.
5. **Lighting:** Adds a dome light (intensity 2000.0).

### Configuration (`GgswarmMarlEnvCfg`)

| Parameter | Value | Description |
| :--- | :--- | :--- |
| `num_agents` | 4 (configurable) | Number of Crazyflie drones per environment |
| `decimation` | 2 | Physics steps per policy step |
| `episode_length_s` | 10.0 | Episode duration in seconds |
| `sim.dt` | 1/100 | Physics timestep |
| `scene.num_envs` | 32 | Parallel environment instances |
| `scene.env_spacing` | 5.0 | Meters between environment origins |
| `thrust_to_weight` | 1.9 | Thrust-to-weight ratio for action scaling |
| `moment_scale` | 0.01 | Torque scaling factor |

Dynamic agent registration happens in `__post_init__()`:

- `possible_agents` → `["drone_0", "drone_1", ..., "drone_N"]`
- `action_spaces` → `{agent: 4}` for each agent
- `observation_spaces` → `{agent: 12}` for each agent

---

## Observation Space (12-dim per agent)

Each agent receives a 12-dimensional observation vector, all in body frame:

| Dims | Content | Source |
| :--- | :--- | :--- |
| 0–2 | Linear velocity | `root_lin_vel_b` |
| 3–5 | Angular velocity | `root_ang_vel_b` |
| 6–8 | Projected gravity | `projected_gravity_b` |
| 9–11 | Relative position to goal | `subtract_frame_transforms(pos, quat, target)` |

## Action Space (4-dim per agent)

| Dim | Mapping |
| :--- | :--- |
| 0 | Collective thrust (mapped `[-1,1]` → `[0, thrust_to_weight * weight]`, split equally across 4 props) |
| 1–3 | Roll / Pitch / Yaw moments (scaled by `moment_scale`, split across 4 props) |

---

## Graph Connectivity (L2)

Computed every step inside `_get_observations()`:

1. **Pairwise distances:** `diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)` → `dist = torch.norm(diff, dim=-1)`
2. **Adjacency threshold:** `adj_matrix = (dist < 2.0).float()` — drones within 2.0m are "connected."
3. **Storage:** Placed in `self.extras["adj_matrix"]` for downstream consumption (GATv2 in Phase 2).

Shape: `[num_envs, num_agents, num_agents]`

---

## Reward Function

| Component | Scale | Formula |
| :--- | :--- | :--- |
| Position (Gaussian) | `+1.0` | `exp(-dist_to_goal / 0.5)` |
| Linear velocity penalty | `-0.05` | `‖lin_vel_b‖` |
| Angular velocity penalty | `-0.01` | `‖ang_vel_b‖` |
| Alive bonus | `+0.1` | Constant per step |

## Termination Conditions

- **Out of bounds:** Drone altitude below `0.1m` or above `3.0m`.
- **Timeout:** Episode exceeds `episode_length_s`.

---

## Reset Logic (`_reset_idx`)

1. **Random positions:** Uniformly sampled within `±spawn_dist` (1.5m) of the environment origin, Z height between 0.5–1.5m.
2. **Random yaw:** Uniform in `[-π, π]`.
3. **Goals:** Set to the initial spawn position (hover-in-place for Phase 1).
4. **Joints:** Reset rotor positions and velocities to defaults.

---

## Demo Script

**Script:** `scripts/phase1_demo.py`

```powershell
..\IsaacLab\isaaclab.bat -p scripts\phase1_demo.py --task=Template-Ggswarm-Marl-Direct-v0
```

**What it does:**

1. Spawns `num_agents` Crazyflie drones with random actions.
2. Prints live adjacency matrix statistics: `"[Phase 1 Evidence] Drones connected (Distance < 2.0m): N pairs"`.
3. Provides visual confirmation (Isaac Sim 3D render) alongside mathematical proof of L2 graph connectivity.

---

## Phase 1 Completion Summary

| GNSC Layer | Status | Implementation |
| :--- | :--- | :--- |
| **L1: Local Sensing** | ✅ Complete | Body-frame velocity, gravity, and relative position observations |
| **L2: GNN Messaging** | ✅ Foundation | Distance-based adjacency matrix computed; GATv2 integration in Phase 2 |
| **L3: Consensus** | ⬜ Phase 3 | — |
| **L4: Safety Shield** | ⬜ Phase 3 | — |
| **L5: Execution** | ✅ Basic | Thrust/moment force application to propellers |
