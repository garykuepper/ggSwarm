# Phase 1: Foundation

**Timeline:** Feb 5 -- Feb 17 (Weeks 5--6)  |  **Gate:** Isaac Lab environment operational with MAPPO training pipeline

**Status: COMPLETE** (2026-02-17). Original Phase 2 codebase from this phase
was archived on 2026-03-26 when the env was rebuilt from the Isaac Lab
quadcopter reference; see [Phase 2](phase2_brain_development.md) for the
fresh-start architecture that superseded the Phase 1 implementation.

## 1. Goals


| ID   | Goal                                     | Success Criteria                                                                    |
| ---- | ---------------------------------------- | ----------------------------------------------------------------------------------- |
| P1.1 | Isaac Lab simulation environment running | Crazyflie drones spawn, physics steps execute, environments clone correctly         |
| P1.2 | 12-dim observation space implemented     | Body-frame velocity, angular velocity, projected gravity, relative-to-goal position |
| P1.3 | 4-dim action space implemented           | Collective thrust + roll/pitch/yaw moments mapped to propeller forces               |
| P1.4 | Graph connectivity layer (L2 foundation) | Distance-based adjacency matrix computed per step, stored in `extras["adj_matrix"]` |
| P1.5 | SKRL MAPPO training pipeline             | End-to-end training loop runs with multi-agent observation/action routing           |
| P1.6 | Curriculum reward structure              | Position-tracking, velocity penalties, alive bonus with configurable scales         |


## 2. Tasks

Set up the Isaac Lab direct-workflow environment in `source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl/`. Key files:

- `drone_swarm_env.py` -- main env class (`GGSwarmMarlEnv`): scene setup, observations, rewards, resets.
- `drone_swarm_env_cfg.py` -- env config (`GGSwarmMarlEnvCfg`): agent count, spaces, reward scales, physics params.
- `__init__.py` -- Gym registration under `Template-GGSwarm-Marl-Direct-v0`.
- `agents/skrl_mappo_cfg.yaml` -- SKRL MAPPO hyperparameters.

Scene setup spawns `num_agents` Crazyflie assets, adds a ground plane, clones environments, creates an `Articulation` view via regex prim path, and adds lighting.

Reset logic samples random positions within spawn distance, random yaw, and sets goals to spawn positions (hover-in-place).

Termination: out-of-bounds (altitude below 0.1 m or above 3.0 m) or episode timeout.

## 3. Design Integration

Phase 1 establishes the bottom layers of the GNSC architecture:

- **L1 (Local Sensing):** body-frame observations (12-dim vector per agent).
- **L2 (GNN Messaging) foundation:** pairwise-distance adjacency matrix, shape `[num_envs, num_agents, num_agents]`. GATv2 integration deferred to Phase 2.
- **L5 (Execution) basic:** thrust/moment force application to propellers.

Config parameters introduced:


| Parameter          | Default | Description                               |
| ------------------ | ------- | ----------------------------------------- |
| `num_agents`       | 4       | Crazyflie drones per environment          |
| `decimation`       | 2       | Physics steps per policy step             |
| `episode_length_s` | 10.0    | Episode duration (seconds)                |
| `sim.dt`           | 1/100   | Physics timestep                          |
| `scene.num_envs`   | 32      | Parallel environment instances            |
| `thrust_to_weight` | 1.9     | Thrust-to-weight ratio for action scaling |
| `moment_scale`     | 0.01    | Torque scaling factor                     |


Cross-references: `docs/design/architecture.md` for the full GNSC layer diagram.

## 4. Results

**COMPLETE.** Phase 1 delivered:

- 12-dim body-frame observation space and 4-dim thrust/moment action space.
- Distance-based adjacency matrix (`extras["adj_matrix"]`, shape `[num_envs, num_agents, num_agents]`).
- Curriculum reward structure: Gaussian position tracking (+1.0), velocity penalty (-0.05), angular velocity penalty (-0.01), alive bonus (+0.1).
- Full SKRL MAPPO integration with multi-agent observation/action routing.
- Gym-registered task ID `Template-GGSwarm-Marl-Direct-v0`.


| GNSC Layer        | Status     | Implementation                                               |
| ----------------- | ---------- | ------------------------------------------------------------ |
| L1: Local Sensing | Complete   | Body-frame velocity, gravity, relative position observations |
| L2: GNN Messaging | Foundation | Distance-based adjacency matrix; GATv2 in Phase 2            |
| L3: Consensus     | Phase 3    | --                                                           |
| L4: Safety Shield | Phase 3    | --                                                           |
| L5: Execution     | Basic      | Thrust/moment force application to propellers                |


---

## See Also

- `docs/design/architecture.md` -- GNSC 5-layer architecture
- `docs/phases/phase2_brain.md` -- Phase 2: GATv2 policy and formation training

