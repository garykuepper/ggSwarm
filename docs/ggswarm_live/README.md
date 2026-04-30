# ggSwarm Live

Real-hardware deployment program. Takes the v1 GATv2/PPO policy out of
simulation, onto PX4-based airframes, and delivers it as the adaptive
execution layer underneath Skybrush drone-light-show choreography.

The capstone (`v1.0.0-capstone`, frozen at [`../capstone/`](../capstone/))
established the simulation foundation. ggSwarm Live builds the path from
there to a paying drone-show product.

## Where to start

- [Vision and program-level requirements](vision.md)
- [Target architecture (Skybrush + RL overlay on PX4)](architecture.md)
- [Phase plan](phases/) — Phase 0 is the capstone baseline; Phases 1–7 are active or planned
- [Backlog](backlog.md) — capstone deferrals plus new program work, mapped to phases
- [References](references.md)

## Phases

| Phase | Title | Status |
| :--- | :--- | :--- |
| 0 | [Capstone Baseline](phases/phase0_capstone_baseline.md) | Complete (v1.0.0-capstone) |
| 1 | [Shared-Scene Multi-Drone Training (sim only)](phases/phase1_shared_scene_sim.md) | Planned |
| 2 | [Sim-to-Real Baseline (Crazyflie + LPS)](phases/phase2_sim2real_baseline.md) | Planned |
| 3 | [Decentralized Assignment + Peer Ranging](phases/phase3_decentralized.md) | Planned |
| 4 | [Scale + Expressive Shapes](phases/phase4_scale_shapes.md) | Planned |
| 4b | [Drone Show Capability (revenue stream)](phases/phase4b_drone_shows.md) | Planned |
| 5 | [Outdoor + Extended Fault Tolerance](phases/phase5_outdoor_faults.md) | Planned |
| 6 | [Onboard Compute + Obstacle-Aware Navigation](phases/phase6_onboard_obstacles.md) | Planned |
| 7 | [Hardware-Agnostic and General-Purpose (stretch)](phases/phase7_hardware_agnostic.md) | Stretch |

## Status

- [changelog](status/changelog.md)
- [log](status/log.md)
