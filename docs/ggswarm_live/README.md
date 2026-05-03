# ggSwarm Live

Real-hardware deployment program. Takes the v1 GATv2/PPO policy out of
simulation, onto PX4-based airframes, and delivers it as the adaptive
execution layer underneath Skybrush drone-light-show choreography.

The capstone (`v1.0.0-capstone`, frozen at [`../capstone/`](../capstone/))
established the simulation foundation. ggSwarm Live builds the path from
there to a paying drone-show product.

**Program shape:** sim phases first (exhaust algorithms before hardware),
then a lean hardware bring-up block (one variable per phase), then the
program's major milestone — the **first paid drone show (Phase 14)** —
then post-show vision phases.

## Where to start

- [Vision and program-level requirements](vision.md)
- [Target architecture (Skybrush + RL overlay on PX4)](architecture.md)
- [Phase plan](phases/) — Phase 0 capstone baseline; Phases 1–14 active or planned; 15–18 post-show vision
- [Consensus mechanisms reference](consensus_mechanisms.md) — what we use, and why not blockchain / not Raft
- [Backlog](backlog.md) — capstone deferrals plus new program work, mapped to phases
- [References](references.md)

## Phases

### Sim block (exhaust algorithms first)

| Phase | Title | Status |
| :--- | :--- | :--- |
| 0 | [Capstone Baseline](phases/phase0_capstone_baseline.md) | Complete (v1.0.0-capstone) |
| 1 | [Shared-Scene Multi-Drone Sim Training](phases/phase1_shared_scene_sim.md) | 1a complete; 1b/1c planned |
| 2 | [Decentralized + Fault-Tolerant Stack (sub-phased)](phases/phase2_decentralized.md) | Planned |
| 2a | &nbsp;&nbsp;[Decentralized Localization](phases/phase2a_localization.md) | Planned |
| 2b | &nbsp;&nbsp;[Decentralized Slot Assignment](phases/phase2b_assignment.md) | Planned |
| 2c | &nbsp;&nbsp;[Multi-Dropout + Fault Catalog](phases/phase2c_fault_tolerance.md) | Planned |
| 2d | &nbsp;&nbsp;[Distributed Dissemination + Consensus](phases/phase2d_consensus_dissemination.md) | Planned |
| 3 | [Scale (5 → 20+ drones)](phases/phase3_scale.md) | Planned |
| 4 | [Expressive Shape Library](phases/phase4_shapes.md) | Planned |
| 5 | [Animated / Time-Varying Formations](phases/phase5_animated.md) | Planned |
| 6 | [Outdoor Disturbance DR](phases/phase6_disturbance_dr.md) | Planned |
| 7 | [Obstacle-Aware Formation Control](phases/phase7_obstacle_sim.md) | Planned |
| 8 | [Onboard Inference Profiling + Distillation](phases/phase8_onboard_distill.md) | Planned |
| 9 | [Multi-Platform Domain Randomization](phases/phase9_multiplatform_dr.md) | Planned |

### Hardware block (one variable per phase)

| Phase | Title | Status |
| :--- | :--- | :--- |
| 10 | [Single-Drone Hardware Bring-Up](phases/phase10_singledrone.md) | Planned |
| 11 | [Anchored Multi-Drone Formation](phases/phase11_anchored_formation.md) | Planned |
| 12 | [Anchor Removal + Decentralized Stack on Hardware](phases/phase12_decentralized_hw.md) | Planned |
| 13 | [Skybrush End-to-End](phases/phase13_skybrush_e2e.md) | Planned |

### The major milestone

| Phase | Title | Status |
| :--- | :--- | :--- |
| 14 | [First Drone Show (sub-phased)](phases/phase14_drone_show.md) | Planned |
| 14a | &nbsp;&nbsp;[Part 107 Certificate](phases/phase14a_part107.md) | Planned |
| 14b | &nbsp;&nbsp;[Single-Drone Outdoor Content](phases/phase14b_outdoor_solo_content.md) | Planned |
| 14c | &nbsp;&nbsp;[Multi-Drone Rehearsals](phases/phase14c_multidrone_rehearsals.md) | Planned |
| 14d | &nbsp;&nbsp;[First Paid Booking](phases/phase14d_first_paid_booking.md) | Planned |

### Post-milestone (vision-level only)

| Phase | Title | Status |
| :--- | :--- | :--- |
| 15 | [Outdoor + Extended Fault Tolerance Hardware](phases/phase15_outdoor_hw.md) | Vision |
| 16 | [Onboard Compute Hardware Integration](phases/phase16_onboard_hw.md) | Vision |
| 17 | [Obstacle-Aware Navigation Hardware](phases/phase17_obstacle_hw.md) | Vision |
| 18 | [Multi-Platform Hardware (stretch)](phases/phase18_multiplatform_hw.md) | Stretch |

## Status

- [changelog](status/changelog.md)
- [log](status/log.md)
