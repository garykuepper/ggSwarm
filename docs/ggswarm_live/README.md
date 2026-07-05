# ggSwarm Live

Real-hardware research program. Takes the v1 GATv2/PPO formation-control
policy out of an idealized, centralized simulation and makes it work
decentralized, under real aerodynamics, and eventually on real drones.

The capstone (`v1.0.0-capstone`, frozen at [`../capstone/`](../capstone/))
proved the policy in sim with perfect state and central coordination.
ggSwarm Live removes those simplifications.

**Program shape:** two phases. Sim first (decentralization + downwash
physics), then hardware transfer as a goal list, not a detailed plan —
detail gets filled in once sim work makes clear what hardware needs.

Simplified 2026-07-03 from an earlier 18-phase plan; that plan and its
Skybrush/drone-show framing are preserved in [`archive/`](archive/) for
reference. **The drone-light-show revenue work is now a separate
project**, not part of ggSwarm — see [`vision.md`](vision.md) for why.

## Where to start

- [Vision](vision.md)
- [Phase 1: Sim](phases/phase1_sim.md) — decentralization + downwash, MAPPO groundwork already built
- [Decentralization detail plan](decentralization_plan.md) — staged plan for peer-to-peer localization (Phase 1 Goal A)
- [Phase 2: Hardware](phases/phase2_hardware.md) — goal list
- [Backlog](backlog.md) — loose, unscheduled ideas
- [References](references.md)
- [Archive](archive/) — the earlier 18-phase plan, kept for reference

## Phases

| Phase | Title | Status |
| :--- | :--- | :--- |
| 0 | [Capstone Baseline](phases/phase0_capstone_baseline.md) | Complete (v1.0.0-capstone) |
| 1 | [Sim: Decentralization + Downwash](phases/phase1_sim.md) | Shared-scene MAPPO groundwork complete; both goals open |
| 2 | [Hardware Transfer](phases/phase2_hardware.md) | Planned (goal list only) |

## Status

- [changelog](status/changelog.md)
- [log](status/log.md)
