# Phase 17: Obstacle-Aware Navigation Hardware (vision-level)

**Status:** Vision only. Detail to be fleshed out post-Phase 14.
Hardware spend gated by Phase 14 revenue and on Phase 16 onboard-compute
prerequisites.

**New capability:** real obstacle perception (stereo depth or lidar, VIO
for self-motion) composed with the Phase 7 sim-validated obstacle-aware
formation control. This is the "navigate through obstacles in unknown
environments" requirement from `vision.md` § 2 (FR-6).

## High-level scope

- Onboard perception hardware: stereo depth or solid-state lidar; VIO for
  self-motion estimation.
- Reactive obstacle avoidance composed with formation control on real
  airframes (Phase 7 closes the algorithm in sim; Phase 17 closes the
  hardware integration).
- Unknown environments (no prior map); reactive only.
- Reference implementations: Agilicious
  ([Foehn 2022](../references.md#foehn2022)) for onboard-compute agile
  quadrotor reference.

## Why deferred to post-show

Phase 7 (sim) closes the algorithmic composition of formation + obstacle
avoidance. Phase 16 closes the onboard-compute integration. Phase 17 adds
real perception sensors and the perception-latency-vs-control-loop
composition — substantial hardware cost (sensors, integration time) that
Phase 14 revenue funds.

## See Also

- [Phase 7 Obstacle-Aware Formation Control (sim)](phase7_obstacle_sim.md)
- [Phase 16 Onboard Compute Hardware](phase16_onboard_hw.md) — prerequisite
- [Phase 14 First Drone Show](phase14_drone_show.md)
- [Vision § Phase 17](../vision.md)
