# Phase 16: Onboard Compute Hardware Integration (vision-level)

**Status:** Vision only. Detail to be fleshed out post-Phase 14.
Hardware spend gated by Phase 14 revenue.

**New capability:** the policy runs entirely on-airframe, removing the
companion-computer dependency that Phases 10–14 retained. This is the
"runs fully onboard; no offboard compute in the control loop" requirement
from `vision.md` § 3 (NFR-3).

## High-level scope

- Upgrade to Jetson Orin Nano-class airframes (ModalAI VOXL2, or custom
  Jetson-Orin-Nano carrier on Holybro frame).
- On-airframe deployment of the Phase 8 distilled student policy
  (≥ 50 Hz inference target met in Phase 8 sim; Phase 16 validates
  on-hardware).
- Onboard ROS 2 host for the policy, gossip stack, and any perception
  hooks (Phase 17 adds perception itself).
- Power / thermal envelope characterized at flight conditions.

## Why deferred to post-show

Phase 8 (sim) closes the inference budget and distillation work. Phase 16
is the on-airframe validation — substantial hardware integration cost,
but the algorithmic risk is already paid down. Sequenced after Phase 14
because the airframe upgrades the carrier requires are funded by show
revenue.

## See Also

- [Phase 8 Onboard Inference Profiling + Distillation (sim)](phase8_onboard_distill.md)
- [Phase 17 Obstacle-Aware Hardware](phase17_obstacle_hw.md) — composes with onboard compute
- [Phase 14 First Drone Show](phase14_drone_show.md)
- [Vision § Phase 16](../vision.md)
