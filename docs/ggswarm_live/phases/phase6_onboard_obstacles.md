# Phase 6: Onboard Compute + Obstacle-Aware Navigation

**Status:** Planned.

**New capability:** unknown-environment navigation.

## Scope

- Upgrade to Jetson Orin Nano-class airframes (ModalAI VOXL2, or custom
  Jetson-Orin-Nano carrier on Holybro frame)
- Onboard GATv2 inference (≥ 50 Hz target)
- Onboard perception: stereo depth or lidar, VIO for self-motion
- Reactive obstacle avoidance composed with formation control (CBFs treating
  obstacles as virtual agents, as explored in capstone Phase 4)
- Unknown environments (no prior map)
- Reference implementations to study: Agilicious
  ([Foehn 2022](../references.md#foehn2022)) is the closest open-source
  analogue for onboard-compute agile quadrotors with BEM-based aerodynamics.

## Backlog items folded into this phase

- **C1** Action-space CBF reactive avoidance ([backlog](../backlog.md#c1))
- **C2** Learned obstacle avoidance revisit ([backlog](../backlog.md#c2))
- **C3** Urban canyon scenario ([backlog](../backlog.md#c3))

## Keeps simple

May reduce scale temporarily (5–10 drones) due to hardware cost; single
platform only.

## Milestone artifact

Outdoor demo video of a swarm flying through a novel, unmapped obstacle
environment with all inference onboard. Repo release with the onboard
inference binary. Social: "No map. No offboard compute. The drones see
and plan themselves."

## Risk hot-spots

- Real-time GATv2 inference budget on Orin Nano → policy distillation, MLP fallback from GATv2 teacher
- CBF QP solve time onboard
- Perception failure modes (reflective surfaces, shadows, sparse visual features)
- Composing perception latency with formation control dynamics

## See Also

- [Vision § Phase 6](../vision.md)
