# Phase 5: Outdoor + Extended Fault Tolerance

**Status:** Planned. Hardware spend gated by Phase 4b revenue.

**New capability:** outdoor calm environment + broader fault model.

## Scope

- Move outdoors to a calm, open field
- **Likely hardware upgrade.** Crazyflies are practically indoor-only; need
  Holybro X500 / ModalAI Seeker-class or similar with real outdoor endurance.
- Wind domain randomization (0–5 m/s, gust models)
- Multi-simultaneous dropout, comms link failure, sensor degradation spikes,
  operator disconnect autonomy, battery-graceful-degradation
- GPS still disabled; continues peer-ranging approach

## Backlog items folded into this phase

- **B2** Non-flat terrain / 3D environments ([backlog](../backlog.md#b2))
- **B4** Heterogeneous agents — full ([backlog](../backlog.md#b4))

## Keeps simple

Still no obstacles. Policy inference may still be offboard (Phase 6 problem).

## Milestone artifact

Outdoor demo video of a swarm holding formation through wind gusts, with
one or two drones intentionally cut to demonstrate recovery. Social:
"Wind, GPS denied, one drone down. Swarm holds."

## Risk hot-spots

- Entire hardware stack changes, so the sim-to-real gap re-opens
- UWB range-finding outdoors has different multipath characteristics
- Battery scaling implications on swarm
- Recovering test hardware after crashes

## See Also

- [Vision § Phase 5](../vision.md)
- [Architecture § 4 Failsafe Architecture](../architecture.md)
