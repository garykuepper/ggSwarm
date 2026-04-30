# Phase 2: Sim-to-Real Baseline

**Status:** Planned.

**New capability:** policy runs on real hardware.

## Scope

- 3–5 Crazyflie 2.1 drones
- Indoor controlled environment
- Loco Positioning System (UWB with fixed anchors; remove anchors in Phase 3)
- Static geometric formations only (FR-1 subset)
- Domain randomization during training: mass, inertia, motor response,
  sensor latency, UWB noise, battery voltage sag, thrust-to-weight variance
- Offboard radio link from laptop for command injection (drones still execute onboard)

> **Fork in the road: Pegasus Simulator.** Pegasus
> ([Jacinto 2024](../references.md#jacinto2024)) is an Isaac Sim extension
> shipping the stack Phase 2 needs (multi-vehicle, PX4, ROS 2, magnetometer
> / GPS / barometer sensors). Building on it instead of rolling custom PX4
> integration could save months. Risk: single-thesis project, sustainability
> uncertain. **Evaluate before Phase 2 starts.**

## Architecture entry point

Phase 2 is where the [target architecture](../architecture.md) starts coming
online — specifically, PX4 offboard mode and the companion-computer setpoint
publisher. Skybrush waypoint ingestion and the RL overlay's "fallback to
raw waypoint" cascade should both be exercised on Crazyflie before the
hardware platform changes.

## Keeps simple

Anchors still in play, no peer ranging, no distributed assignment, no
letters / arbitrary shapes, no outdoor, no obstacles, no multi-dropout.

## Milestone artifact

First real-hardware formation-hold video. Repo release tagged
`v2.0.0-sim2real`. Social: "GATv2 trained entirely in sim, flying real
Crazyflies in formation for the first time."

## Risk hot-spots

- Sim-to-real gap doesn't close → extended domain randomization, real-to-sim calibration loop
- Battery sag breaks policy → voltage as explicit observation or randomized thrust scaling
- Motor saturation not captured in sim
- UWB latency differs from sim latency

## See Also

- [Vision § Phase 2](../vision.md)
- [Architecture](../architecture.md)
