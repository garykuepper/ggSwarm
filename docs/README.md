# ggSwarm Documentation

This repository hosts two distinct programs.

## ggSwarm Capstone (v1, shipped April 2026)

A simulation-only multi-agent RL drone swarm — Isaac Lab + SKRL PPO with
GATv2 graph attention. Shipped as a CSULB capstone deliverable on
2026-04-24. **Frozen.** All capstone documentation lives under
[`capstone/`](capstone/) and the snapshot is tagged `v1.0.0-capstone`
on the `capstone` branch.

Start here: [`capstone/README.md`](capstone/README.md).

## ggSwarm Live (active development)

Real-hardware deployment program. Takes the v1 policy out of simulation,
onto PX4-based airframes, and delivers it as the adaptive-execution layer
underneath Skybrush drone-light-show choreography. Target architecture:
[`ggswarm_live/architecture.md`](ggswarm_live/architecture.md).

Start here: [`ggswarm_live/README.md`](ggswarm_live/README.md).
