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

Real-hardware research program. Takes the v1 policy out of an
idealized, centralized simulation and makes it work decentralized,
under real aerodynamics, and eventually on real drones. Two phases:
sim (decentralization + downwash), then hardware transfer as a goal
list. The drone-light-show revenue work is a separate project, not
part of ggSwarm.

Start here: [`ggswarm_live/README.md`](ggswarm_live/README.md).
