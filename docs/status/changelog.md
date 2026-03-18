# Changelog: ggSwarm Development

This document tracks major technical changes and milestone completions for each project phase.

## Phase 1: Foundation (Weeks 5-6)

- [2026-03-16] Initialized project repository and structure.
- [2026-03-16] Codified project rules and architecture documentation.
- [2026-03-16] Established status reporting framework.
- [2026-03-16] Implemented `GGSwarmMarlEnv` (DirectMARLEnv) with multi-agent Crazyflie spawning, environment cloning, and Articulation initialization.
- [2026-03-16] Implemented `GGSwarmMarlEnvCfg` with dynamic agent/space registration via `__post_init__()`.
- [2026-03-16] Built 12-dim observation space (body-frame velocities, gravity, relative goal position).
- [2026-03-16] Built 4-dim action space (thrust + 3-axis moments) mapped to propeller forces/torques.
- [2026-03-16] Implemented distance-based adjacency matrix (L2 graph connectivity, threshold 2.0m).
- [2026-03-16] Implemented reward function (Gaussian position reward, velocity penalties, alive bonus).
- [2026-03-16] Implemented reset logic with random spawn positions and yaw.
- [2026-03-16] Created `phase1_demo.py` for visual verification and adjacency matrix proof.
- [2026-03-16] Registered environment as `Template-GGSwarm-Marl-Direct-v0`.
- [2026-03-16] Set up SKRL MAPPO training pipeline (`train.py`, `play.py`, `skrl_mappo_cfg.yaml`).
- [2026-03-16] Renamed env files to `drone_swarm_env.py` / `drone_swarm_env_cfg.py` for clarity.
- [2026-03-16] Finalized comprehensive Phase 1 technical documentation.

## Phase 2: Brain Development (Weeks 7-8)

- [2026-03-18] Parameterized magic numbers in `drone_swarm_env.py` and `drone_swarm_env_cfg.py` (Rule 12).
- [2026-03-18] Implemented curriculum reward shaping with alpha-scaled formation and cohesion rewards.
- [2026-03-18] Added `rew_scale_formation`, `rew_scale_cohesion`, and `rew_scale_separation` to config (Rule 9).
- [2026-03-18] Created `skrl_gnn_policy.py` bridging PyTorch Geometric GATv2 with SKRL Gaussian models.
- [2026-03-18] Integrated `--gnn` flag in `train.py` for optional GNN policy activation.
- [2026-03-18] Optimized MAPPO hyperparameters (`rollouts: 32`, `mini_batches: 4`, `lr: 1e-4`).
