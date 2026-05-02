# ggSwarm Live Changelog

Reverse-chronological. Capstone changelog (frozen) lives at
[`../../capstone/status/changelog.md`](../../capstone/status/changelog.md).

## 2026-05-01 — Phase 1a re-cut as MAPPO + DirectMARLEnv

The original Phase 1a plan (in-place refactor of `GgswarmEnv` with a
`_FusedRobot` facade over A=8 sibling Articulation objects) was abandoned.
Two framework-level facts invalidated it: (1) Isaac Lab's scene cloner
does not support multiple sibling Articulations per env (one Articulation
per scene-key, replicated across all clones), and (2) `DirectRLEnv`
hardcodes `self.num_envs == scene.cfg.num_envs` as the gym vec dimension,
so "8 drones in one shared PhysX scene per env" conflicts with capstone's
per-drone-as-agent semantic. Smoke confirmed: SKRL allocates buffers at
`[num_envs, *]` and rejects `[N_drones, *]` per-drone obs.

Pivoted to `DirectMARLEnv` + SKRL `MAPPO`: physical env count =
`scene.cfg.num_envs`, A=8 drones per env in one PhysX scene, per-agent
dict obs/actions, shared GNN actor + shared centralized critic
(parameter sharing across all 8 drone-agents). Probe at
`scripts/probe/multi_drone_layout.py` validated the regex Articulation
spawn pattern produces env-major flat `[num_envs * A, *]` tensors.

### Tasks completed

- **Task 0** — Capstone reference rollouts captured. 5 seeds × 500 steps
  against `p4-revert-4` checkpoint, stored at `logs/ref/v1.0.0-capstone/`.
  Reference metric distributions: `mean_slot_error_m=0.219±0.019`,
  `collision_pairs_per_step=0.0`, `final_distance_to_goal_m=0.094±0.0002`.
- **Task 1** — `phase1a-shared-scene` branch created. Pre-refactor smoke
  passed (64 envs × 5 iters, 16.1s).
- **Task 2** — DirectMARLEnv skeleton (`GgswarmMarlEnv`) registered as
  `ggswarm-marl-v0` alongside the single-agent `ggswarm-v0`. Manual
  per-drone USD spawn + leaf-regex Articulation per the probe finding.
  Skeleton smoke passed at 2 envs × 1 iter (2.4s).
- **Task 3 / G1a-1** — Full env logic ported: formation/cloud reward,
  MINCO smoothing, CBF safety shield, SwarmRaft dropout, KNN obs
  expansion, forest deflection, collision detection, full `_reset_idx`
  with circular spawn (B3 fix folded in). Per-step allocation ban
  honored. Smoke passed at 16 envs × 5 iters (13.8s).
- **Task 4** — `GgswarmCentralizedValue` critic added. 3-layer MLP,
  144-dim input (8 drones × 18-dim obs), scalar output. All A agents
  share the same value model instance for parameter sharing.
- **Task 5** — MAPPO branch wired into `scripts/skrl/train.py`. Smoke
  passed at 16 envs × 5 iters MAPPO (29.1s).

### G1a-2 throughput sweep (Task 6)

Local 3070, MAPPO + GNN actor + centralized critic, A=8 drones per env,
5 iterations × 24 rollouts = 120 sim steps per row.

| num_envs | wall_clock_s | env-steps/s | status |
| :--- | :--- | :--- | :--- |
| 16 | 29.15 | 65.9 | ok |
| 32 | 30.95 | 124.1 | ok |
| 64 | 32.09 | 239.3 | ok |
| 128 | 34.10 | 450.4 | ok |
| 256 | 36.59 | 839.6 | ok |
| 512 | 40.58 | 1514.0 | ok |

Throughput scales near-linearly through 512 envs — 3070 has not
bottlenecked. Chosen knee: `num_envs=512` (matches capstone's 4096
total drones with A=8 per env). Cfg default already set to 512 in
`GgswarmMarlEnvCfg.scene.num_envs`. Sweep CSV at
[`logs/sweeps/phase1a_throughput.txt`](../../../logs/sweeps/phase1a_throughput.txt).

## 2026-04-30 — Program kickoff

- Reorganized `docs/` into `docs/capstone/` (frozen) and
  `docs/ggswarm_live/` (active).
- Promoted `ggswarm_v2_plan.md` → `ggswarm_live/vision.md`.
- Promoted `ggswarm_architecture.md` → `ggswarm_live/architecture.md`
  (target architecture; phased build-out).
- Deleted `phase7_post_capstone.md`; deferrals folded into
  [`backlog.md`](../backlog.md) and mapped onto Phase 1–6 stubs.
- Renumbered phase plan: Phase 0 = capstone baseline (complete);
  Phases 1–7 = ggSwarm Live.
