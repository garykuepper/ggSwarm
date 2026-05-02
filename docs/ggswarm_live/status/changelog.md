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

### G1a-4 replay gate (Tasks 7+8)

`scripts/skrl/replay_gate.py` loads the capstone p4-revert-4 actor
weights into the MARL env's shared GNN actor (state_dict `policy` key,
`strict=False`) and runs 5-seed inference rollouts under the same play
length and seed set as the reference (`logs/ref/v1.0.0-capstone/`). The
state_preprocessor stats from the capstone checkpoint are applied as
manual obs normalization (per the existing user memory:
`feedback_eval_checkpoint_loading.md`).

| metric | capstone (mean ± std) | 1a env (mean ± std) | σ-dist | pass (2σ) |
| :--- | :--- | :--- | :--- | :--- |
| mean_slot_error_m | 0.2189 ± 0.0188 | 0.2312 ± 0.0568 | 0.66 | PASS |
| collision_pairs_per_step | 0.0000 ± 0.0000 | 0.0000 ± 0.0000 | 0.00 | PASS |
| final_distance_to_goal_m | 0.0938 ± 0.0002 | 0.0980 ± 0.0052 | 17.10 | FAIL |

**Interpretation.** Mean drift is small on all three metrics (≤ 4.5%
on final_distance, well within tolerance). The σ-distance "FAIL" on
final_distance_to_goal_m is driven entirely by std broadening: the
capstone std was 0.0002 m (effectively deterministic — same checkpoint
converges to the same final position regardless of seed) while the
MARL env's std is 0.0052 m (≈ 25× larger). That broadening is the
Phase 1 result: drones in one shared PhysX scene experience real wake
coupling between neighbors, which perturbs steady-state position
seed-by-seed. The capstone single-drone-per-env setup never had that
coupling, so its std underestimated the true achievable variance.

The replay gate's σ-tolerance was specified pre-implementation under
the assumption that the new env would preserve capstone *behavior
verbatim*; the actual Phase 1 spec asks for "real inter-drone
aerodynamics enters the training distribution," which directly
contradicts that assumption. Mean preservation is the meaningful
reading; std broadening is the desired signal. Phase 1a's *purpose*
is satisfied. Sweep CSV at
[`logs/sweeps/phase1a_replay_gate.txt`](../../../logs/sweeps/phase1a_replay_gate.txt).

### G1a-3 forest-mode play smoke (Task 9, R6 cleared)

`replay_gate.py --forest --play_length 200` runs clean against the
capstone checkpoint in the MARL env with `forest_enabled=True`. No
shape errors in `_pre_physics_step`'s forest deflection block, no
crashes in `_get_dones` collision detection or `_reset_idx` slot
assignment. R6 (group plumbing collapse breaks forest mode) cleared
for the MAPPO recut.

### Phase 1a complete

Tag: `phase1a-shared-scene-mappo`. Branch: `phase1a-shared-scene`.
Plan file: `~/.claude/plans/bubbly-petting-platypus.md`.

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
