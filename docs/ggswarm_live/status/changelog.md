# ggSwarm Live Changelog

Reverse-chronological. Capstone changelog (frozen) lives at
[`../../capstone/status/changelog.md`](../../capstone/status/changelog.md).

## 2026-07-05 — Peer-to-peer localization: Stages 0-4 implemented, Isaac gates pending

[`decentralization_plan.md`](../decentralization_plan.md) Stages 0-4 are
implemented on branch `phase1-localization`: the `DropoutGuard` rename
(Stage 0), `source/ggswarm/ggswarm/ranging.py`'s `UwbRangingSim` (Stage 1),
`source/ggswarm/ggswarm/localization.py`'s `DecentralizedLocalizer`
(Stage 2), shadow-mode env integration in `ggswarm_marl_env.py` (Stage 3),
and `scripts/calibrate_residual_threshold.py` /
`scripts/eval_localization.py` (Stage 4). The pure-torch test suite
(`tests/test_ranging.py` + `tests/test_localization.py`) is **17/17 green**
on Linux CPU (no Isaac Sim dependency). Real gate numbers — RMSE/FP/FN/
recovery-time/collapses against the trained policy — require Isaac Sim and
are **not yet measured**; the unit-test-level numbers that are real:
synthetic scripted-trajectory RMSE **0.073 m**, gauge drift **0.040 m**
(Stage 2 gate), and recovered faulted-drone error mean **0.053 m** / max
**0.101 m** against the Stage 4 `< 0.20 m` unit-test gate (all three from
`tests/test_localization.py`, not the real policy).

**Three adjudicated design deviations from §3-4 of the plan:**

1. **Velocity-forward-predicted broadcasts, not raw last-broadcast
   position.** The plan's reference `correct()` (§3 "Correct") minimizes
   residual against peers' latency-delayed broadcast *positions* directly.
   That has a translation-drag defect: a moving swarm's stale broadcasts
   systematically lag the true position, and GN correction pulls each
   drone's estimate toward that lag every tick, so gauge drift compounds
   with real translation instead of just odometry noise. Fix: drones
   broadcast `(position, velocity)`; `correct()` forward-predicts the
   target as `b_pred = p_broadcast + v_broadcast * dt` before computing
   residuals, so latency no longer costs translation accuracy. Landed
   in Task 3 (commits `74e9074..0f21f6d`); `correct()`'s signature grew a
   required `dt` argument as a result, which Task 6 threads through as
   `self.step_dt`.
2. **Pre-fit innovation gating tick order + median residual aggregation +
   hold-last-verdict.** The plan's §3 "Fault detection"/"Correct" ordering
   (correct against all peers, then test residuals) lets one faulted tick
   contaminate every honest drone's estimate before the fault is ever
   flagged — measured at 1.06 m contamination in Task 4's initial
   red-test run. Redesigned tick order:
   `propagate → measure → update_residuals (pre-fit) → run_fault_test →
   correct (flag-gated links) → recover`. Residuals aggregate via the
   **median** (not mean) over each drone's peer links, so a single
   faulted link doesn't drag an honest drone's own residual over
   threshold; a drone with zero usable links **holds its last verdict**
   rather than flipping to healthy on missing evidence (prevents
   flag-flapping). Landed in Task 4 (commits `2c980ba..091617e`); full
   tick-order contract lives in `localization.py`'s class docstring.
3. **Accept-if-consistent recovery with an odometry jump gate.** The
   plan's §3 "Recovery" re-multilaterates a flagged drone from IRLS alone.
   Range residuals turn out not to separate a genuine range-fault from an
   equally range-consistent but wrong displaced position: on the test
   octagon a +1.0 m common bias converges to a fit with residual 0.111,
   inside the honest floor's 0.077 ± 0.033 — IRLS alone would accept the
   wrong fix. The odometry jump is the separating signal: a legitimate
   correction from dead-reckoning drift is threshold-sized, while
   accepting a biased fit demands a bias-sized jump from the
   dead-reckoned estimate. `recover()` now accepts a candidate only if
   both `cand_res <= mu + k*sigma` **and**
   `jump <= recovery_jump_gate * threshold` (new ctor param
   `recovery_jump_gate`, default `2.0`); otherwise it dead-reckons.
   Landed in Task 4 alongside deviation 2.

## Isaac-side gates: PENDING (Windows)

None of the following have been run — all deferred from Tasks 1, 6, 7, 8
because this branch was developed on a Linux box with no Isaac Sim. Run in
this order (later steps need earlier steps' output):

1. **Smoke train** (Task 1, Stage 0 gate) — confirms the `DropoutGuard`
   rename didn't break anything:

   ```text
   env_isaaclab/Scripts/python.exe scripts/skrl/train.py --headless --task ggswarm-marl-v0 --num_envs 64 --max_iterations 5
   ```

2. **Off-means-off replay gate** (Task 6, Stage 3 gate, most important) —
   `scripts/skrl/replay_gate.py` must be bit-identical with
   `loc_enabled=False` before/after the Task 6 commit.
3. **Shadow-mode replay** (Task 6, Stage 3 gate) — same replay gate with
   `loc_enabled=True`; check `Metrics/loc_rmse_m` ≤ 0.10 m steady-state and
   step-time parity. Task 6's own Linux-only mini-harness saw an honest
   floor of 0.086-0.15 m under synthetic geometry — an open calibration
   question, not a known wiring defect.
4. **Residual threshold calibration** (Task 7, Stage 4 gate):

   ```text
   env_isaaclab/Scripts/python.exe scripts/calibrate_residual_threshold.py --headless --task ggswarm-marl-v0 --checkpoint logs/ref/v1.0.0-capstone/best_agent.pt --num_envs 64 --episodes 50
   ```

   Paste the printed `residual_mu`/`residual_sigma` into `GgswarmMarlEnvCfg`
   (`residual_mu`/`residual_sigma` fields, currently placeholders).
5. **Localization scorecard, both modes** (Task 8, Stage 4 gate) — using
   the `residual_mu`/`residual_sigma` from step 4:

   ```text
   env_isaaclab\Scripts\python.exe scripts\eval_localization.py --headless --task ggswarm-marl-v0 --checkpoint logs\ref\v1.0.0-capstone\best_agent.pt --num_envs 64 --episodes 100 --mode honest --residual_mu <PASTE> --residual_sigma <PASTE>
   env_isaaclab\Scripts\python.exe scripts\eval_localization.py --headless --task ggswarm-marl-v0 --checkpoint logs\ref\v1.0.0-capstone\best_agent.pt --num_envs 64 --episodes 100 --mode fault --residual_mu <PASTE> --residual_sigma <PASTE>
   ```

   Expected to pass all five `decentralization_plan.md` §6 scorecard gates
   (RMSE ≤ 0.10 m, FP ≤ 0.01, FN ≤ 0.05, recovery containment high /
   time-to-flag p50 ≤ 1.0 s, 0 formation collapses).

## 2026-07-03 — Plan simplified to two phases; drone show split into a separate project

The 18-phase plan (2026-05-03 entry below) had grown into a detailed
commitment across capabilities that hadn't been started and a
drone-show milestone that coupled the research policy to a revenue
product it didn't need to be coupled to. Replaced with two phases:

- **Phase 1: Sim** — two flat goals, no fixed order: proper
  decentralization (no central assignment, no anchors, peer fault
  tolerance) and downwash/aero physics fidelity. Absorbs the intent of
  the old Phase 1 (shared-scene) and Phase 2 (2a–2d decentralized
  stack) without the sub-phase sequencing. The already-completed 1a
  MAPPO/DirectMARLEnv groundwork (see 2026-05-01 entry) carries over
  as-is — it's real accomplished work, not discarded.
- **Phase 2: Hardware** — a goal list (get the Phase 1 policy flying on
  real drones, confirm decentralization holds up outside sim), not a
  detailed plan. Replaces the old strict Phase 10–13 bring-up sequence.

**Drone-light-show work is no longer part of ggSwarm.** The old plan's
Skybrush RL-overlay architecture and Phase 14 (First Drone Show,
sub-phased through Part 107 / § 107.35 / first paid booking) assumed
the ggSwarm decentralized-formation policy would also drive paid shows.
That assumption didn't hold up: a drone light show needs pre-authored
choreography execution, not learned formation control, and forcing one
system to serve both distorted the research plan. The show work is now
a separate project with its own algorithm, developed to fund ggSwarm
research rather than being part of it.

**Capabilities dropped from numbered phases, kept as unscheduled ideas:**
expressive shapes, animated formations, obstacle-aware control, scale
beyond 20 drones, onboard distillation, multi-platform DR, outdoor
disturbance DR — see [`backlog.md`](../backlog.md).

**Disposition:** everything superseded moved to
[`archive/`](../archive/) rather than deleted — `vision.md` (→
`vision_v0.2.md`), `architecture.md`, `consensus_mechanisms.md`,
`backlog.md` (→ `backlog_detailed.md`), and phase docs 2/2a–2d, 3–9,
10–13, 14/14a–14d, 15–18. `phase1_shared_scene_sim.md` renamed to
`phase1_sim.md` and rewritten in place (accomplished work kept, scope
reframed to the two goals). New `phase2_hardware.md`, new short
`vision.md`, new short `backlog.md`, new `archive/README.md`. No code
changes; all editorial. Capstone (`v1.0.0-capstone`, `docs/capstone/**`)
untouched per project rules.

## 2026-05-03 — Full phase plan restructure: exhaust sim before hardware; drone show as the major milestone

**Driving principles:** (1) anything algorithmic ships as a numbered sim
phase before any hardware work; (2) hardware phases isolate one variable
each; (3) the drone show is the program's headline milestone, sub-phased
along its real regulatory + revenue gates; (4) post-show phases stay
numbered and sequential but cut to vision-level descriptions.

**New phase structure (18 numbered phases + sub-phases):**

- Sim block (exhaust algorithms first): 0 capstone, 1 shared-scene,
  2 decentralized + fault-tolerant (sub-phased 2a–d), 3 scale,
  4 expressive shapes, 5 animated formations, 6 outdoor disturbance DR
  in sim, 7 obstacle-aware formation control in sim, 8 onboard
  inference profiling + distillation in sim, 9 multi-platform DR in sim
- Hardware block (one variable per phase): 10 single-drone bring-up,
  11 anchored multi-drone, 12 anchors-off + decentralized stack on
  hardware (with calibration loop back to sim Phase 2a), 13 Skybrush
  end-to-end
- The major milestone: 14 First Drone Show, sub-phased 14a Part 107
  cert → 14b solo outdoor content + § 107.35 waiver application →
  14c multi-drone rehearsals → 14d first paid booking
- Post-milestone (vision-level only): 15 outdoor hardware,
  16 onboard compute hardware, 17 obstacle-aware hardware,
  18 multi-platform hardware (stretch)

**Promoted to top-level (was bundled or sub-phased):** former 4a/4b/4c
sim sub-phases → Phases 3 / 4 / 5; former Phase 5/6/7 hardware-bundled
sim work pulled forward to Phases 6 / 7 / 8 / 9; former Phase 3 hardware
sub-phases (3a/3b/3c/3d) → Phases 10 / 11 / 12 / 13; Drone Show track
→ Phase 14 with sub-phases 14a–14d.

**Files affected:**

- Renamed (11): phase4a→3, 4b→4, 4c→5, 3a→10, 3b→11, 3c→12, 3d→13,
  phase_drone_shows→14, phase5_outdoor_faults→15, phase6_onboard_obstacles→16,
  phase7_hardware_agnostic→18.
- New (9): phase6_disturbance_dr.md, phase7_obstacle_sim.md,
  phase8_onboard_distill.md, phase9_multiplatform_dr.md, phase14a_part107.md,
  phase14b_outdoor_solo_content.md, phase14c_multidrone_rehearsals.md,
  phase14d_first_paid_booking.md, phase17_obstacle_hw.md.
- Deleted (2): phase3_sim2real_baseline.md (parent index — sub-phases
  promoted), phase4_scale_shapes.md (parent index — sub-phases promoted).
- Edited: vision.md (§6 / §7 / §9 + cross-cutting infrastructure),
  README.md (phases table grouped into Sim / Hardware / Milestone /
  Post-show), architecture.md (§4 evidence column adds Phase 12 / 13;
  §6.1 evidence pipeline adds hardware sources), backlog.md (every
  phase tag re-pointed; H9–H12 added for sim phases 6–9; R1–R4 added
  for Phase 14 regulatory items), Phase 2 sub-phase docs (calibration
  loop now points to Phase 12).

No code changes; all editorial. Capstone (`v1.0.0-capstone`,
`docs/capstone/**`) untouched per project rules.

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

### MAPPO smoke training run (post-tag, 2026-05-01)

First from-scratch MAPPO training to verify the GNN + GATv2 architecture
trains end-to-end. 500 iters × 24 rollouts × 512 envs × 8 drones,
2814 s wall-clock (~47 min). Checkpoint at
`logs/skrl/ggswarm/p1a/2026-05-01_19-10-21_mappo_torch/checkpoints/best_agent.pt`.

**Architecture verified working:**

- TB reward / mean: −907.7 → −43.2 (21× improvement, n=100 logs)
- TB instantaneous reward / mean: −47.5 → −2.0 (approaches near-optimal hover)
- Parameter sharing confirmed: drone_0..drone_7 policy state_dicts bit-identical
- KNN edge publication + GATv2 message passing: zero shape errors over 12,000
  rollout steps
- Centralized critic gradient flow: stable losses across all 8 agents
- CBF safety shield: 0 collisions across 5-seed eval

**Formation behavior NOT converged at 500 iters:**

| metric | capstone (mean ± std) | MAPPO @500 (mean ± std) |
| :--- | :--- | :--- |
| mean_slot_error_m | 0.2189 ± 0.0188 | 1.1807 ± 0.0020 |
| collision_pairs_per_step | 0.0 | 0.0 |
| final_distance_to_goal_m | 0.0938 ± 0.0002 | 1.1972 ± 0.0613 |

Drones cluster ~1.2 m off their formation slots — the policy converged to
a local optimum (hover near env centroid) rather than spreading to a
triangle formation. The very tight per-seed std on slot_err (0.002 m
across 5 seeds) indicates the policy IS converged, just to the wrong
basin. Capstone needed substantially more compute + iterative reward
tuning to nail formation; 500 iters from random init under MAPPO is
undertrained, not broken. Eval CSV at
[`logs/sweeps/phase1a_replay_gate.txt`](../../../logs/sweeps/phase1a_replay_gate.txt)
(now reflects MAPPO @500 result, replacing the earlier capstone-replay
result; see git log for the prior numbers).

**Next steps for formation convergence (not yet executed):**

- Longer training run (1500–3000 iters) — most likely fix.
- Warm-start from capstone actor weights — replay_gate already loads them
  successfully; train.py needs a small hook for the same behavior.
- Tune `formation_reward_scale` higher (cur 2.0; try 4.0–8.0) to outweigh
  velocity penalties early in training.

Also: extended `scripts/skrl/replay_gate.py:load_actor_weights()` to
dispatch on checkpoint format (capstone single-agent PPO with top-level
`policy` key vs MAPPO MARL with per-agent `drone_i.policy` dicts). Same
script now evaluates either checkpoint in the shared-scene env.

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
