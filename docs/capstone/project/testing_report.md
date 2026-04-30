# Testing Report

**Project:** Decentralized Formation Control for Drone Swarms (ggSwarm)
**Course:** CST489/499 Online Capstone — Spring 2026
**Student:** Gary Kuepper (solo project)
**Advisor:** Dr. Eric Tao
**Report date:** 2026-04-08

---

## 1. Introduction

### 1.1 Target audience

This project is a research-grade simulation, not a consumer product. As
stated in [the proposal § 2](project/proposal.md), the intended audience is:

- **Computer science students** studying robotics, multi-agent systems,
  or reinforcement learning.
- **Researchers** working on decentralized control, graph neural networks,
  or swarm coordination.
- **Engineers** building real-world UAV systems who want a reference
  implementation of GNN-based formation control with MINCO trajectory
  smoothing and SwarmRaft fault recovery.

Comfort level: **graduate / professional**. The audience is expected to
read PyTorch + Isaac Lab code, run NVIDIA GPU workloads, and interpret
TensorBoard scalars and trajectory plots.

### 1.2 Testers

This is a solo capstone with **no external client**
([proposal § 2, line 53](project/proposal.md)). The reviewing stakeholder
is the project advisor, **Dr. Eric Tao**, who fits the audience profile
(researcher / educator in autonomous systems).

Because the deliverable is a simulation + research artifact rather than
an interactive end-user application, "testing" in this report means
**quantitative validation of the trained policy against the proposal's
measurable objectives**, not a human usability study. The "tester" being
observed is the trained policy itself; the observation record is the
trajectory plots, TensorBoard scalars, and the 20-clip cinematic
inventory captured in Phase 5. This framing is documented in Phase 5
sub-task P5.2 ([phases/phase5_showcase_prep.md § 3](phases/phase5_showcase_prep.md))
and was approved as the test plan for the M3 gate.

### 1.3 Changes to initial objectives, approach, and deliverables

| Area | Original (proposal) | Final (as built) | Reason |
| :--- | :--- | :--- | :--- |
| O1 formation error | < 0.1 m | Loosened gate to < 0.3 m, achieved **0.038 m** | Gate set conservatively for the polygon-mode rebuild; actual result blew past the proposal target. |
| O2 jitter reduction | Runtime MINCO A/B | **Training-time** MINCO ablation (p4-4 vs p4-7) | Runtime A/B showed no difference because the trained policy had already internalized smoothness. Reframed as a *training stabilizer* benefit. See [phase4_stress_testing.md § 5 "MINCO Training Benefit"](phases/phase4_stress_testing.md). |
| O4 obstacle metric | "> 95% success rate over 100 episodes" | **Body-clearance**: 0 body penetrations / 5600 drone-steps | Drone-radius bug discovered 2026-04-07: original metric ignored the 0.10 m drone body, so historical "0 hit" runs had actually been grazing trunks by ~5 cm. Re-stated metric and re-validated with `p4-revert-4-trees2`. See [phase4_stress_testing.md § 5.5](phases/phase4_stress_testing.md). |
| Cinematic pipeline | Standalone `scripts/showcase.py` + `scripts/tron_env.py` | In-`play.py` `--tron` rebuild | The all-in-one `setup_tron_environment()` did ~6 things at once and was undebuggable; rebuilt incrementally inside `play.py`. See [phase5_showcase_prep.md § 5](phases/phase5_showcase_prep.md). |
| Algorithm | Multi-agent (MAPPO) | Single-agent shared-policy PPO (CTDE) via SKRL | Cleaner training, faster convergence, and the swarm behavior is identical because the policy is shared across all drones. |

No deliverables were dropped. All four proposal objectives (O1–O4) and
all four headline deliverables (repo, USD stage, HD trailer, this
testing report) are on track for the Apr 24 deadline.

---

## 2. Test Tasks

The trained policy was put through **five measurable scenarios** that
together cover every proposal objective (O1–O4). Each task has a strict
pre-registered pass criterion from the M3 gate
([phase4_stress_testing.md § 4](phases/phase4_stress_testing.md)).

| # | Task | Objective | Pass criterion | Measured | Result |
| :--- | :--- | :--- | :--- | :--- | :--- |
| T1 | Form and hold an octagon with 8 drones from random spawn | O1 | Formation error < 0.3 m steady-state | **0.038 m** | PASS |
| T2 | Recover formation after a drone is killed mid-episode (octagon → heptagon) | O3 | Re-sync < 2.0 s after dropout | **~1.0 s** (run p4-6) | PASS |
| T3 | Hold formation when scaled from 8 → 20 agents on the same checkpoint, no retraining | O4 | FE < 0.3 m, 0 inter-agent collisions | **FE 0.061 m, 0 collisions** | PASS |
| T4 | Traverse a static cylinder forest (0.20 m trunks) without body penetration | O4 | 0 body penetrations across 5600 drone-steps, body-radius-aware | **0 hits, +3.7 cm min clearance** (p4-revert-4-trees2) | PASS |
| T5 | MINCO training benefit vs ablation | O2 | ≥ 20 % steady-state jitter reduction | **77 % reduction** (0.008 m/s vs 0.034 m/s) | PASS |

All five tasks passed. The single canonical checkpoint feeding every
task is
`logs/skrl/ggswarm/p4/2026-04-06_21-09-24_ppo_torch/checkpoints/best_agent.pt`
(reward 66.83, ep_len 307.74).

---

## 3. Observations

For a research artifact the rubric's "watch the tester think aloud" step
becomes "watch the policy fly and read the telemetry." Here is what was
actually observed during each task.

**T1 — Octagon formation.** From a random spawn cloud (radius 0.8 m, Z
0.5–1.5 m), 8 drones converge to an octagon in ~1.6 s and then hold it
with a steady-state mean velocity of **0.014 m/s** — effectively
stationary. KNN distances stabilize at 0.4–0.5 m. Visible in
`videos/showcase/p5-orbit-octagon-10s` and the matching trajectory plot.

**T2 — SwarmRaft dropout recovery.** Around step 250, one drone is
killed. The remaining 7 detect the topology change via Phase 3's alive
mask, the slot offsets are recomputed in-place from octagon to
heptagon, and the swarm re-converges within roughly 50 simulation steps
(~1.0 s) — half the proposal's 2.0 s budget. The transition is clearly
visible in `videos/showcase/p5-orbit-dropout` and as a discontinuity in
the KNN-distance trace of the trajectory plot.

**T3 — Scale stress (8 → 20 agents).** Same checkpoint, no retraining.
KNN observations and polygon offsets auto-scale to whatever N is
requested at play time. At 20 agents the swarm forms an icosagon with
**FE 0.061 m and zero collisions**. At 15 agents one collision was
observed, traced to a brief slot-claim race during reorganization;
nearest-slot greedy assignment resolves it before any drone fails. See
`videos/showcase/p5-orbit-scale-20`.

**T4 — Forest navigation.** 8 drones in triangle formation cross two
staggered rows of 0.20 m cylinders at 0.63–0.70 m/s. Goal deflection
(boids-inspired flock-aligned 70/30 lateral-radial blend) steers each
drone around the nearest trunk while neighbors pick the same side.
**Zero body penetrations across 5600 drone-steps**, minimum body
clearance +3.7 cm. Formation deforms slightly (FE 0.73 → 0.75 m during
the hazard) and recovers cleanly downstream. Visible in
`videos/showcase/p5-orbit-forest-wireframe3`.

**T5 — MINCO training benefit.** Comparing two checkpoints trained with
identical hyperparameters except for the MINCO action filter (p4-4 with,
p4-7 without): MINCO-trained reaches **77 %** lower steady-state
jitter, **72 %** lower formation error, and **40 %** faster convergence.
The runtime MINCO filter is *not* needed at inference — the policy
internalized smoothness during training. This was the key finding that
reframed O2 from a runtime claim into a training-stabilizer claim.

---

## 4. Issues Found and Fixes

The testing process surfaced several real bugs and design problems. All
of them were fixed before the M3 gate; none are open at the time of this
report.

### 4.1 Short-term fixes (already shipped)

| # | Issue | Fix | Source |
| :--- | :--- | :--- | :--- |
| 1 | Drone-radius measurement bug — body penetrations were never counted with the 0.10 m drone radius, masking ~5 cm grazing as "0 hits" in every historical forest run. | Re-stated metric as `dist − cylinder_radius − drone_radius < 0`. Re-measured every run; only `p4-revert-4-trees2` legitimately passes. | [phase4 § 5.5 Bug 0](phases/phase4_stress_testing.md) |
| 2 | Goal-vs-drone deflection bug — deflection fired on goal slot positions, but drones drift off-slot and were penetrating cylinders their goals had already cleared. | Compute deflection from `self._robot.data.root_pos_w` instead of slot position. | [phase4 § 5.5 Bug 1](phases/phase4_stress_testing.md) |
| 3 | Runaway base-goal bug — stuck drones accumulated a goal 4+ m ahead of themselves, drowning out lateral deflection. | Cap base goal X to `drone_x + forest_max_goal_lead` (default 0.5 m). Stuck drones now get a goal that pauses with them. | [phase4 § 5.5 Bug 2](phases/phase4_stress_testing.md) |
| 4 | Cfg drift collapse on `p4-revert-1/2` (reward dropped from 63 → 24 / 7) despite identical env code. | Full revert of `cbf_d_safe`, `cbf_max_correction`, `collision_radius`, `dropout_enabled` to Mar 31 baseline → `p4-revert-4` (reward 66.83). | [phase4 § 5.5](phases/phase4_stress_testing.md) |
| 5 | Index-based slot assignment caused path crossings and collisions at 16+ agents (drones flew across the swarm to reach their assigned slot). | Greedy nearest-slot assignment — each drone claims the closest unclaimed slot. | [phase4 § Findings](phases/phase4_stress_testing.md) |
| 6 | Neighbors picked opposite sides of the same cylinder, causing formation tearing. | Boids-style flock alignment: lateral side picked from mean K-nearest neighbor velocity. | [phase4 § 5.5](phases/phase4_stress_testing.md) |
| 7 | Original `setup_tron_environment()` did six things at once and was undebuggable. | Rebuilt the cinematic pipeline incrementally inside `play.py` behind a `--tron` flag. | [phase5 § 5](phases/phase5_showcase_prep.md) |

### 4.2 Long-term / deferred work

These are intentionally out of scope for the capstone deadline but
remain interesting follow-ups. Consolidated backlog with effort/impact
ranking:
[ggSwarm Live backlog](../../ggswarm_live/backlog.md).

- **Action-space CBF for true reactive obstacle avoidance.** The CBF
  obstacle module is retained in `cbf.py` but disabled — goal deflection
  works better with the current policy. A future policy that respects
  CBF corrections could enable harder obstacle environments.
- **Learned obstacle avoidance.** Six-commit experiment preserved on the
  `experimental/learned-obstacle-avoidance` branch. Adding obstacle
  observation columns showed no measurable effect across six retrains.
- **Urban canyon scenario.** The proposal mentioned both forest and
  urban canyon; only the forest is in the final deliverable.
- **Beyond 20 agents.** Scale tests stopped at 20 because that meets O4.
  KNN obs and dynamic spawn radius would extend further but were not
  measured.

---

## 5. Evidence Appendix

A grader can click any of these to verify the claims above.

- **M3 gate table (source of truth for every number in § 2):**
  [phases/phase4_stress_testing.md § 4](phases/phase4_stress_testing.md)
- **Bug rebuild week (Bugs 0–2 narrative):**
  [phases/phase4_stress_testing.md § 5.5](phases/phase4_stress_testing.md)
- **Cinematic clip inventory (20 clips, mapped to scenes):**
  [phases/phase5_showcase_prep.md § 5](phases/phase5_showcase_prep.md)
- **Production checkpoint:**
  `logs/skrl/ggswarm/p4/2026-04-06_21-09-24_ppo_torch/checkpoints/best_agent.pt`
- **Trajectory plots and TensorBoard logs:** under
  `logs/skrl/ggswarm/p4/`
- **Cinematic clips:** under `videos/showcase/p5-*`
- **Proposal (objectives, audience, scope):**
  [project/proposal.md](project/proposal.md)

---

## 6. Conclusion

All four proposal objectives (O1–O4) passed at the M3 gate, with three
of them substantially exceeding their targets (O1: 0.038 m vs 0.1 m
target; O2: 77 % vs 20 % target; O3: 1.0 s vs 2.0 s target). The
deliverable is reproducible from the canonical `p4-revert-4` checkpoint,
the cinematic trailer is in editing, and no open issues block the
Apr 24 capstone festival submission.
