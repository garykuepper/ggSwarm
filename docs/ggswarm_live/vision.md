# ggSwarm v2: Research Program Requirements & Phase Plan

*Living document, v0.2. Post-capstone planning. Solo pace, no timeline.
Reverse-engineered from the vision endpoint, adjusted as the picture
sharpens. Snapshot of capstone-state-before-v2 lives at tag
`v1.0.0-capstone` and branch `capstone`.*

---

## 1. Vision Statement

A **general-purpose decentralized drone swarm** of 10–30 drones that forms
arbitrary shapes, letters, and animated formations on command, navigates
unknown outdoor environments with obstacles, operates fully GPS-denied with
no fixed infrastructure, tolerates hardware failures, and runs entirely on
onboard compute with a hardware-agnostic policy.

If you had to describe it in one line: *Tell any drone "form a triangle and
fly through that forest," and it happens, with no humans, no anchors, no GPS,
no central controller.*

---

## 2. Functional Requirements

| ID | Requirement | Notes |
| :--- | :--- | :--- |
| FR-1 | Form static geometric shapes (circle, polygon, line, grid) with parametric size | Baseline capability |
| FR-2 | Form alphanumeric glyphs (letters, numbers) | Requires glyph-to-points generator |
| FR-3 | Form arbitrary point clouds from uploaded waypoint files | Requires slot assignment over arbitrary target sets |
| FR-4 | Execute time-varying / animated formations (morphing, rotating, flocking) | Requires temporal reward structure during training |
| FR-5 | Accept commands via one-to-swarm interface; command propagates via gossip | Any single drone hearing the command suffices |
| FR-6 | Navigate through obstacles in unknown environments | Requires onboard perception + reactive avoidance |
| FR-7 | Maintain formation while translating through space (moving centroid) | Already demonstrated in capstone |
| FR-8 | Reorganize autonomously on drone loss | Re-run assignment, fill the gap |
| FR-9 | Operate with arbitrary swarm size within supported range | Policy is size-agnostic within training distribution |

## 3. Non-Functional Requirements

| ID | Requirement | Notes |
| :--- | :--- | :--- |
| NFR-1 | Fully GPS-denied; localization via drone-to-drone UWB peer ranging only | No fixed anchors, no mocap |
| NFR-2 | No fixed infrastructure (no ground station, no anchors, no mocap at runtime) | Development can use these; deployment cannot |
| NFR-3 | Runs fully onboard; no offboard compute in the control loop at runtime | Policy inference, perception, assignment all onboard |
| NFR-4 | Hardware-agnostic policy that works across multiple quad platforms without retraining | Domain randomization over dynamics parameters |
| NFR-5 | Peer-to-peer communication only (mesh radio, no internet, no cellular) | UWB for ranging, separate radio for data |
| NFR-6 | Fault-tolerant to: single dropout, multi-simultaneous dropout, comms failure, sensor degradation, operator disconnect, battery-graceful-degradation | Ranked priority per capstone proposal |
| NFR-7 | Scale: 10–30 drones in operational envelope | Training distribution should cover this range |

## 4. Performance Targets *(TODO: needs numeric nail-down)*

| Metric | Capstone baseline | Phase 1 target | Endpoint target |
| :--- | :--- | :--- | :--- |
| Formation error (steady state) | 0.038 m (sim) | TBD on real HW | TBD |
| Settling time (command → formed) | ~4 s (sim) | TBD | TBD |
| Dropout recovery time | ~1.0 s (sim) | TBD | TBD |
| Max translation speed in formation | TBD | TBD | TBD |
| Policy inference rate (onboard) | N/A (sim only) | N/A | ≥ 50 Hz |
| Mission duration | N/A | ≥ 5 min | ≥ 20 min |

## 5. Explicit Assumptions & Out-of-Scope

**Assumed (not solving):**

- Benign weather: no rain, no snow, wind < 5 m/s
- Visual-range operations only (no BVLOS regulatory work)
- Cooperative drones only (no adversarial scenarios, no anti-swarm)
- No human-drone interaction beyond operator command
- Each drone knows its own dynamic model family at deploy time

**Explicitly out-of-scope:**

- Heterogeneous swarms (mixed quad / fixed-wing / ground)
- Payload delivery or manipulation
- Formation flight with unmanaged non-swarm traffic
- Certification / airworthiness / regulatory compliance paperwork

---

## 6. Phase Breakdown

Reverse-engineered from the endpoint backward to where the capstone left off.
Each phase is framed as: what new capability it adds, what it deliberately
keeps simple, and what the milestone artifact is (typically a recorded
demo, a social-media post, or a repo release, not a paper).

### Phase 0: Capstone Baseline (complete)

- Isolated envs (1 drone per physics scene), perfect state, centrally precomputed slots
- GATv2 + shared-policy PPO, MINCO filter, CBF shield, dynamic slot recompute on dropout
- Simulation only
- **Status:** Shipped April 2026.

### Phase 1: Shared-Scene Multi-Drone Training (sim only)

**New capability:** real inter-drone aerodynamics (downwash, wake turbulence)
enters the training distribution.

**Scope:**

- 8 drones in one shared Isaac physics scene per env (replaces 8 isolated envs)
- Still perfect state, still centrally precomputed slots. *Isolate the aero variable.*
- Downwash modeling: analytic force model ported from gym-pybullet-drones
  [[Panerati 2021]](references.md#panerati2021), or learned residual on
  the relative-pose graph following Neural-Swarm2
  [[Shi 2022]](references.md#shi2022), or both as an ablation. Shared
  scene alone only gives contact/collision coupling; explicit downwash is
  a modeling layer you add on top.
- Retrain shared GATv2 policy; compare against Phase 0 checkpoint

**Keeps simple:** perfect GPS, no peer ranging, no assignment, no outdoor, no obstacles.

**Milestone artifact:** side-by-side comparison video (shared-scene vs
isolated-scene) + updated checkpoint in the repo + short write-up on
social media about what changed when aero entered the loop.

**Why first:** all downstream work inherits the policy that was trained here.
If shared-scene training breaks something, you need to know before you're
debugging it on a real drone.

---

### Phase 2: Decentralized + Fault-Tolerant Stack (sim only, sub-phased)

**New capability:** entire decentralized + fault-tolerant stack — peer
ranging, distributed slot assignment, multi-dropout fault tolerance,
gossip-based command + centroid consensus — validated in simulation
before any real hardware. See
[phase2_decentralized.md](phases/phase2_decentralized.md) for the parent
index and [consensus_mechanisms.md](consensus_mechanisms.md) for the
"why not blockchain / not Raft" rationale.

**Why before sim-to-real:** these subsystems are largely algorithmic
(multilateration, auction convergence, gossip versioning, average
consensus). Debugging them in Isaac Lab with full state observability is
dramatically cheaper than debugging them on real Crazyflies. Phase 3
then takes a fully decentralized fault-tolerant policy to hardware in
one disciplined step.

**Why sub-phased:** the original Phase 2 bundled five orthogonal
subsystems behind one milestone, making regressions un-attributable
during MAPPO training and preventing a clean go/no-go gate per
capability. The sub-phase split below isolates one variable per
sub-phase, each with its own milestone artifact and scorecard. The
scorecards double as Part 107.35 evidence rows feeding the failsafe
cascade in [architecture.md § 4](architecture.md#4-failsafe-architecture).

| Sub-phase | Title | One-line scope |
| :--- | :--- | :--- |
| 2a | [Decentralized Localization](phases/phase2a_localization.md) | Simulated UWB peer ranging + multilateration; SwarmRaft (paper) residual-test + recovery |
| 2b | [Decentralized Slot Assignment](phases/phase2b_assignment.md) | Backlog E1 (slot-pref logits) stepping stone, then Bertsekas auction over peer mesh |
| 2c | [Multi-Dropout + Fault Catalog](phases/phase2c_fault_tolerance.md) | Extend single-dropout alive-mask to N-simultaneous; documented fault catalog; rename code's `SwarmRaft` → `AliveMask` |
| 2d | [Distributed Dissemination + Consensus](phases/phase2d_consensus_dissemination.md) | Versioned gossip + CRDT command channel; average-consensus centroid |

**Keeps simple:** sim only — Phase 3 takes this to hardware. Static
shapes, 8-drone shared scene inherited from Phase 1.

**Whole-Phase-2 milestone artifact:** sim demo + write-up of the Phase 1c
shared-scene policy now running with peer-ranging-only localization,
decentralized auction-based slot assignment, multi-dropout fault
tolerance, and gossip-based command + centroid consensus — with all
simulated anchors removed. Repo checkpoint of the full decentralized
fault-tolerant policy. Social-media framing: "Decentralized swarm in
sim — no anchors, no central controller, fault-tolerant under
multi-dropout, command propagation under packet loss."

**Risk hot-spots:** peer-range localization accuracy may be insufficient
for tight formation error targets; auction convergence under realistic
comms loss; centroid drift under partition; sim noise models for UWB /
mesh under-representing real hardware (calibrate against Phase 3 logs
once available); multi-dropout exposing a policy that overfit to
single-dropout scenarios (curriculum across the fault catalog).

---

### Phase 3: Scale (5 → 20+ drones, sim only)

**New capability:** policy + decentralized stack scales to 20+ drones in
sim. Auction convergence retest at N ≥ 20. Downwash / density limits
documented. Bandwidth ceiling documented.

See [phase3_scale.md](phases/phase3_scale.md). Independent of Phases 4–9;
parallel-capable.

---

### Phase 4: Expressive Shape Library (sim only)

**New capability:** parametric shape generator (N-gon, glyphs via
glyph-to-points, arbitrary uploaded point clouds); curriculum over
shape-count and shape-type; held-out generalization measurement.

See [phase4_shapes.md](phases/phase4_shapes.md). Independent of Phases 3,
5–9; parallel-capable.

---

### Phase 5: Animated / Time-Varying Formations (sim only)

**New capability:** morphing between shapes, rotating formations,
temporal reward structure. Choreography primitive library
(morph / rotate / hold / translate) for Phase 14 consumption.

See [phase5_animated.md](phases/phase5_animated.md). Soft-depends on
Phase 4 (interesting morphs need interesting shapes); otherwise
parallel-capable.

---

### Phase 6: Outdoor Disturbance Domain Randomization (sim only)

**New capability:** policy trained against simulated outdoor disturbances
(wind 0–5 m/s + gust + thermal + outdoor-multipath sensor noise) before
any real outdoor flight. Phase 2c fault catalog re-run under disturbance.
Pulled forward from the old "Phase 5 outdoor" framing under the
*exhaust sim before hardware* principle.

See [phase6_disturbance_dr.md](phases/phase6_disturbance_dr.md).
Parallel-capable.

---

### Phase 7: Obstacle-Aware Formation Control (sim only)

**New capability:** policy composes formation control with reactive
obstacle avoidance in unknown simulated environments. CBF reactivated;
learned obstacle encoding ablation (column / occupancy / ray-cast /
graph-edges); urban-canyon scenario added. Pulled forward from the old
"Phase 6 onboard + obstacles" framing under the *exhaust sim before
hardware* principle.

See [phase7_obstacle_sim.md](phases/phase7_obstacle_sim.md).
Parallel-capable.

---

### Phase 8: Onboard Inference Profiling + Distillation (sim only)

**New capability:** policy profiled and distilled to fit ≥ 50 Hz onboard
inference on Jetson Orin Nano-class compute. ONNX export pipeline; CBF QP
solve-time budget confirmed; memory + power profiled. Pulled forward from
the old "Phase 6 onboard" framing under the *exhaust sim before hardware*
principle.

See [phase8_onboard_distill.md](phases/phase8_onboard_distill.md).
Parallel-capable.

---

### Phase 9: Multi-Platform Domain Randomization (sim only)

**New capability:** policy trained against extended airframe-family DR
(Crazyflie 2.1 → Holybro X500-class) so that one checkpoint can deploy
to multiple real platforms in Phase 18. Pulled forward from the old
"Phase 7 hardware-agnostic stretch" framing.

See [phase9_multiplatform_dr.md](phases/phase9_multiplatform_dr.md).
Parallel-capable.

---

### Phase 10: Single-Drone Hardware Bring-Up

**New capability:** one Crazyflie 2.1 flies under offboard control with
Skybrush waypoint ingestion; all four failsafe layers verified
individually on real hardware.

> **Fork in the road: Pegasus Simulator.** Pegasus
> [[Jacinto 2024]](references.md#jacinto2024) is an Isaac Sim extension
> that already ships the stack Phase 10 needs (multi-vehicle, PX4, ROS 2,
> magnetometer/GPS/barometer sensors). Building on it instead of rolling
> custom PX4 integration could save months. Risk: single-thesis project,
> sustainability uncertain. **Evaluate ahead of Phase 10.** See
> [references.md § 2](references.md#2-simulator-ecosystem).

See [phase10_singledrone.md](phases/phase10_singledrone.md). Strict
prerequisite for Phase 11.

---

### Phase 11: Anchored Multi-Drone Formation

**New capability:** Phase 1c shared-scene policy flies a formation on 3–5
real Crazyflies with LPS anchors providing absolute positioning as a
safety scaffold. *One variable at a time:* policy on hardware, anchors on,
centralized assignment retained.

See [phase11_anchored_formation.md](phases/phase11_anchored_formation.md).
Strict prerequisite for Phase 12.

---

### Phase 12: Anchor Removal + Decentralized Stack on Hardware

**New capability:** full Phase 2a/b/c/d decentralized + fault-tolerant
stack runs on real Crazyflies with no fixed positioning infrastructure.
Calibration loop back to Phase 2a (and Phases 6, 7) noise / disturbance /
obstacle models.

See [phase12_decentralized_hw.md](phases/phase12_decentralized_hw.md).
Strict prerequisite for Phase 13.

---

### Phase 13: Skybrush End-to-End Integration

**New capability:** Option B architecture runs end-to-end on real hardware:
Skybrush Studio CSV → companion → RL overlay → PX4. First show-shaped
artifact (60-second authored piece). Repo release `v2.0.0-sim2real`.

See [phase13_skybrush_e2e.md](phases/phase13_skybrush_e2e.md). Strict
prerequisite for Phase 14.

---

### Phase 14: First Drone Show — The Major Milestone

**New capability:** small-venue paid drone show performance. The program's
headline milestone. Sub-phased per the regulatory + revenue gates that
sequence the work:

| Sub-phase | Title | Gate |
| :--- | :--- | :--- |
| 14a | [Part 107 Certificate](phases/phase14a_part107.md) | Certificate granted |
| 14b | [Single-Drone Outdoor Content](phases/phase14b_outdoor_solo_content.md) | Outdoor portfolio published; § 107.35 waiver application filed |
| 14c | [Multi-Drone Rehearsals](phases/phase14c_multidrone_rehearsals.md) | Reliable multi-drone outdoor rehearsals post-waiver |
| 14d | [First Paid Booking](phases/phase14d_first_paid_booking.md) | A paid show flown to completion |

See [phase14_drone_show.md](phases/phase14_drone_show.md) for the parent
index. Repo release `v3.0.0-first-show` at completion.

---

### Phase 15: Outdoor + Extended Fault Tolerance Hardware (vision-level, post-show)

**New capability:** outdoor flight at scale on larger airframes
(Holybro X500 / ModalAI Seeker-class) — beyond the Crazyflie envelope of
Phases 10–14. Hardware spend gated by Phase 14 revenue.

See [phase15_outdoor_hw.md](phases/phase15_outdoor_hw.md). Detail
deferred to post-Phase 14.

---

### Phase 16: Onboard Compute Hardware Integration (vision-level, post-show)

**New capability:** policy runs entirely on-airframe (Jetson Orin
Nano-class), removing the companion-computer dependency that
Phases 10–14 retained. Closes the NFR-3 "fully onboard" requirement.

See [phase16_onboard_hw.md](phases/phase16_onboard_hw.md). Detail
deferred to post-Phase 14.

---

### Phase 17: Obstacle-Aware Navigation Hardware (vision-level, post-show)

**New capability:** real obstacle perception (stereo / lidar / VIO)
composed with the Phase 7 sim-validated obstacle-aware formation control.
Closes the FR-6 "navigate through obstacles in unknown environments"
requirement.

See [phase17_obstacle_hw.md](phases/phase17_obstacle_hw.md). Detail
deferred to post-Phase 14; depends on Phase 16.

---

### Phase 18: Multi-Platform Hardware (vision-level, stretch, post-show)

**New capability:** same policy checkpoint deployed to ≥ 2 different real
quad airframes without retraining. Closes the NFR-4 "hardware-agnostic
policy" requirement.

See [phase18_multiplatform_hw.md](phases/phase18_multiplatform_hw.md).
Detail deferred to post-Phase 14; stretch even then.

---

## 7. Phase Dependencies

```text
SIM BLOCK (exhaust algorithms first; phases 3–9 parallel-capable)
   Phase 0 (capstone, frozen)
        │
        ▼
   Phase 1 (shared-scene, 1a complete)
        │
        ▼
   Phase 2 (decentralized + fault-tolerant)  ┐
        ├──┬──┬──┐                           │ Phase 2 sub-phases
        ▼  ▼  ▼  ▼                           │ 2a → 2c (strict)
        2a 2b 2c 2d                          │ 2b, 2d parallel
        └──┴──┴──┘                           ┘
             │
             ├──┬──┬──┬──┬──┬──┐
             ▼  ▼  ▼  ▼  ▼  ▼  ▼
             3  4  5  6  7  8  9   (sim phases — parallel-capable
             │  │  │  │  │  │  │    given training-compute budget)
             └──┴──┴──┴──┴──┴──┘
                       │
                       ▼
HARDWARE BLOCK (one variable per phase, strict sequence)
                Phase 10 (single-drone bring-up)
                       │
                       ▼
                Phase 11 (anchored multi-drone)
                       │
                       ▼
                Phase 12 (anchors off + decentralized stack)  ─┐
                       │                                        │
                       │                       calibration loop │
                       │                       back to Phases   │
                       │                       2a / 6 / 7       │
                       ▼                                        │
                Phase 13 (Skybrush end-to-end)                 ─┘
                       │
                       ▼
THE MAJOR MILESTONE
                Phase 14 — First Drone Show
                  14a → 14b → 14c → 14d  (strict regulatory + revenue gates)
                       │
                       ▼
POST-MILESTONE (vision-level only; sequencing decided post-14)
                Phase 15 (outdoor hardware)
                Phase 16 (onboard compute hardware)
                Phase 17 (obstacle-aware hardware) — depends on 16
                Phase 18 (multi-platform hardware, stretch)
```

**Sequencing notes:**

- **Sim block (Phases 0 → 9):** Phases 3–9 can run in parallel given
  training-compute budget. Numbering reflects gate ordering, not forced
  sequence.
- **Phase 2 sub-phases (2a/2b/2c/2d):** 2a is strict prerequisite for 2c
  (need ranging before ranging-fault detection). 2b and 2d can run in
  parallel with 2a once the localization interface is defined.
- **Hardware block (Phases 10 → 13):** strict sequence. Each phase isolates
  one variable; the calibration loop from Phase 12 flows back into the sim
  block (Phase 2a primarily; Phases 6 and 7 secondarily) and may trigger
  a sim retrain before the next Phase 12 session or Phase 13 entry.
- **Phase 14 sub-phases:** strict sequence (Part 107 → solo content +
  waiver application → multi-drone rehearsals → first paid booking).
  Real-world prerequisites enforce the order.
- **Post-show phases (15–18):** sequencing decided post-Phase 14 based on
  funding from show revenue and on lessons from the show production
  envelope. Phase 17 depends on Phase 16; otherwise open.

---

## 8. Cross-Cutting Infrastructure

Work that serves multiple phases and needs to be built once:

- **Simulation environment.** Isaac Lab 2.3 scene templates, domain randomization rig, deterministic eval harness.
- **Training infrastructure.** Local RTX 3070 for debugging, cloud L4/A100 for sweeps; checkpoint versioning; experiment tracking (W&B or equivalent).
- **Flight stack (Phase 15+).** **PX4** autopilot on the flight controller;
  **MAVLink** as the wire protocol between autopilot and companion
  computer; **ROS 2** on the companion computer hosting policy,
  perception, and decentralized algorithms; **MAVROS** or the newer
  **uXRCE-DDS** bridge translating between MAVLink and ROS 2. ArduPilot
  is the main alternative but not chosen: PX4 has the dominant research
  ecosystem (Pegasus Simulator, Agilicious, most multi-vehicle academic
  work) and native Isaac Sim bridges. Phases 10–14 on Crazyflie bypass
  this stack entirely (CRTP over Crazyswarm2). See
  [references.md § 4](references.md#4-flight-stacks-and-middleware).
- **Hardware pipeline.** Drone build process, pre-flight checklist, spare parts stock, battery rotation, crash-recovery SOPs.
- **Test environments.** Indoor volume (phases 10–13), outdoor calm field (phase 14b+), outdoor obstacle course (phase 17).
- **Evaluation benchmarks.** A fixed set of repeatable scenarios run at the end of each phase to detect regressions.
- **Safety systems.** Emergency land triggers, geofences, kill switch, log-everything black-box.
- **Documentation.** Per-phase write-up, repo-release notes, social-media post drafts, open-source repo hygiene.

---

## 9. Risk Register (top items by phase)

| Phase | Risk | Likely mitigation |
| :--- | :--- | :--- |
| 1 | Shared-scene training too slow on 3070 | Debug locally, sweep on cloud |
| 2a | Calibrated UWB noise model under-represents Crazyflie multipath | Phase 12 logs feed back to recalibrate; 2a retrains |
| 2a | Multilateration ill-conditioned with too-few non-flagged peers | Fall back to short-window IMU dead reckoning (paper's INS fallback) |
| 2a | Residual-test false positives during aggressive maneuvers | Calibrate threshold against the maneuver envelope, not just hover |
| 2b | Auction does not converge under simulated packet loss | Stale-aware bidding; eventual-consistency convergence |
| 2b | E1 stepping stone destabilizes Phase 1c checkpoint | Curriculum learning rate; freeze policy core, train preference head separately |
| 2b | Auction does not scale to 20+ drones | Phase 3 retests with hierarchical clustering / consensus-based bundle algorithm |
| 2c | Multi-dropout exposes a policy that overfit to single-dropout | Curriculum training across the fault catalog, not just easy classes |
| 2c | `SwarmRaft` → `AliveMask` rename introduces incidental bugs | Standalone refactor PR ahead of substantive 2c work; full smoke + replay-gate suite before merge |
| 2d | Average consensus drifts under sustained partition | Partition-detect heuristic freezes anchor updates until reconvergence |
| 2d | Gossip bandwidth grows with N × command-history-depth | Cap history depth, prune by monotonic version, broadcast deltas after baseline confirmed |
| 2 | Sim noise models for UWB / mesh under-represent real hardware | Treat sim numbers as upper-bound; calibrate against Phase 12 logs |
| 3 | Auction does not scale to N=20+ | Hierarchical clustering or consensus-based bundle algorithm |
| 3 | Crazyflie downwash dominates at tight spacing | Enforce minimum spacing in parametric shape generator (Phase 4 reads this limit) |
| 4 | Held-out shapes underperform | Expand training distribution; honest envelope documentation if gap > 2× |
| 4 | Glyph-to-points uneven density breaks slot assignment | Uniform-density sampler; reject glyphs below minimum spacing |
| 5 | Morph triggers formation collapse | Tighter jerk bound; longer transition time; per-shape morph-rate limit |
| 5 | Temporal reward destabilizes existing policy | Start with small temporal window; ablation against instantaneous baseline |
| 6 | Wind model under-represents real outdoor turbulence | Treat as upper-bound until calibrated against Phase 11+ logs |
| 6 | Disturbance + multi-dropout interaction destabilizes policy | Curriculum schedule with both dimensions slowly increased |
| 7 | CBF QP solve time too high to compose with policy inference | Reduce obstacle count; differentiable barrier; defer QP to Phase 8 distillation |
| 7 | Curriculum collapses — policy ignores formation under obstacle pressure | Reward-shaping rebalance; explicit formation-coherence floor |
| 8 | Distillation gap too large | Stop distilling; ship GATv2 teacher with tight K cap; revisit post-show |
| 8 | Jetson Orin Nano dev kit unavailable | QEMU emulation or AWS Graviton proxy; document approximation |
| 9 | DR envelope too wide → policy regresses on Crazyflie | Curriculum from narrow to wide; floor on Crazyflie performance |
| 10 | Pegasus / Crazyswarm2 / PX4 stack-decision misjudgement | Decision recorded in Phase 10; phase rework cost only |
| 10 | Companion-computer setpoint scheduling jitter > 5% at 50 Hz | Real-time scheduler before Phase 11 |
| 11 | Sim-to-real gap larger than expected on first multi-drone session | Retrain Phase 1c with broader DR; retain LPS scaffolding through retrain cycles |
| 11 | Multi-Crazyflie radio interference at small N | Document the limit; factor into Phase 12 entry |
| 12 | Real peer-range outliers degrade auction faster than calibrated model predicted | Outlier rejection layer in multilateration; revisit Phase 2a noise tail |
| 12 | Sim-noise recalibration drives a large policy retrain | Warm-start from Phase 1c; retrain only adapter head if possible |
| 12 | Multi-dropout drill destroys hardware | Controlled telemetry power-down; spare-drone stock; one drill at a time |
| 13 | Skybrush CSV pipeline brittle (encoding / units / time alignment) | Scripted importer with unit tests; never author CSVs by hand |
| 13 | RL overlay introduces show-quality artifacts | Tighten offset clamp, or accept Layer-1-bypass-only mode for show ops |
| 14a | Fail Part 107 real test | Re-take after waiting period; high practice-test bar before scheduling |
| 14b | § 107.35 waiver timeline drags or grant likelihood lower than expected | Continue 14b indefinitely; engage UAS attorney if needed |
| 14b | Outdoor wind exceeds Phase 6 DR envelope | Calm conditions only initially; extend DR before pushing weather limits |
| 14c | Reliability bar repeatedly missed | Expand rehearsal window; revisit Phases 6 / 12 calibration before scheduling 14d |
| 14c | Outdoor sim-to-real gap re-opens at multi-drone scale | Multi-drone-specific DR retrain; possibly extend Phase 6 envelope |
| 14d | Crash on a paid show | Phase 14c reliability bar must be met; weather go/no-go enforced strictly |
| 14d | Weather no-go on show day | Reschedule clauses in booking contract; weather contingency communication |
| 15 | Outdoor + larger-airframe sim-to-real gap re-opens | Extend Phase 6 DR using Phase 14b/14c real outdoor logs |
| 16 | Onboard inference can't hit 50 Hz on real airframe | Phase 8 distillation; MLP fallback from GATv2 teacher |
| 17 | Perception fails in edge-case environments | Narrow the "unknown environment" definition; carry honest envelope |
| 18 | Platform transfer doesn't generalize | Frame as "family within randomization envelope," not true zero-shot |

Note on Phase 3 auction scale: simulator throughput is not the
bottleneck. Aerial Gym [[Kulkarni 2023]](references.md#kulkarni2023) has
already demonstrated thousands of multirotors in parallel on Isaac Gym.
The limiter is auction convergence time under realistic comms loss, not
how many drones the simulator can push.

Budget risks (cross-cutting):

- Hardware cost creep (5 → 30 drones + spares is $5k Crazyflie → $60k+ Jetson-class)
- Cloud training cost accumulation
- Outdoor test site access / liability / insurance

---

## 10. Open Questions (need to nail down before writing formal spec)

1. **Numeric performance targets** per phase (formation error, settling time, recovery time, translation speed).
2. **Communication stack specifics.** What radio for peer data link? (ESB, LoRa, ESP-NOW, custom?) Bandwidth budget?
3. **Mission duration requirements.** Affects battery choice and therefore airframe.
4. **Operator interface.** How is a command issued? Physical controller? CLI? Web UI? Natural language?
5. **Safety envelope.** Geofence rules, max altitude, emergency procedures, pre-flight checklist.
6. **Evaluation benchmark design.** What fixed scenarios define "phase passed"?
7. **Sharing strategy.** Which platforms (YouTube long-form, Twitter/X threads, Instagram reels, TikTok, personal blog, Hacker News) best match each milestone artifact?
8. **Open-source strategy.** Fully open? Core open, applications reserved?
9. **Hardware platform decisions** per phase (Phase 5 and 6 specifically: Crazyflie replacement and Jetson carrier).
10. **Realistic self-funded hardware budget ceiling.** Informs phase sequencing and parallelism.

---

---

## 11. References and Ecosystem Watch

All academic references, simulator ecosystem entries, and hardware
comparisons live in [`references.md`](references.md). Growing resource;
this plan links into it by author-year anchors
(e.g. `[Shi 2022]` → `references.md#shi2022`).

---

*Next revision should close out Section 4 (performance targets) and the
open questions in Section 10.*
