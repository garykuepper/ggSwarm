# ggSwarm v2: Research Program Requirements & Phase Plan

*Living document, v0.1. Reverse-engineered from the vision endpoint.*

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
keeps simple, and what the deliverable looks like.

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
- Retrain shared GATv2 policy; compare against Phase 0 checkpoint

**Keeps simple:** perfect GPS, no peer ranging, no assignment, no outdoor, no obstacles.

**Deliverable:** updated checkpoint, ablation analysis (shared-scene vs
isolated-scene formation quality), workshop-paper material or supplementary
to Paper 1.

**Why first:** all downstream work inherits the policy that was trained here.
If shared-scene training breaks something, you need to know before you're
debugging it on a real drone.

---

### Phase 2: Sim-to-Real Baseline *(Paper 1)*

**New capability:** policy runs on real hardware.

**Scope:**

- 3–5 Crazyflie 2.1 drones
- Indoor controlled environment
- Loco Positioning System (**UWB with fixed anchors**; keep infrastructure for now, remove it in Phase 3)
- Static geometric formations only (FR-1 subset)
- Domain randomization during training: mass, inertia, motor response, sensor latency, UWB noise, battery voltage sag, thrust-to-weight variance
- Offboard radio link from laptop for command injection (drones still execute onboard)

**Keeps simple:** anchors still in play, no peer ranging, no distributed
assignment, no letters/arbitrary shapes, no outdoor, no obstacles, no
multi-dropout.

**Deliverable (Paper 1):** *"Sim-to-real GNN-based decentralized formation
control on micro-quadrotors."* Core contribution: closing the sim-to-real
gap for a GATv2-policy-based multi-agent formation system; demonstrated
single-dropout recovery.

**Risk hot-spots:** motor saturation not captured in sim; UWB latency ≠ sim
latency; battery sag changing thrust-to-weight mid-flight.

---

### Phase 3: Decentralized Assignment + Peer Ranging *(Paper 2)*

**New capability:** no fixed infrastructure.

**Scope (three concurrent subsystems):**

1. **Peer UWB ranging.** Replace LPS anchors with drone-to-drone distance measurements only.
2. **Distributed state estimation.** Each drone estimates its position in the swarm-relative frame via multilateration over peer ranges + IMU fusion.
3. **Distributed slot assignment.** Auction algorithm (Bertsekas-style) over the peer mesh; replaces central slot precomputation.
4. **Gossip command dissemination.** Versioned command flooding; any-drone-hears → everyone-acts.
5. **Distributed centroid consensus.** Average consensus over IMU positions, used as shape anchor.

**Keeps simple:** still 3–5 Crazyflies, still indoor, still static shapes,
still single-dropout only.

**Deliverable (Paper 2):** *"Fully decentralized GPS-denied formation control
with peer-range localization."* Core contribution: the four decentralized
subsystems composed together and end-to-end validated on real hardware.

**Risk hot-spots:** peer-range localization accuracy may be insufficient for
tight formation error targets; auction convergence under realistic comms
loss; centroid drift in relative-frame operation.

---

### Phase 4: Scale & Expressive Shapes *(Paper 3)*

**New capability:** formation library + size-agnostic policy.

**Scope:**

- Scale from 5 → 10+ drones
- Parametric shape generator (circle, polygon with arbitrary N, letters via glyph-to-points, arbitrary uploaded point clouds)
- Time-varying formations (morphing between shapes, rotating formations)
- Policy retrained with curriculum over shape-count and shape-type

**Keeps simple:** still indoor, still Crazyflie, still no obstacles, still
limited fault modes.

**Deliverable (Paper 3):** *"Scalable multi-shape formation control with a
single shared policy: alphanumeric and time-varying targets."*

**Risk hot-spots:** assignment complexity grows with N (auction convergence
time); policy generalization to out-of-distribution shapes; crazyflie
formation density limits (downwash becomes dominant at close spacing).

---

### Phase 5: Outdoor + Extended Fault Tolerance *(Paper 4)*

**New capability:** outdoor calm environment + broader fault model.

**Scope:**

- Move outdoors to a calm, open field
- **Likely hardware upgrade.** Crazyflies are practically indoor-only; need Holybro X500 / ModalAI Seeker-class or similar with real outdoor endurance.
- Wind domain randomization (0–5 m/s, gust models)
- Multi-simultaneous dropout, comms link failure, sensor degradation spikes, operator disconnect autonomy, battery-graceful-degradation
- GPS still disabled; continues peer-ranging approach

**Keeps simple:** still no obstacles, policy inference may still be offboard
(phase 6 problem).

**Deliverable (Paper 4):** *"Fault-tolerant decentralized formation control
in outdoor calm conditions."*

**Risk hot-spots:** entire hardware stack changes, so the sim-to-real gap
re-opens; UWB range-finding outdoors has different multipath characteristics;
battery scaling implications on swarm; recovering test hardware after crashes.

---

### Phase 6: Onboard Compute + Obstacle-Aware Navigation *(Paper 5)*

**New capability:** unknown-environment navigation.

**Scope:**

- Upgrade to Jetson Orin Nano-class airframes (ModalAI VOXL2, or custom Jetson-Orin-Nano carrier on Holybro frame)
- Onboard GATv2 inference (≥ 50 Hz target)
- Onboard perception: stereo depth or lidar, VIO for self-motion
- Reactive obstacle avoidance composed with formation control (CBFs treating obstacles as virtual agents, as explored in your capstone Phase 4)
- Unknown environments (no prior map)

**Keeps simple:** may reduce scale temporarily (5–10 drones) due to hardware
cost; single platform only.

**Deliverable (Paper 5):** *"Obstacle-aware decentralized formation control
in unknown outdoor environments."*

**Risk hot-spots:** real-time GATv2 inference budget on Orin Nano; CBF QP
solve time onboard; perception failure modes (reflective surfaces, shadows,
sparse visual features); composing perception latency with formation control
dynamics.

---

### Phase 7: Hardware-Agnostic & General-Purpose *(Paper 6, stretch)*

**New capability:** platform transfer + mission generalization.

**Scope:**

- Train with domain randomization over multiple quad platforms (mass, inertia, thrust-curve, size ranges)
- Demonstrate the same policy checkpoint deployed on 2+ real quad platforms without retraining
- Expand mission set: formation + area-coverage search + escort + navigation
- Task-conditioned policy or mission-parameter observation extension

**Deliverable (Paper 6):** *"Hardware-agnostic policy transfer for
multi-mission decentralized drone swarms."*

---

## 7. Phase Dependencies

```text
Phase 0 (capstone) ─→ Phase 1 (shared-scene)
                          │
                          ▼
                     Phase 2 (sim-to-real, Crazyflie+LPS)   ← Paper 1
                          │
                          ▼
                     Phase 3 (decentralized)               ← Paper 2
                          │
                          ├─────────────────────┐
                          ▼                     ▼
                     Phase 4 (scale+shapes)   Phase 5 (outdoor+faults)
                     ← Paper 3                ← Paper 4
                          │                     │
                          └─────────┬───────────┘
                                    ▼
                             Phase 6 (onboard+obstacles)   ← Paper 5
                                    │
                                    ▼
                             Phase 7 (platform transfer)   ← Paper 6
```

Phases 4 and 5 can execute in parallel if hardware budget allows two test
stacks. Everything else is strictly sequential.

---

## 8. Cross-Cutting Infrastructure

Work that serves multiple phases and needs to be built once:

- **Simulation environment.** Isaac Lab 2.3 scene templates, domain randomization rig, deterministic eval harness.
- **Training infrastructure.** Local RTX 3070 for debugging, cloud L4/A100 for sweeps; checkpoint versioning; experiment tracking (W&B or equivalent).
- **Hardware pipeline.** Drone build process, pre-flight checklist, spare parts stock, battery rotation, crash-recovery SOPs.
- **Test environments.** Indoor volume (phases 2–4), outdoor calm field (phase 5+), outdoor obstacle course (phase 6).
- **Evaluation benchmarks.** A fixed set of repeatable scenarios run at the end of each phase to detect regressions.
- **Safety systems.** Emergency land triggers, geofences, kill switch, log-everything black-box.
- **Documentation.** Per-phase report, per-paper manuscript, open-source repo hygiene.

---

## 9. Risk Register (top items by phase)

| Phase | Risk | Likely mitigation |
| :--- | :--- | :--- |
| 1 | Shared-scene training too slow on 3070 | Debug locally, sweep on cloud |
| 2 | Sim-to-real gap doesn't close | Extended domain randomization; real-to-sim calibration loop |
| 2 | Battery sag breaks policy | Voltage as explicit observation or randomized thrust scaling |
| 3 | Peer-range localization too noisy for target formation error | Accept larger formation error, or add lightweight visual fiducials |
| 3 | Auction assignment doesn't converge with packet loss | Stale-aware bidding; eventual-consistency assignment |
| 4 | Auction doesn't scale to 20+ drones | Hierarchical clustering or consensus-based bundle algorithm |
| 5 | Wind exceeds DR envelope | Data-collection from real flights → retrain |
| 5 | Hardware crashes destroy budget | Start with 3 larger drones, not 10 |
| 6 | Onboard inference can't hit 50 Hz | Policy distillation; MLP fallback from GATv2 teacher |
| 6 | Perception fails in edge-case environments | Narrow the "unknown environment" definition in the paper |
| 7 | Platform transfer doesn't generalize | Frame as "family of platforms within randomization envelope," not true zero-shot |

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
7. **Publishing venues.** ICRA/IROS/RSS/L4DC per paper?
8. **Open-source strategy.** Fully open? Core open, applications reserved?
9. **Hardware platform decisions** per phase (Phase 5 and 6 specifically: Crazyflie replacement and Jetson carrier).
10. **Realistic self-funded hardware budget ceiling.** Informs phase sequencing and parallelism.

---

*Next revision should close out Section 4 (performance targets) and the
open questions in Section 10.*
