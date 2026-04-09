# Weekly Status Updates

Baseline timeline is in the [Proposal](../project/proposal.md#7-timeline-and-milestones)

> **Week numbering convention:** Academic weeks run Wednesday → Tuesday.
> Updates are posted each Tuesday (end of week). For example, Week 12
> runs Wed Mar 25 – Tue Mar 31, and the update is posted on 2026-03-31.

| Weeks | Dates | Phase | Activity | Milestone |
| :--- | :--- | :--- | :--- | :--- |
| 5–6 | Feb 5 – Feb 17 | 1. Foundation | Isaac Lab setup, env implementation | - |
| 7–8 | Feb 25 – Mar 24 | 2. Brain Development | PPO hover + formation control | M1 (Mar 25): **Complete** |
| 9–11 | Mar 25 – Mar 29 | 3. Muscle Refinement | MINCO, CBF, SwarmRaft | M2 (Mar 29): **Complete, 9 days early** |
| 12–13 | Mar 30 – Apr 13 | 4. Stress Testing | Agent loss, obstacles, scale benchmarks | M3 (Apr 13): Mission validation |
| 14–15 | Apr 14 – Apr 20 | 5. Showcase Prep | RTX rendering, HD demo, Testing Report | M4 (Apr 20): HD showcase |
| 16 | Apr 21 – Apr 24 | 6. Delivery | Capstone Festival, submissions | **Final: Apr 24** |

---

## Week 15 Update (2026-04-08) — Phase 5 COMPLETE, Phase 6 kickoff

### Headline

**Phase 5 is done, 12 days ahead of the M4 gate.** The cinematic
trailer was edited from the 20 captured clips and published to
YouTube. A new project banner (`docs/assets/banner.jpg`) was added to
the README, and the README schedule + status badge were rolled
forward to reflect Phases 4 and 5 complete. **Phase 6 (Delivery) is
now in progress** — focus shifts to the Capstone Festival
presentation, Portfolio + Learning Journal submissions, final
documentation sweep, and a clean-clone reproducibility check.

### What happened

- Cinematic trailer cut and uploaded to YouTube (general-audience
  description targeting drones / AI / reinforcement learning).
- README banner added; schedule table and status badge updated to
  Phase 5 Complete.
- Phase 5 and Phase 6 docs updated to reflect new status.

### Next

- Manual rehearsal of the Capstone Festival presentation.
- Portfolio + Learning Journal final pass.
- Reproduce repo from a clean clone using only the README.

---

## Week 14 Update (2026-04-08) — Phase 5 sub-A complete, 20 cinematic clips captured

### Headline

Phase 5 sub-phase A is **done**. The Tron-styled cinematic pipeline is
fully wired into `--tron` on play.py, with four camera modes and clean
visual baseline. **17 stitchable clips** captured across orbit /
top-down / chase camera modes for octagon, triangle, grid, letter G
(8 + 16 agents), 20-agent scale-up, dropout, forest navigation, and
forest with red wireframe trees. Enough material to manually stitch
the cinematic trailer in DaVinci Resolve / Premiere without further
code changes.

### What happened

- **Tron baseline locked in (commit `eb958dd0`).** After two days of
  guess-and-check that ended in researching the [IsaacLab GitHub
  issue #622](https://github.com/isaac-sim/IsaacLab/issues/622) and
  finding the root cause (USD instancing freezes visuals at the
  prototype level — `make_uninstanceable` first), the working Tron
  pipeline is: remove default lights via stage traversal,
  `make_uninstanceable` on each Drone, edit the existing `DroneMat`
  shader's `diffuseColor`/`emissiveColor` to amber linear-RGB
  `(1.0, 0.262, 0.0)`, remove the texture-based terrain, spawn a
  custom 50m flat plane + 102 thin teal grid line quads. Drives the
  env render camera via `env.unwrapped.sim.set_camera_view()` per
  frame (NOT `viewport.set_active_camera`, which only updates the
  live Kit viewport — NvencRecorder reads from `env.render()`).
  Phase 5 doc § 0 captures the full setup + all the
  things-that-look-like-they-should-work-but-don't traps. See the
  changelog for the full debugging journey.

- **Forest cylinder wireframe (commit `b0070351`).** When `--tron`
  and `--forest` are both set, each forest cylinder is restyled as a
  Tron wireframe: cylinder body painted black (silhouette), 24 thin
  red vertical strips around the circumference, 5 mid-height annulus
  rings + 1 thicker top-edge ring. Annulus rings are constructed as
  flat torus-like meshes (32 angular segments between an inner and
  outer radius), NOT solid cylinder primitives — those rendered as
  disks because Cylinder is a solid volume. Forest navigation
  behavior is unchanged.

- **`--cam_mode` flag with 4 modes (commit `2d15fa03`).** orbit
  (default, slow rotation around centroid), top_down (locked
  overhead at 3m + subtle yaw drift), low_angle (dramatic
  ground-level looking up, slow arcing), chase (trails behind the
  centroid in -X for forest fly-through). All four modes drive
  `sim.set_camera_view()` per frame so the recorded video follows
  the chosen view. Tunable knobs in the `tron_orbit` dict at the
  top of the `--tron` block.

- **`--cloud` flag (in commit `b0070351`).** Sets
  `formation_mode = "cloud"` before `gym.make()` so play time uses
  the boid-like cloud reward path. **Off-distribution** for the
  polygon-trained `p4-revert-4` checkpoint, so the result is not
  used for the final cinematic. A fresh cloud-mode training run is
  deferred — out of GCE credits.

### 17 captured clips

Inventory in [`videos/showcase/`](../../videos/showcase/) at the repo
root. Phase 5 doc § 0 has the full table with mode + formation +
agent count + length per clip. All renderable from a single play.py
invocation with flag combinations against the `p4-revert-4`
production checkpoint:

- **Orbit (10s × 8, 15s × 1)**: octagon, triangle, grid,
  letter_G (8 + 16 agents), letter_G-16-15s, scale-20 (polygon),
  dropout (triangle + dropout), forest with wireframe trees
- **Top-down (10s × 5, 15s × 1)**: triangle, octagon, grid,
  letter_G-16, letter_G-16-15s, forest
- **Chase (15s × 1)**: forest navigation
- **Off-distribution (10s × 1)**: orbit cloud-mode (not used)

### Next

Manual cinematic editing in DaVinci Resolve / Premiere using the
captured clips + title cards + music. **No more code changes
needed for the trailer.** Phase 5 sub-B/C/D (cinematic lighting,
scene morphing sequencer, full automated cinematic) all deferred —
the manual workflow with 17 clips covers the Capstone Festival
deliverable.

Open items (only if there's time):

- Drone color refinement (currently reads more yellow than amber —
  the linear-RGB gamma trap from the debugging journey)
- Cold open clip (no drones in frame) — needs a small `--no_drones`
  flag
- HD upgrade — re-render best clips at 60fps instead of current 30fps
- Lift Tron color constants into `GgswarmEnvCfg` for cleaner config

- **Timeline:** 16 days remain to deadline (Apr 24).

---

## Week 13 Update (2026-04-07) — Phase 4 mid-flight: regression recovery + forest fix

### Headline

Forest navigation now works cleanly. **0 body penetrations** in 700-step forest
play with 8 drones and 0.20m-radius (40cm-diameter) tree trunks. The recovery
required walking back through three compounding bugs and one failed
hyper-tuning experiment. p4-revert-4 checkpoint (reward 66.83 / ep_len 307.74)
trains slightly *better* than the Mar 31 baseline.

### What happened

- **Reverted the learned-obstacle-avoidance experiment** (`fa2e16ab`→`428b2f2c`).
  Six commits worth of obstacle penalty + obs columns + curriculum + static
  goal placement gave zero measurable benefit across six p4-obstacle GCE runs.
  Experimental work archived to `experimental/learned-obstacle-avoidance`
  branch; main returned to the goal-deflection approach.

- **Diagnosed and recovered from cfg drift regression.** Two retrain attempts
  on the reverted code (`p4-revert-1`, `p4-revert-2`) collapsed to reward
  24 / 7 vs the Mar 31 baseline of 63. An Explore-agent diagnostic confirmed
  the env code path was character-identical to Mar 31 — root cause was 100%
  cfg drift in `cbf_d_safe`, `cbf_max_correction`, `collision_radius`, and
  `dropout_enabled`. A third attempt (`p4-revert-3`) raised
  `cbf_max_correction` to 0.50 based on a wrong claim that Mar 31 had no
  field; in reality the value was hardcoded `_MAX_CORRECTION = 0.15` in
  `cbf.py` since `daae89c6` (Mar 28). The 0.50 value reproduced the p3-16
  unclamped-CBF flip regression. `p4-revert-4` reverts every drifted field
  to its true Mar 31 value and trains to reward 66.83 / ep_len 307.74.

- **Found and fixed two compounding goal-deflection bugs**:
  1. **Deflection used goal position not drone position** — drones whose
     slots happened to land just outside the deflection radius drifted
     toward cylinders due to formation pressure and were never protected.
  2. **Base goal advanced unconditionally** — when a drone got stuck at a
     cylinder, its goal kept advancing at 0.5 m/s and was up to 4.12m ahead
     by step 650, drowning out the lateral deflection with a runaway X
     gradient. Fix: cap the goal lead over the drone with new cfg field
     `forest_max_goal_lead = 0.5m`.

- **Boids-style flock alignment for deflection direction.** Replaced pure
  radial deflection with a 70/30 lateral/radial blend. Lateral side is
  picked using mean K-nearest neighbor velocity (boids alignment principle),
  with fallback to drone's own velocity, with final fallback to geometric
  Y-sign. Drones now coordinate which side of a cylinder to dodge.

- **`apply_cbf_obstacles` constants promoted to function parameters.** Pure
  cleanup per CLAUDE.md "no magic numbers". Function remains uncalled but is
  now ready for re-enablement as an action-space backstop if needed.

- **Forest cylinders widened to 40cm diameter** for visual realism (young
  tree trunks). Bumped `cbf_obstacle_d_safe` to 0.60m to compensate for the
  reduced edge-to-edge reaction margin.

### Forest play results (p4-revert-4 checkpoint, 700 steps × 8 drones)

| Metric | broken (start of week) | after fixes |
| :--- | :--- | :--- |
| Body penetrations | 128 (and stuck at start) | **0** |
| Min body clearance | -0.049m | **+0.037m (positive)** |
| Min pair distance | 0.100m | 0.177m |
| Final mean X | -0.28 (didn't traverse) | +6.22 |
| Final min X | -0.88 (drones piled up) | +5.39 (no stuck drones) |
| Stuck drones | 1+ | **0** |

### Discovered along the way

- **The historical "0 hits" claim from p4-forest-14/15/16 and the M3 gate
  was a measurement bug.** Body penetrations were never counted with the
  drone radius (0.10m) included. Re-measured every historical forest run
  and found they all grazed cylinders by ~5cm (41–309 body penetrations
  per 700-step run). M3 gate's "obstacle success rate" still passes for
  *forward progress*, but the proximity claims need to be re-stated
  (done in `phase4_stress_testing.md` § 4 and § 5.5).

- **Drones currently hover stacked on top of each other at spawn.** Real
  Crazyflies have a downwash interaction zone — the upper drone destabilizes
  the lower one. With the recent runs the drones are no longer bunching up
  at spawn so this isn't blocking anything. Logged as **post-capstone future
  work** (real-world hardware fidelity improvement, not on the critical path).

### Phase 4 wrap-up

**Phase 4 COMPLETE.** All seven M3 criteria pass (with the corrected
body-radius obstacle metric). Production checkpoint locked in at
`logs/skrl/ggswarm/p4/2026-04-06_21-09-24_ppo_torch/`.
**Phase 5 (Showcase Prep) starts 2026-04-07, 7 days early** vs the
original Apr 14 schedule.

### Next

- **Phase A of Phase 5:** add `--tron` flag to `play.py` that calls
  `setup_tron_environment()` after `gym.make()`. ~30 lines, lets us
  capture the first Tron-styled forest video same session.
- **HD demo capture** with the trees2 forest config (40cm trunks) and
  the production checkpoint.
- **Phases B–D of Phase 5:** Tron visual debug, formation morphing
  scene sequencer, full cinematic trailer (~2:30 at 1080p 60fps).

- **Timeline:** 17 days remain to deadline (Apr 24).

---

## Week 12 Update (2026-03-31) — Phase 3 Complete

### Phase 2 wrap-up (Mar 25-27)

- **FRESH START + FORMATION IN ONE DAY.**
  Archived old Phase 2 codebase, rebuilt from Isaac Lab quadcopter reference,
  and achieved working formation control — all in a single session.
- **Phase 2A (hover):** 4 runs (p2a-1 through p2a-4). Hover solved with
  `DirectRLEnv` + PPO shared policy (CTDE). ep_len 499/500.
- **Phase 2B (formation):** 8 runs (p2b-1 through p2b-8). Progressive fixes:
  - p2b-1/2: formation reward ~0 (curriculum too slow, env_origins not subtracted)
  - p2b-3: formation signal appearing (env_origins fix)
  - p2b-4: BREAKTHROUGH — formation reward 3.5 mid-training, then collapsed
  - p2b-5: stable formation (reward 1.59, no collapse) but play showed tumbling
  - p2b-8: **ALL FIXES APPLIED.** ep_len min=499 (zero crashes), formation=1.39.
    Drones visually form triangle with fixed centroid goal.
- **Key bugs found and fixed:**
  - SKRL bypasses gym wrappers → formation logic in env, not wrapper
  - env_origins subtraction for neighbor obs
  - Episode timeout stagger caused drone "despawn" → synced within groups
  - Correct circumradius formula for formation offsets
  - Group-aware goal sampling with shared centroid
- **New tooling:** NVENC video, trajectory plots with centroid/inter-drone distance,
  random spawn positions, fixed centroid for play mode.
- **Architecture:** `GgswarmEnv(DirectRLEnv)`, 1 drone/env, PPO shared policy,
  formation via expanded obs (18D) + curriculum-scaled formation reward.

### Phase 3: Muscle Refinement (Mar 28-29)

- **PHASE 3 COMPLETE. M2 gate met 9 days early.**
  Implemented all L2-L4 layers: GATv2 GNN with K-hop sparse edges, MINCO
  minimum-jerk trajectory filter, CBF-QP safety shield, SwarmRaft agent
  dropout, virtual collision detection, and KNN-based cohesion. 13 training
  runs (p3-14 through p3-26) across two days.

- **CBF-QP safety filter (L4):** Rewrote from heuristic moment injection to
  proper barrier constraint enforcement. p3-16 unclamped corrections caused
  drone tumbling (ep_len 18). p3-17 fixed with clamped corrections (_MAX=0.15):
  reward 65.3, ep_len 463.

- **MINCO minimum-jerk filter (L3):** Single-segment min-jerk trajectory
  optimization. p3-18 (T=0.10s) too sluggish — drones crashed (ep_len 51).
  p3-19 (T=0.04s) restored stability: reward 46.3, ep_len 472, visibly
  smoother attitude than EMA baseline.

- **Critical fix: MINCO-CBF state sync.** MINCO was overwriting CBF corrections
  every step. Syncing `_minco_pos` to post-CBF output made corrections sticky.
  p3-23: all 8 drones survive full 500 steps, KNN floor at 0.25m.

- **Virtual collision detection:** Pairwise distance check (r=0.10m) triggers
  collective group reset. Strongest training signal for separation learning.

- **KNN-based cohesion:** Replaced centroid cohesion (doesn't scale to 20+
  drones) with mean K-nearest neighbor distance reward.

- **SwarmRaft agent dropout (L3 consensus):** `_agent_alive` mask, random
  dropout at step 100-250, dead drones excluded from all computations.
  p3-26: reward 19.6, ep_len 332, 7/8 drones survive after dropout.

- **Best overall run: p3-24** — reward 36.4, ep_len 242, KNN 0.3-0.6m.
  All 8 drones survive full 500-step episode. Smooth attitude (+/-2 deg).

### Phase 4: Stress Testing (Mar 30-31)

- **Polygon-mode training (p4-1 to p4-4):** Switched from cloud to polygon/triangle
  formation. p4-3 (polygon, 500 iter): reward 56.4. p4-4 (triangle mesh): reward 62.9.
  Triangle mesh has more varied geometry, generalizes better to other shapes.
- **Formations module:** New `formations.py` with polygon, grid, triangle_mesh,
  and letter presets. Train once, play any shape via `--formation` arg.
- **Finding 1: Index-based slot assignment breaks at scale.** 16+ agents caused
  path crossings and collisions. Fixed with greedy nearest-slot matching.
- **Dynamic spawn radius:** Auto-scales with agent count to maintain 0.75m spacing.
- **SwarmRaft polygon dropout (p4-5, p4-6):** Dynamic slot recomputation for N-1
  agents. Octagon→heptagon visible. Recovery ~1.0s (passes O3 < 2.0s target).
- **Assumptions documented:** 8 simplifying assumptions with future work paths.
- **Remaining:** Scale testing (10-20 agents), forest navigation, MINCO validation,
  eval suite.

- **Timeline:** 24 days remain to deadline (Apr 24).

---

## Week 11 Update (2026-03-24)

- **Phase 2A COMPLETE.** Hover-stability solved after discovering the train-eval gap root cause:
`load_policy_from_checkpoint()` did not restore the `RunningStandardScaler` preprocessor
statistics. Fix: use SKRL's `agent.load()`. PD16 re-eval: 0.08° roll, 99.99% airborne,
zero crashes.
- **PD11–PD20 debugging journey:** Explored PD gain tuning (PD11–PD15), direct moments (PD16),
write_data_to_sim removal (PD17), entropy_loss_scale fix (PD18), Isaac Lab config matching
(PD19–PD20). All were red herrings — the real bug was in the eval checkpoint loader.
- **Key findings preserved for future phases:**
  - GNN adjacency matrix never reaches policy (Phase 2B blocker — needs pipeline fix)
  - `entropy_loss_scale` should stay 0.0 (prevents noise-dependent control)
  - Direct moments (no PD controller) is the correct approach
  - MLP sufficient for Phase 2A; GNN needed for Phase 2B+ formation
- **Phase 2B starting:** Formation hover control. Run naming: `p2b-1`, `p2b-2`, etc.
Priority: yaw control (drones drift ~0.4m over 10s due to uncontrolled spin).
- **Timeline:** Phase 2A took 1 day longer than planned. 30 days remain to deadline (Apr 24).

### Timeline and Milestone Snapshot (as of 2026-03-24)


| Weeks | Dates           | Phase                | Activity                                                                                            | Milestone                          |
| ----- | --------------- | -------------------- | --------------------------------------------------------------------------------------------------- | ---------------------------------- |
| 5–6   | Feb 5 – Feb 17  | 1. Foundation        | Install NVIDIA Isaac Lab; configure simulated multirotor assets; finalize graph connectivity logic. | -                                  |
| 7–11  | Feb 25 – Mar 25 | 2. Brain Development | Train GATv2 with MAPPO; test formation in empty space.                                              | M1: Phase 2A complete (Mar 25)     |
| 12–13 | Mar 26 – Apr 7  | 3. Muscle Refinement | Integrate MINCO post-processing and SwarmRaft consensus logic.                                      | M2 Week 13 (4/7)                   |
| 14    | Apr 8 – Apr 14  | 4. Stress Testing    | Conduct simulated agent loss tests; benchmark swarm navigation.                                     | M3 (4/14)                          |
| 15    | Apr 14 – Apr 21 | 5. Showcase Prep     | Finalize RTX tiled rendering; record HD demo; compile results.                                      | M4 (4/21)                          |
| 16    | Apr 22 – Apr 24 | 6. Delivery          | Present at Capstone Festival; submit Portfolio and Learning Journals.                               | **Final Presentation due 4/24/26** |


## Week 10 Update (2026-03-17)

- **Phase 2 Development Started:** GNN message passing layer (L2) implemented.
- Added PyTorch Geometric `GATv2Conv` policy wrapper to SKRL.
- Parameterized environment constants and implemented Curriculum Reward Shaping (Formation, Cohesion, Separation).
- Successfully ran verification smoke tests with the hybrid Isaac Lab/PyG pipeline.

### Timeline and Milestone Snapshot (as of 2026-03-18)


| Weeks | Dates                                | Phase                | Activity                                                                                              | Milestone                                        |
| ----- | ------------------------------------ | -------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| 5–6   | Feb 5 – Feb 17                       | 1. Foundation        | Install NVIDIA Isaac Lab; configure simulated multirotor assets; finalize graph connectivity logic.   | -                                                |
| 7–8   | ~~Feb 18 – Mar 3~~ → Feb 25 – Mar 24 | 2. Brain Development | Train GATv2 with MAPPO; test formation in empty space.                                                | ~~M1 Week 8~~ → M1 Week 11                       |
| 9–11  | ~~Mar 4 – Mar 24~~ → Mar 25 – Apr 7  | 3. Muscle Refinement | Integrate MINCO post-processing and SwarmRaft consensus logic.                                        | ~~M2 Week 11 (3/24)~~ → M2 Week 13 (4/7)         |
| 12–13 | Apr 8 – Apr 14                       | 4. Stress Testing    | Conduct simulated agent loss tests; benchmark swarm navigation in high-density obstacle environments. | M3 (Week 14, 4/14): Mission success validation   |
| 14–15 | Apr 14 – Apr 21                      | 5. Showcase Prep     | Finalize RTX tiled rendering; record HD demo; compile results.                                        | M4 (Week 15, 4/21): HD showcase + Testing Report |
| 16    | Apr 22 – Apr 24                      | 6. Delivery          | Present at Capstone Festival; submit Portfolio and Learning Journals.                                 | **Final Presentation due 4/24/26**               |


## Week 10 (2026-03-17)

- **Milestones:** Finalized foundational MARL environment (`GGSwarmMarlEnv`) including multi-agent spawning,
graph connectivity, and reward shaping. Validated the MAPPO pipeline using the new hover baseline and
unified run helper workflow. Replanned schedule to catch up.
- **Next Week's Plan:** Complete Phase 2 (GATv2 coordination policy) over the next two days, then transition to Phase 3 (MINCO trajectory optimization and SwarmRaft consensus).
- **Challenges:** High learning curve integrating Isaac Lab, PyTorch Geometric, SKRL, and MARL concepts. No instructor assistance needed yet.

### Timeline and Milestone Snapshot (as of 2026-03-17)


| Weeks | Dates                                | Phase                | Activity                                                                                              | Milestone                                        |
| ----- | ------------------------------------ | -------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| 5–6   | Feb 5 – Feb 17                       | 1. Foundation        | Install NVIDIA Isaac Lab; configure simulated multirotor assets; finalize graph connectivity logic.   | -                                                |
| 7–8   | ~~Feb 18 – Mar 3~~ → Feb 25 – Mar 24 | 2. Brain Development | Train GATv2 with MAPPO; test formation in empty space.                                                | ~~M1 Week 8~~ → M1 Week 11                       |
| 9–11  | ~~Mar 4 – Mar 24~~ → Mar 25 – Apr 7  | 3. Muscle Refinement | Integrate MINCO post-processing and SwarmRaft consensus logic.                                        | ~~M2 Week 11 (3/24)~~ → M2 Week 13 (4/7)         |
| 12–13 | Apr 8 – Apr 14                       | 4. Stress Testing    | Conduct simulated agent loss tests; benchmark swarm navigation in high-density obstacle environments. | M3 (Week 14, 4/14): Mission success validation   |
| 14–15 | Apr 14 – Apr 21                      | 5. Showcase Prep     | Finalize RTX tiled rendering; record HD demo; compile results.                                        | M4 (Week 15, 4/21): HD showcase + Testing Report |
| 16    | Apr 22 – Apr 24                      | 6. Delivery          | Present at Capstone Festival; submit Portfolio and Learning Journals.                                 | **Final Presentation due 4/24/26**               |


## Week 9 (2026-03-09)

- **Milestones:** Configured local Isaac Sim/Lab environment, ran standard Crazyflie flight tests, and built structured project repository (GNSC 5-layer architecture documentation, reporting framework, rules).
- **Next Week's Plan:** Dive into Isaac Lab API, begin training the GATv2/PPO policy, and complete tutorials.
- **Challenges:** Transitioning from single-drone simulation to full MARL swarm coordination via Isaac Lab's internal logic. All progress via self-guided research.

### Timeline and Milestone Snapshot (as of 2026-03-09)


| Weeks | Dates           | Phase                | Activity                                                                                                 | Milestone                                                                                             |
| ----- | --------------- | -------------------- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 5–6   | Feb 5 – Feb 17  | 1. Foundation        | Install NVIDIA Isaac Lab; configure simulated multirotor assets; finalize graph connectivity logic.      | -                                                                                                     |
| 7–8   | Feb 18 – Mar 3  | 2. Brain Development | Train the GATv2 policy using **Multi-Agent PPO (MAPPO)**; test basic formation keeping in empty space.   | **M1 (Week 8):** GNN policy training                                                                  |
| 9–11  | Mar 4 – Mar 24  | 3. Muscle Refinement | Integrate MINCO trajectory optimization as a post-processing layer; implement SwarmRaft consensus logic. | **M2 (Week 11, by 3/24):** Logic integration                                                          |
| 12–13 | Mar 25 – Apr 7  | 4. Stress Testing    | Conduct simulated agent loss tests; benchmark swarm navigation in high-density obstacle environments.    | -                                                                                                     |
| 14–15 | Apr 8 – Apr 21  | 5. Showcase Prep     | Finalize RTX Tiled Rendering; record HD demo; compile results.                                           | **M3 (Week 14, by 4/14):** Mission success validation; **M4 (Week 15):** HD showcase + Testing Report |
| 16    | Apr 22 – Apr 24 | 6. Delivery          | Present at Capstone Festival; submit Portfolio and Learning Journals.                                    | Final Presentation due 4/24/26                                                                        |


