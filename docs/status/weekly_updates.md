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


