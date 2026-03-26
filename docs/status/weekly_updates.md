# Weekly Status Updates

Baseline timeline is in the [Proposal](../project/proposal.md#7-timeline-and-milestones)


| Weeks | Dates                                | Phase                | Activity                                                                                              | Milestone                                        |
| ----- | ------------------------------------ | -------------------- | ----------------------------------------------------------------------------------------------------- | ------------------------------------------------ |
| 5–6   | Feb 5 – Feb 17                       | 1. Foundation        | Install NVIDIA Isaac Lab; configure simulated multirotor assets; finalize graph connectivity logic.   | -                                                |
| 7–8   | ~~Feb 18 – Mar 3~~ → Feb 25 – Mar 24 | 2. Brain Development | Train GATv2 with MAPPO; test formation in empty space.                                                | ~~M1 Week 8~~ → M1 Week 11                       |
| 9–11  | ~~Mar 4 – Mar 24~~ → Mar 25 – Apr 7  | 3. Muscle Refinement | Integrate MINCO post-processing and SwarmRaft consensus logic.                                        | ~~M2 Week 11 (3/24)~~ → M2 Week 13 (4/7)         |
| 12–13 | Apr 8 – Apr 14                       | 4. Stress Testing    | Conduct simulated agent loss tests; benchmark swarm navigation in high-density obstacle environments. | M3 (Week 14, 4/14): Mission success validation   |
| 14–15 | Apr 14 – Apr 21                      | 5. Showcase Prep     | Finalize RTX tiled rendering; record HD demo; compile results.                                        | M4 (Week 15, 4/21): HD showcase + Testing Report |
| 16    | Apr 22 – Apr 24                      | 6. Delivery          | Present at Capstone Festival; submit Portfolio and Learning Journals.                                 | **Final Presentation due 4/24/26**               |


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


