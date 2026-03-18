# Weekly Status Updates

Formatted for synchronization with the project website.

*For the high-level project timeline, please refer to the **[Development Phases Schedule in the README](../README.md#development-phases)***.

## Week 11 (2026-03-18)

- **Phase 2 Development Started:** GNN message passing layer (L2) implemented.
- Added PyTorch Geometric `GATv2Conv` policy wrapper to SKRL.
- Parameterized environment constants and implemented Curriculum Reward Shaping (Formation, Cohesion, Separation).
- Successfully ran verification smoke tests with the hybrid Isaac Lab/PyG pipeline.

## Week 10 (2026-03-17)

- **Milestones:** Finalized foundational MARL environment (`GGSwarmMarlEnv`) including multi-agent spawning, graph connectivity, and reward shaping. Validated MAPPO pipeline via `phase1_demo.py`. Replanned schedule to catch up.
- **Next Week's Plan:** Complete Phase 2 (GATv2 coordination policy) over the next two days, then transition to Phase 3 (MINCO trajectory optimization and SwarmRaft consensus).
- **Challenges:** High learning curve integrating Isaac Lab, PyTorch Geometric, SKRL, and MARL concepts. No instructor assistance needed yet.

---

## Week 9 (2026-03-09)

- **Milestones:** Configured local Isaac Sim/Lab environment, ran standard Crazyflie flight tests, and built structured project repository (GNSC 5-layer architecture documentation, reporting framework, rules).
- **Next Week's Plan:** Dive into Isaac Lab API, begin training the GATv2/PPO policy, and complete tutorials.
- **Challenges:** Transitioning from single-drone simulation to full MARL swarm coordination via Isaac Lab's internal logic. All progress via self-guided research.
