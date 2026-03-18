# Weekly Status Updates

Formatted for synchronization with the project website.

## Week 10 (2026-03-16)

### Week 10 Status: Completed

### 1. Week 10 Milestones

- **Phase 1 Complete:** Finalized the foundational Isaac Lab MARL environment (`GGSwarmMarlEnv`), including multi-agent Crazyflie spawning, 12-dim observation space, 4-dim action space, distance-based graph connectivity (L2 adjacency matrix), reward shaping, and reset logic.
- **Verification:** Created and ran `phase1_demo.py` to visually confirm drone rendering in Isaac Sim and validate real-time adjacency matrix computation.
- **SKRL Pipeline:** Set up the MAPPO training pipeline with `train.py`, `play.py`, and `skrl_mappo_cfg.yaml`.
- **Documentation:** Expanded `phase1_foundation.md` into a comprehensive technical reference. Updated changelog with all Phase 1 milestones.
- **Code Cleanup:** Renamed environment files to `drone_swarm_env.py` / `drone_swarm_env_cfg.py` for clarity.

### 2. Week 11 Plan

- **Phase 2 Kickoff (Brain Development):** Begin training the GATv2 coordination policy using PPO/MAPPO via SKRL.
- **SKRL Config Tuning:** Increase network capacity from `[32, 32]` to larger layers, extend training timesteps, and fix the experiment directory name.
- **Reward Evolution:** Evolve from hover-in-place rewards to formation-aware rewards (inter-agent spacing targets).
- **GATv2 Integration:** Start building a custom GNN policy that consumes the adjacency matrix to enable spatial reasoning.

### 3. Week 10 Challenges

- **GATv2 Architecture:** Designing the custom GNN policy network that integrates with SKRL's model interface requires careful study of both frameworks.
- **Reward Engineering:** Transitioning from simple position rewards to formation-aware rewards without causing training instability.
- **No instructor assistance needed** at this time.

---

## Week 9 (2026-03-09)

### Week 9 Status: Completed

### 1. Week 9 Milestones

This week was focused on setting up the development environment:

- **Environment Setup:** Installed NVIDIA Isaac Sim and Isaac Lab locally. Verified hardware performance for development.
- **Initial Testing:** Successfully ran the Crazyflie drone model example in Isaac Sim and began initial flight tests (straight-line flight attempts).
- **Project Structure:** Established repository infrastructure, including GNSC 5-layer architecture documentation, status reporting framework, and project governance rules.

### 2. Week 10 Plan

- **Isaac Lab API Familiarity:** Dive deeper into the Isaac Lab API to gain precise control over drone model articulations.
- **Brain Development Initialization:** Begin training the GATv2 coordination policy using Proximal Policy Optimization (PPO).
- **Tutorial Progression:** Complete the Isaac Lab tutorial series to bridge the gap between simulation and training logic.

### 3. Week 9 Challenges

- **Technical Hurdle:** Mastering the Isaac Lab API for multi-agent reinforcement learning (MARL) control and training.
- **Learning Curve:** Transitioning from basic drone simulation to complex swarm coordination requires significant familiarity with the framework's internal logic.
- **Assistance:** No direct instructor assistance is needed at this time; progress continues through self-guided tutorials and documentation research.
