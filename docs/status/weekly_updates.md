# Weekly Status Updates

Formatted for synchronization with the project website.

## Week 10 (2026-03-17)

### What project milestones did you accomplish this week? If you're working in a team, please list what you personally contributed, not the project status overall

I finalized the foundational Isaac Lab MARL environment (`GGSwarmMarlEnv`), including multi-agent Crazyflie spawning, observation/action spaces, distance-based graph connectivity, and reward shaping. I also set up the MAPPO training pipeline in SKRL and validated it via `phase1_demo.py`. While I fell behind in getting the initial environment setup and running—meaning Phase 2 completion is roughly two weeks behind my original March 3rd target—I have replanned my schedule and am back on track to complete Phase 2 in the next 2 days. Since this is a solo project, I individually contributed all architecture and codebase development.

### What is your plan for next week?

My plan is to fully complete Phase 2 (Brain Development) in the next 2 days by successfully training the GATv2 coordination policy. After that, I will transition into Phase 3, which focuses on MINCO trajectory optimization and SwarmRaft decentralized consensus. I know Phase 3 will represent the bulk of the project's work, but my replanned timeline still leaves me sufficient time to complete that task before the final showcase.

### What challenges, if any, are you currently facing in project development? Do you need instructor assistance?

My primary challenge has been the sheer complexity of combining many advanced topics and ideas (Isaac Lab, PyTorch Geometric, SKRL, MARL) while simultaneously having to learn them from scratch. While falling behind on the environment setup was tough, breaking through those integration hurdles has already proven to be a very rewarding experience. At this time, I do not need instructor assistance.

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
