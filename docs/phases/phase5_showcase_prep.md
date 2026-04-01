# Phase 5: Showcase Prep

**Timeline:** Apr 14 -- Apr 20  |  **Gate:** M4 -- HD showcase and Testing Report delivered

## 1. Goals

| ID | Goal | Success Criteria |
| :--- | :--- | :--- |
| P5.1 | Cinematic trailer (~2:30) | Tron-inspired, 1080p 60fps, 8 scenes, formation morphing + dropout |
| P5.2 | Proposal objectives verified | Testing Report finalized with pass/fail for O1-O4 |
| P5.3 | Formation error < 0.3m steady-state | Verified across evaluation suite |
| P5.4 | Presentation-ready repository | Clean README, reproducible commands |

## 2. Cinematic Trailer (P5.1)

### Concept

Tron Legacy-inspired cinematic showcase. Dark void, volumetric teal fog,
emissive drone materials, cinematic camera rigs. Signature color: #64d5d2
(Gary Gigabytes teal). ~2:30 runtime at 1080p 60fps.

### Storyboard

```mermaid
flowchart TD
    S1["Scene 1: Cold Open (3s)<br/>Black void, grid materializes from fog"] --> S2
    S2["Scene 2: Drone Spawn (5s)<br/>8 drones appear, GNN edges visible as teal lines"] --> S3
    S3["Scene 3: Octagon Formation (5s)<br/>Formation snaps into place — orbit camera"] --> S4
    S4["Scene 4-6: Formation Morphing (15s)<br/>Octagon → Triangle → Grid → Letter G — top-down camera"] --> S5
    S5["Scene 7: Drone Failure (5s)<br/>One drone LED cuts out, edges snap — low angle"] --> S6
    S6["Scene 8: SwarmRaft Recovery (10s)<br/>Heptagon reforms within 2s — orbit camera"] --> S7
    S7["Scene 9: Scale-Up (15s)<br/>20-agent polygon fills the frame"] --> S8
    S8["Scene 10: Title Card (5s)<br/>ggSwarm — NVIDIA Isaac Lab — Spring 2026"]

    style S1 fill:#141818,color:#64d5d2
    style S2 fill:#141818,color:#64d5d2
    style S3 fill:#2e4c5b,color:#64d5d2
    style S4 fill:#2e4c5b,color:#64d5d2
    style S5 fill:#e74c3c,color:#fff
    style S6 fill:#2ecc71,color:#fff
    style S7 fill:#3498db,color:#fff
    style S8 fill:#141818,color:#64d5d2
```

### Technical Implementation

| Component | File | Description |
| :--- | :--- | :--- |
| Tron environment | `scripts/tron_env.py` | Black void, fog, grid, lighting, drone materials, camera rigs |
| Showcase script | `scripts/showcase.py` | Scene sequencer, formation morphing, dropout orchestration |
| Trained policy | p4-6 checkpoint | Triangle mesh + dropout trained, generalizes to all shapes |

### Tron Environment Features

- **Black void stage** — default lights removed, RTX path tracing enabled (32 SPP)
- **Volumetric fog** — tinted #64d5d2, subtle density (0.08), 3-25m range
- **Emissive grid plane** — 50m quad with teal OmniPBR emissive material
- **3-point cinematic lighting** — key (teal-white), fill (navy), rim (bright teal edge glow)
- **Drone material** — near-black metallic body + #64d5d2 emissive LED at intensity 10

### Camera Rigs (TronCameraRig)

| Mode | Description | Used In |
| :--- | :--- | :--- |
| `orbit` | Slow 360° around swarm centroid | Scenes 1-3, 8-10 |
| `top_down` | Locked overhead — formation shape reveal | Scenes 4-6 |
| `low_angle` | Dramatic ground-level looking up | Scene 7 (drone failure) |
| `chase` | Follows swarm centroid | Optional |

### Run Command

```powershell
env_isaaclab/Scripts/python.exe scripts/showcase.py `
    --task ggswarm-v0 `
    --checkpoint logs/skrl/ggswarm/p4/<best_run>/checkpoints/best_agent.pt `
    --prefix p5-1-showcase
```

## 3. Testing Report (P5.2)

Compile Phase 4 evaluation results into `docs/testing_report.md`:

- Objective pass/fail table (O1-O4) with measured values
- Formation error metrics by scenario (polygon, grid, letter, scale)
- SwarmRaft recovery time measurements
- MINCO jitter A/B comparison data
- Scale benchmark results (8, 10, 15, 20 agents)
- Trajectory plot gallery

## 4. Design Integration

No architectural changes. Consumes Phase 4 validated stack and packages
for presentation.

| Deliverable | Source |
| :--- | :--- |
| Cinematic trailer | `scripts/showcase.py` + `scripts/tron_env.py` |
| Demo clips | `scripts/skrl/play.py` with `--prefix` |
| Trajectory plots | `ggswarm.viz.trajectory_plots` |
| Testing Report | Phase 4 evaluation data + `scripts/eval_metrics.py` |

## 5. Results

Phase 5 early start (Mar 31) — showcase script and Tron environment created.
Full recording scheduled for Apr 14-20 after Phase 4 testing complete.

- **p5-1:** Showcase script created. Tron environment with fog, grid, lighting.
  8-scene cinematic sequence with formation morphing + SwarmRaft dropout.
  Using p4-6 checkpoint (triangle + dropout, 1000 iter).

---

## See Also

- [Phase 4: Stress Testing](phase4_stress_testing.md)
- [Phase 6: Delivery](phase6_delivery.md)
- [Assumptions](../design/assumptions.md)
