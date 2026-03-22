# Phase 5: Showcase Prep (Weeks 14–15, Apr 14 – Apr 21)

Phase 5 converts the validated Phase 4 GNSC stack into the project's public-facing
deliverables: an HD demonstration video of 20+ agents navigating a cluttered forest,
a compiled Testing Report with quantitative metrics, and any final polish needed to
meet the proposal's M3 milestone targets before the Capstone Festival.

---

## Objectives

| ID | Objective | Success Criteria |
| :--- | :--- | :--- |
| P5.1 | HD demo video of 20+ agents in cluttered forest | Delivered ≥ 1080p, ≥ 30 s, showing formation keep + fault recovery + obstacle nav |
| P5.2 | Proposal objectives O1–O4 verified and documented | Testing Report finalized with pass/fail against each criterion |
| P5.3 | M3 milestone met | Formation error < 0.1 m steady-state; 0 collisions; recovery < 2.0 s |
| P5.4 | Presentation-ready repository | Clean README, reproducible run commands, no debug artifacts |

Aligns with proposal **Milestone M3 (Week 14, Apr 14)** and **M4 (Week 15, Apr 21):**
"Mission success — agents maintain formation through obstacle field with simulated
failures" and "HD showcase + Testing Report."

---

## Architecture Changes

Phase 5 introduces no new environment or policy code. All work is tooling,
configuration, and documentation:

### New Files

| File | Purpose |
| :--- | :--- |
| `drone_swarm_env_cfg_showcase.py` | Pre-built scenario configs for recording (`GGSwarmShowcaseCfg`, `GGSwarmClutteredForestCfg`, `GGSwarmUrbanCanyonCfg`) |
| `scripts/compile_testing_report.py` | Reads JSON/CSV from `eval_phase3.py` and `bench_scale.py`; writes a markdown Testing Report |

---

## 5A. Visual Environment

### Goal

Produce the "Cluttered Forest" USD stage referenced in the proposal as the primary
showcase environment. The scene must be visually compelling for the Capstone Festival
and demonstrate all four proposal objectives in a single recording session.

### Design

Isaac Sim's built-in asset library provides tree and rock props for the forest
setting. An urban canyon variant (grid-pattern building facades) is a secondary
deliverable.

**Cluttered Forest (`GGSwarmClutteredForestCfg`):**

```python
obstacle_count: int = 25
obstacle_radius: float = 0.15
obstacle_field_size: float = 3.0
cbf_obstacle_d_safe: float = 0.25
```

**Urban Canyon (`GGSwarmUrbanCanyonCfg`):**

```python
obstacle_count: int = 14         # grid of 7×2 tall columns
obstacle_radius: float = 0.20
obstacle_field_size: float = 3.0
cbf_obstacle_d_safe: float = 0.25
```

### RTX Rendering

Isaac Sim's RTX real-time renderer is used for the final recording:

- Enable path tracing via `sim.renderer.type = "RayTracedLighting"` in the USD stage
  settings before running `scripts/skrl/play.py`.
- Camera placement: overhead perspective at 45° elevation to show the full formation
  structure; follow-cam on the leader agent for fault-recovery sequences.
- Target frame rate: ≥ 30 fps during recording (reduce `num_envs` to 1 if needed).

---

## 5B. HD Demo Recording

### Scenarios to Record

| Sequence | Duration | Features Shown |
| :--- | :--- | :--- |
| Formation keeping, open space | 15 s | Baseline GNSC; formation error annotation overlay |
| Shape transition (circle → hexagon) | 10 s | 6-agent reconfiguration; smooth MINCO trajectories |
| Obstacle navigation, cluttered forest | 20 s | CBF activations visible; no collisions |
| Fault recovery | 15 s | Agent killed mid-flight; SwarmRaft redistribution; convergence timer overlay |
| 20+ agents, full scenario | 30 s | All features active simultaneously (O4) |

Total target runtime: **90 s** uncut; edit to ≤ 3 min for festival presentation.

### Recording Workflow

1. Pull the latest Phase 4 checkpoint from GCS:

```bash
python scripts/cloud/pull_results_from_gcs.py --family marl --latest 1
```

1. Launch Isaac Sim in GUI mode with RTX renderer:

```bash
python scripts/run.py phase5 play --checkpoint logs/skrl/ggswarm_marl/<run>/checkpoints/best_agent.pt
```

1. Enable Isaac Sim's built-in screen recorder (`Window → Extensions → omni.kit.capture.viewport`).
2. Run each scenario sequence; capture at 1920×1080.
3. Export to H.264 MP4 at 30 fps.

### Agent Count for O4

The proposal requires 20+ agents for the final video. Configuration:

```python
# In GGSwarmShowcaseCfg:
num_agents: int = 20
num_envs: int = 1       # single env for clean camera framing
```

At `num_envs=1` and `num_agents=20`, VRAM usage on an RTX 3070 (8 GB) must be
monitored. If VRAM is insufficient locally, record on the GCE L4 instance with
remote desktop streaming.

---

## 5C. Results Compilation

### Testing Report

`scripts/compile_testing_report.py` consumes the outputs of the Phase 4 evaluation
suite and produces `docs/testing_report.md` (the formal proposal deliverable):

| Section | Source |
| :--- | :--- |
| Executive Summary | Manually authored |
| Objective Table (O1–O4) | Pass/fail from `logs/phase4_eval_<ts>.json` |
| Scenario Metrics | Aggregated from `logs/phase4_eval_<ts>.json` |
| Scale Benchmark | `logs/bench_scale_<ts>.csv` |
| Comparison plots | `logs/phase4_eval_<ts>.json` (with/without CBF, MINCO) |

### Usage

```bash
# Generate report from latest eval and bench outputs
python scripts/compile_testing_report.py \
  --eval logs/phase4_eval_<timestamp>.json \
  --bench logs/bench_scale_<timestamp>.csv \
  --out docs/testing_report.md
```

### Required Comparison Plots

| Plot | Data Source | Purpose |
| :--- | :--- | :--- |
| Formation error: CBF on vs. off | Phase 4 eval JSON | Quantify O1 benefit |
| Velocity std: MINCO on vs. off | Phase 4 eval JSON | Quantify O2 benefit |
| Recovery latency CDF | Phase 4 eval JSON | Validate O3 threshold |
| Scale curve (agents vs. VRAM / steps-per-sec) | Bench CSV | O4 feasibility evidence |

---

## M3 Validation Checklist

Run before submitting the Testing Report:

- [ ] `python scripts/run.py phase3 eval --num_episodes 10` → formation error < 0.1 m
- [ ] `python scripts/run.py phase4 eval --scenario obstacle` → 0 collision events
- [ ] `python scripts/run.py phase4 eval --scenario kill` → gap-fill latency < 2.0 s
- [ ] `python scripts/run.py phase4 bench` → VRAM < 20 GB at 20 agents
- [ ] Demo video exists at ≥ 1080p showing 20+ agents in cluttered forest

---

## Implementation Plan

### Step 1: Scenario Configs (~0.5 days)

1. Finalize `drone_swarm_env_cfg_showcase.py` with cluttered forest and urban canyon params.
2. Register `GGS-Showcase-v0`, `GGS-ClutteredForest-v0`, `GGS-UrbanCanyon-v0`.
3. Add `python scripts/run.py phase5 play` command.

### Step 2: Report Compilation Script (~0.5 days)

1. Implement `scripts/compile_testing_report.py` to parse JSON + CSV and emit markdown.
2. Add `python scripts/run.py phase5 report` command.
3. Verify report renders correctly in GitHub markdown preview.

### Step 3: Recording (~1–2 days)

1. Configure RTX renderer and camera rigs.
2. Record all five scenario sequences listed in §5B.
3. Edit to final runtime; add annotation overlays for metric readouts.

### Step 4: M3 + M4 Validation (~0.5 days)

1. Run the M3/M4 validation checklist above — target completion by **Apr 21**.
2. Update `docs/status/weekly_updates.md` and `docs/status/changelog.md`.
3. Tag git commit `v0.5.0-phase5-complete`.

---

## Evaluation Procedure

Phase 5 does not introduce new evaluation logic. The M3 checklist reuses:

- `scripts/eval_phase3.py` for nominal + Phase 3 objective metrics
- `scripts/eval_phase3.py` with Phase 4 configs for agent-loss and obstacle metrics
- `scripts/bench_scale.py` for scale data

---

## Risks and Mitigations

| Risk | Mitigation |
| :--- | :--- |
| RTX rendering too slow for clean recording at 20 agents | Record on GCE L4 with remote display; or reduce to `num_envs=1`, disable shadows |
| Formation error > 0.1 m target not met | If Phase 4 shows 0.1–0.5 m residual error, initiate a short Phase 5 fine-tune run (20k steps); adjust reward scales |
| O4 (20+ agents) infeasible locally (VRAM) | Use GCE L4 for recording; results are equivalent to local execution |
| Video editing time underestimated | Keep raw captures; use Isaac Sim's built-in recorder; no external editing required for Capstone submission |

---

## Dependencies

- Phase 4 complete: 100-episode evaluation suite passed; `best_agent.pt` from final training run
- Isaac Sim GUI mode available (requires display; use X11 forwarding or Windows local)
- `matplotlib` or `plotly` for comparison plots (already in env_isaaclab)
- `ffmpeg` for video encoding (system dependency; pre-installed on GCE image)
