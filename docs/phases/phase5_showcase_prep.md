# Phase 5: Showcase Prep

**Timeline:** Apr 14 -- Apr 21 (Week 15)  |  **Gate:** M4 -- HD showcase and Testing Report delivered

## 1. Goals

| ID | Goal | Success Criteria |
| :--- | :--- | :--- |
| P5.1 | HD demo video of 20+ agents | >= 1080p, >= 30 s, showing formation keeping + fault recovery + obstacle navigation |
| P5.2 | Proposal objectives O1--O4 verified | Testing Report finalized with pass/fail against each criterion |
| P5.3 | M3 milestone met | Formation error < 0.1 m steady-state; 0 collisions; recovery < 2.0 s |
| P5.4 | Presentation-ready repository | Clean README, reproducible run commands, no debug artifacts |

## 2. Tasks

No new environment or policy code. All work is tooling, configuration, and documentation.

**Visual environment setup** -- finalize `drone_swarm_env_cfg_showcase.py` with cluttered
forest and urban canyon scenario configs. Configure RTX rendering (path tracing via
`RayTracedLighting`), camera placement (overhead 45-degree + leader follow-cam), and
target >= 30 fps at `num_envs=1`.

**HD demo recording** -- record five scenario sequences using Isaac Sim's viewport
capture: formation keeping (15 s), shape transition (10 s), obstacle navigation (20 s),
fault recovery (15 s), 20+ agents full scenario (30 s). Edit to <= 3 min for festival.
Export H.264 MP4 at 1920x1080, 30 fps.

**Testing Report compilation** -- `scripts/compile_testing_report.py` consumes Phase 4 eval JSON and bench CSV outputs. Produces `docs/testing_report.md` with objective pass/fail table, scenario metrics, scale benchmark, and comparison plots (CBF on/off, MINCO on/off, recovery latency CDF, scale curve).

**M3/M4 validation checklist** -- run Phase 3 eval (formation error < 0.1 m), Phase 4 obstacle eval (0 collisions), Phase 4 kill eval (gap-fill < 2.0 s), bench (VRAM < 20 GB at 20 agents), confirm demo video exists.

## 3. Design Integration

Phase 5 introduces no architectural changes. It consumes the validated Phase 4 stack and packages results for presentation.

New files:

| File | Purpose |
| :--- | :--- |
| `drone_swarm_env_cfg_showcase.py` | Pre-built scenario configs for recording |
| `scripts/compile_testing_report.py` | Reads eval JSON + bench CSV, writes markdown Testing Report |

Commands: `python scripts/run.py phase5 play`, `python scripts/run.py phase5 report`.

Cross-references: `docs/phases/phase4_stress_testing.md`, `docs/design/architecture.md`.

## 4. Results

Phase 5 has not started.

---

## See Also

- `docs/phases/phase4_stress_testing.md` -- Phase 4: stress testing and evaluation suite
- `docs/design/architecture.md` -- GNSC 5-layer architecture
