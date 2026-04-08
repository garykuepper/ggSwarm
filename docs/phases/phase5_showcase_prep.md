# Phase 5: Showcase Prep

**Timeline:** Apr 14 -- Apr 20  |  **Gate:** M4 -- HD showcase and Testing Report delivered

**Status:** Started 2026-04-07, **7 days early.** Phase 4 wrapped after the
forest-deflection rebuild produced clean obstacle navigation
(`p4-revert-4` checkpoint, 0 body penetrations through 0.20m-radius trunks).
Phase 5 begins by capturing HD video against the working forest scenario
before moving to the Tron-styled trailer.

**Production checkpoint:** [logs/skrl/ggswarm/p4/2026-04-06_21-09-24_ppo_torch/checkpoints/best_agent.pt](../../logs/skrl/ggswarm/p4/2026-04-06_21-09-24_ppo_torch/checkpoints/best_agent.pt)
(reward 66.83, ep_len 307.74, formation tracking + flock-aligned forest
navigation).

## 0. Tron Baseline — Setup Reference (commit `eb958dd0`)

**This section locks in the working Tron baseline so we don't have to
re-derive it next time we touch visuals.** It took two days, hours of
guess-and-check, and an Isaac Lab GitHub issue to get here. The baseline
is implemented entirely in [scripts/skrl/play.py](../../scripts/skrl/play.py)
behind `--tron`. Both `tron_env.py` and `showcase.py` are unchanged and
unused — the all-in-one `setup_tron_environment()` from `tron_env.py`
made debugging impossible (it does ~6 things at once: light removal,
render mode change, fog, grid, lighting, materials), so the rebuild
adds each Tron feature as a single inline step in `play.py`.

### Final pipeline (in order, all inside `if args_cli.tron:`)

| Iter | What | Why |
| :--- | :--- | :--- |
| **1** | Traverse stage, remove every `UsdLux` light by type name | Default lights painted the scene; the hardcoded path list in `tron_env._set_black_void` missed `/World/ground/terrain/SphereLight` |
| **4** | `sim_utils.make_uninstanceable` on each drone | Crazyflie is instanceable; without this, all `diffuseColor.Set()` calls below silently no-op (visuals frozen at the prototype) |
| **2** | Edit existing `DroneMat` shader's `diffuseColor`/`emissiveColor` in place to amber linear `(1.0, 0.262, 0.0)` | Reuses the env's existing material instead of fighting binding strength with a new one |
| **5** | Remove `/World/ground`, spawn 50m flat black quad via `sim_utils.spawn_preview_surface` | Terrain has a baked grid texture that overrides constant `diffuseColor`; replace with a flat invisible substrate |
| **6** | Spawn 102 thin quads (51 H + 51 V, 2cm × 50m, 2m spacing) at z=0.005 with bright cyan emissive material `(0.3, 3.0, 2.8)` linear | The lines ARE the visible grid — geometry can't be defeated by sampler caches |

The orbit camera is positioned per-frame via direct
`env.unwrapped.sim.set_camera_view(eye, target)` calls in the play loop
— **not** via `TronCameraRig` or `viewport.set_active_camera`, both of
which only updated the live Kit viewport instead of the env render
camera that `NvencRecorder` actually reads from. Z-up math (Isaac Lab
convention, not the Y-up that `TronCameraRig` assumed):

```text
angle += speed_deg
rad = radians(angle)
eye_x = centroid_x + radius * sin(rad)
eye_y = centroid_y + radius * cos(rad)
eye_z = centroid_z + height
target = (centroid_x, centroid_y, centroid_z)
```

Defaults: `radius=2.0m`, `height=0.6m`, `speed=0.6 deg/frame`. At 250
frames per clip that's 150° of arc — about 40% of a full orbit, enough
parallax for a cinematic shot.

### Things that look like they should work but don't

These are the traps the rebuild walked into. Documenting so we don't
re-walk them:

- **Setting `diffuseColor` on an instanceable Crazyflie shader.** Returns
  a valid `UsdShadeInput`, accepts the `Set()` call, looks like it works
  in the USD layer dump. Has zero visual effect because the prototype
  visuals are cached. Always `make_uninstanceable` first when targeting
  Isaac Lab assets that came from a USD reference.
- **Modifying the terrain's grid color.** The grid lines are baked into
  a tileable texture sampled by the terrain shader. The constant
  `diffuseColor` only tints the base; the texture re-paints the lines on
  top every frame. The "cyan flash" at the start of the iter 3 attempt
  was the constant briefly winning before the sampler kicked in.
- **`viewport.set_active_camera()` for video recording.** Updates the
  live Kit viewport but **NvencRecorder reads from `env.render()`**,
  which uses the env's internal viewer driven by
  `ViewportCameraController.update_view_location` →
  `sim.set_camera_view()`. Two separate camera pipelines.
- **`TronCameraRig` math from `tron_env.py`.** It's Y-up, Isaac Lab is
  Z-up. The rig's `_update_orbit` formula puts the camera at
  `y = target.y + height` which in Z-up is *horizontal offset*, not
  *up*. The drones end up small and offset in the frame.
- **Path tracing (`/rtx/rendermode = PathTracing`).** When you remove
  the dome light, path tracing loses its environment to sample from and
  paints the background red as a fallback. `RaytracedLighting` (real-time
  RT) respects `/rtx/backgroundColor*` settings and was the right choice.
  Even better: with the geometric `make_uninstanceable` + custom plane
  approach, render mode doesn't really matter — the visible surface is
  geometry, not a fallback color.
- **`spawn_preview_surface(diffuse_color=(1.0, 0.55, 0.0))` displaying
  as orange.** All Isaac Lab visual material color inputs are LINEAR RGB
  (per `IsaacLab/source/isaaclab/isaaclab/sim/spawners/materials/visual_materials.py`
  line 30 docstring). To display sRGB `#ff8c00 = (1.0, 0.549, 0.0)`,
  pass linear `(1.0, 0.262, 0.0)` (gamma decode `0.549^2.2 ≈ 0.262`).
  The amber drones in the current baseline still read as more yellow
  than amber in the final video — this is a known refinement target,
  not a known bug.

### Reference video

[`videos/showcase/p5-iter6-grid-tuned-episode-0.mp4`](../../videos/showcase/p5-iter6-grid-tuned-episode-0.mp4)
— canonical baseline clip. Black sky, black floor, bright cyan grid
(2m cells, 2cm lines), amber-ish drones in triangle formation,
orbit camera around the swarm centroid.

### Reproducing the baseline

```text
env_isaaclab/Scripts/python.exe scripts/skrl/play.py --task ggswarm-v0 \
  --num_agents 8 \
  --checkpoint logs/skrl/ggswarm/p4/2026-04-06_21-09-24_ppo_torch/checkpoints/best_agent.pt \
  --tron --formation triangle \
  --video --video_length 250 --play_length 250 \
  --prefix p5-baseline
```

### Tunable knobs (all in `play.py` `--tron` block)

| Variable | Current | What it controls |
| :--- | :--- | :--- |
| `tron_orbit["radius"]` | 2.0 | XY orbit radius around centroid (m) |
| `tron_orbit["height"]` | 0.6 | Z offset above centroid (m) |
| `tron_orbit["speed_deg"]` | 0.6 | degrees per frame (0.6 × 250 = 150° per clip) |
| `_AMBER` | `(1.0, 0.262, 0.0)` | Drone diffuse linear RGB (sRGB `#ff8c00`) |
| `_AMBER_BRIGHT` | `(1.5, 0.39, 0.0)` | Drone emissive linear RGB |
| `_size` (iter 5) | 50.0 | Half-extent of base plane and grid (m) |
| `_line_w` (iter 6) | 0.01 | Half-width of grid line quads (m) |
| `_spacing` (iter 6) | 2.0 | Distance between grid lines (m) |
| Lines emissive | `(0.3, 3.0, 2.8)` | Bright cyan glow on lines, linear RGB |

### Next iterations (planned, not yet built)

- **Drone color refinement** — currently more yellow than amber. Try
  bumping linear green even lower to compensate for renderer tone
  mapping, or try a magenta/red instead.
- **Cinematic 3-point lighting** in cyan tones for proper shading.
  Drones currently lit only by grid emission and look flat.
- **Volumetric fog** for atmosphere. Risky — caused white-outs in
  earlier tron_env attempts. Add carefully if needed.
- **Lift `TRON_AMBER`, `TRON_TEAL`, `TRON_LINE_WIDTH` etc. into
  `GgswarmEnvCfg`** as named constants per the user's earlier
  suggestion. Currently hardcoded in play.py.
- **Story scene clips** — formation morphing, dropout, scale-up, etc.
  Each should compose on top of the baseline (`--tron --formation
  grid`, `--tron --dropout`, etc.) without further visual changes.

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

### Implementation Plan (incremental, 4 phases)

Build the showcase incrementally — each phase independently testable.
Do NOT try to do everything at once.

```mermaid
flowchart LR
    A["Phase A<br/>--tron flag on play.py"] --> B["Phase B<br/>Fix Tron visuals"]
    B --> C["Phase C<br/>Formation morphing"]
    C --> D["Phase D<br/>Full cinematic"]

    style A fill:#3498db,color:#fff
    style B fill:#f39c12,color:#fff
    style C fill:#2ecc71,color:#fff
    style D fill:#8e44ad,color:#fff
```

**Phase A: `--tron` flag on play.py** (~30 lines)

Add Tron visuals to the existing working play.py. No new wrapper ordering,
no scene sequencing. Just call `setup_tron_environment()` after `gym.make()`
but before `NvencRecorder`, create a `TronCameraRig`, and step it each frame.

```powershell
python scripts/skrl/play.py --task ggswarm-v0 --checkpoint <path> `
    --tron --video --prefix p5-tron-test
```

**Phase B: Fix Tron visuals** (iterative)

Debug colors: verify `/World/Light` and `/World/ground` removal, test
emissive materials in RTX-Interactive vs Path Tracing mode, tweak fog
density and grid material. Only modify `scripts/tron_env.py`.

**Phase C: Formation morphing via `--showcase`** (~50 lines)

Add `--showcase` flag to play.py that enables scripted formation changes
at timed intervals. Reuse existing `get_formation()` + nearest-slot
assignment. No separate showcase.py needed.

```powershell
python scripts/skrl/play.py --task ggswarm-v0 --checkpoint <path> `
    --tron --showcase --video --prefix p5-showcase
```

**Phase D: Full cinematic** (drone kill + camera cuts)

Add SwarmRaft dropout trigger, camera mode switching (orbit → top_down →
low_angle), and optional scale-up. Either extend play.py `--showcase`
or keep in separate `scripts/showcase.py`.

### Files

| Phase | File | Changes |
| :--- | :--- | :--- |
| A | `scripts/skrl/play.py` | `--tron` flag, call setup_tron_environment, step TronCameraRig |
| B | `scripts/tron_env.py` | Fix light/ground removal, test materials |
| C | `scripts/skrl/play.py` | `--showcase` flag with formation timer |
| D | `scripts/showcase.py` or `play.py` | Drone kill, camera cuts |

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
