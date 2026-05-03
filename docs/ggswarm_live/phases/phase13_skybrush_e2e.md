# Phase 13: Skybrush End-to-End Integration

**Status:** Planned. Soft-depends on Phase 12 (integration work can begin in
parallel; milestone capture requires Phase 12 stable).

**New capability:** the Option B architecture
([architecture.md § 2.2](../architecture.md#22-option-b-rl-waypoint-overlay-selected))
runs end-to-end on real hardware: Skybrush Studio CSV → companion-computer
trajectory loader → RL overlay (Phase 1c+ policy) → PX4 / CRTP setpoint.

## Scope

1. **Skybrush Studio choreography ingestion.** CSV ZIP loader; per-drone
   CSV parser; trajectory interpolation; lookahead buffer.
2. **RL overlay live.** Policy outputs position offset from reference
   waypoint; offset clamped to configured radius; Layer 1 fallback to raw
   waypoint when offset exceeds threshold.
3. **End-to-end show flight.** A 60-second Skybrush-authored short piece
   flies cleanly with the RL overlay active.
4. **Layer 1 fallback exercised.** Forced-large-RL-offset test triggers the
   bypass; system reverts to raw waypoint without disruption.
5. **Show-shaped artifact.** The first artifact that *looks like a drone
   show*, even if minimal — sets up Phase 14 (First Drone Show).

## Inputs from prior phase

- Phase 12 stable decentralized stack on hardware
- Phase 12 retrained policy with calibrated DR

## Hardware methodology

- Same indoor volume as Phases 10–12.
- Skybrush Studio installation + Blender addon documented; CSV export
  pipeline scripted, not manual.
- 60-second piece authored deliberately for sim-friendly motions (slow
  morph, hover, slow morph back) — *not* a polished show, just a
  validation artifact.
- RL overlay enabled; baseline comparison against Layer-1-bypass-only
  (raw-waypoint-following) recorded.

## Milestone artifact

Video of a 60-second Skybrush-authored piece flying cleanly with the RL
overlay active. Side-by-side comparison: same piece with RL overlay vs.
raw waypoint following. Repo release tagged `v2.0.0-sim2real`. Video
recorded with `--video_prefix p13-1`.

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| End-to-end show completion rate (10 attempts) | ≥ 9 / 10 |
| RL overlay offset (steady state, well-tracked piece) | ≤ 0.30 m |
| Layer 1 fallback trigger rate during nominal flight | ≤ 0.01 per minute |
| Tracking error vs. Skybrush reference (RL overlay on) | ≤ 1.10× tracking error (RL overlay off) |
| Failsafe layer 1 trigger latency under forced large offset | ≤ 100 ms |

## FAA evidence produced

Feeds **architecture.md § 4 Layer 1 (RL fallback)** evidence column with
real-hardware measurements (Phase 2a provided sim-only FP/FN numbers;
Phase 13 provides the hardware trigger latency and nominal-flight
false-trigger rate).

## Risks

- Skybrush CSV pipeline brittle (encoding / units / time alignment) →
  scripted importer with unit tests; do not author CSVs by hand.
- RL overlay introduces tracking artifacts that hurt show quality → tighten
  offset clamp; or accept the Layer-1-bypass-only mode for show ops while
  RL training catches up.
- Authoring a 60-second piece reveals choreography ergonomics gaps →
  documented; carries forward to Phase 14.

## Decline list

- **Outdoor flying in Phase 13** — declined; Phase 14b solo outdoor
  content / Phase 15 outdoor hardware.
- **LED / color choreography in Phase 13** — declined; Phase 14.
- **Music sync in Phase 13** — declined; Phase 14.

## See Also

- [Phase 12 Anchor Removal + Decentralized Stack](phase12_decentralized_hw.md)
- [Phase 14 First Drone Show](phase14_drone_show.md)
- [Architecture § 2.2 Option B selected](../architecture.md#22-option-b-rl-waypoint-overlay-selected)
- [Architecture § 3 Selected architecture](../architecture.md#3-selected-architecture-option-b-on-px4)
- [Vision § Phase 13](../vision.md)
