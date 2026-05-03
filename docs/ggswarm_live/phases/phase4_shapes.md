# Phase 4: Expressive Shape Library (sim only)

**Status:** Planned. Sim phase. Independent of Phases 3 and 5; can proceed
in parallel.

**New capability:** the policy generalizes across a parametric shape library
(N-gons, alphanumeric glyphs, arbitrary uploaded point clouds) without
retraining per shape.

## Scope

1. **Parametric shape generator.** Circle, polygon with arbitrary N, line,
   grid, alphanumeric glyphs via glyph-to-points, arbitrary uploaded point
   cloud as a CSV / JSON.
2. **Glyph-to-points pipeline.** Font → vector outline → uniform-density
   point sampling → slot-set. Documented as a reusable utility.
3. **Curriculum over shape-count and shape-type.** Train with shape sampled
   per episode from a growing library; held-out test set of shapes the
   policy never saw during training.
4. **Generalization measurement.** Formation error on training shapes vs.
   held-out shapes. The headline metric is the gap.
5. **Slot-spacing enforcement.** Parametric generator respects per-shape
   minimum spacing (informed by Phase 3's downwash-density limits).

## Inputs from prior phase

- Phase 2 sim-validated decentralized + fault-tolerant stack
- Phase 1c GATv2 policy (size-agnostic by construction; this phase tests the
  shape-agnostic property)
- Phase 3 downwash-density limits (per shape) — soft input

## Sim methodology

- Shape library defined as a JSON / Python registry with parametric
  constructors.
- Curriculum schedule: `n_shapes_in_distribution` grows over training; held
  out a fixed test set throughout.
- Evaluation harness reuses Phase 1a's replay-gate machinery for per-shape
  rollouts.

## Milestone artifact

Video reel of the swarm cycling through 12 shapes (mixed: 5 N-gons, 5
letters, 2 arbitrary point clouds), all from the same policy checkpoint.
Held-out-shape generalization plot. Video recorded with
`--video_prefix p4-1`.

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| Formation error on training-distribution shapes (median) | Within 1.10× Phase 2 baseline |
| Formation error on held-out shapes (median) | Within 1.50× training-distribution median |
| Settling time (command → formed) for held-out shape | Within 1.50× training distribution |
| Glyph-to-points reproducibility (same input → same slot set) | Bit-exact |
| Library shape-count at milestone | ≥ 12 distinct shapes including ≥ 5 letters |

## FAA evidence produced

Indirect. Establishes the shape envelope the safety case can claim
generalization within. The held-out-shape generalization measurement is the
honest answer to "does the policy work on shapes it has never seen?"

## Risks

- Held-out shapes underperform → expand training distribution; revisit
  curriculum schedule; if generalization gap remains > 2×, document the
  envelope honestly rather than overclaiming.
- Glyph-to-points produces uneven density that breaks slot assignment →
  enforce uniform-density sampler; reject glyphs whose minimum spacing
  falls below the Phase 3-documented limit.
- Arbitrary point-cloud upload is a security / safety hole if not
  constrained → require pre-flight validation that no two points are below
  the minimum spacing and the cloud fits within the operating envelope.

## Decline list

- **Time-varying shapes in Phase 4** — declined; that is Phase 5.
- **Outdoor in Phase 4** — declined; Phase 6 covers outdoor disturbance DR
  in sim, Phase 15 covers real outdoor.
- **3D shapes (out-of-plane structure) in Phase 4** — declined for now;
  current parametric generator targets formations the hardware envelope can
  fly; revisit as hardware envelope grows.

## See Also

- [Phase 3 Scale](phase3_scale.md) — density limits feed shape generator
- [Phase 5 Animated Formations](phase5_animated.md)
- [Phase 14 First Drone Show](phase14_drone_show.md) — consumes the shape library
- [Vision § Phase 4](../vision.md)
