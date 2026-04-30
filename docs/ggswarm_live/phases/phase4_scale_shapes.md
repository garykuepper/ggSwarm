# Phase 4: Scale + Expressive Shapes

**Status:** Planned.

**New capability:** formation library + size-agnostic policy.

## Scope

- Scale from 5 → 10+ drones
- Parametric shape generator (circle, polygon with arbitrary N, letters
  via glyph-to-points, arbitrary uploaded point clouds)
- Time-varying formations (morphing between shapes, rotating formations)
- Policy retrained with curriculum over shape-count and shape-type

## Backlog items folded into this phase

- **D1** Beyond 20 agents ([backlog](../backlog.md#d1))
- **B4** Heterogeneous agents — partial; only if curriculum motivates it ([backlog](../backlog.md#b4))

## Keeps simple

Still indoor, still Crazyflie, still no obstacles, still limited fault modes.

## Milestone artifact

Video reel of the swarm cycling through a dozen shapes including letters,
numbers, and a morph between two of them. Social: "Same policy. Different
shapes. No retraining."

## Risk hot-spots

- Assignment complexity grows with N (auction convergence time)
- Policy generalization to out-of-distribution shapes
- Crazyflie formation density limits (downwash dominates at close spacing)

## See Also

- [Vision § Phase 4](../vision.md)
