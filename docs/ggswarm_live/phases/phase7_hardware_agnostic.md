# Phase 7: Hardware-Agnostic and General-Purpose (stretch)

**Status:** Stretch. Far horizon.

**New capability:** platform transfer + mission generalization.

## Scope

- Train with domain randomization over multiple quad platforms (mass,
  inertia, thrust-curve, size ranges)
- Demonstrate the same policy checkpoint deployed on 2+ real quad platforms
  without retraining
- Expand mission set: formation + area-coverage search + escort + navigation
- Task-conditioned policy or mission-parameter observation extension

## Milestone artifact

Split-screen video of the same policy checkpoint flying on two different
airframes. Social: "One brain. Different bodies. No retraining."

## Risk hot-spots

- Platform transfer doesn't generalize → frame as "family of platforms within
  randomization envelope," not true zero-shot

## See Also

- [Vision § Phase 7](../vision.md)
