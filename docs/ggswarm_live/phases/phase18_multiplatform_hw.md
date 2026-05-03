# Phase 18: Multi-Platform Hardware (vision-level, stretch)

**Status:** Vision only. Stretch. Detail to be fleshed out post-Phase 14
and only if Phases 15–17 produce a stable production envelope to extend.

**New capability:** the same policy checkpoint deployed to ≥ 2 different
real quad airframes without retraining. This is the "hardware-agnostic
policy" requirement from `vision.md` § 3 (NFR-4).

## High-level scope

- Same policy checkpoint flown on 2+ real quad platforms (Crazyflie 2.1
  + Holybro X500-class at minimum).
- Mission set expanded: formation + area-coverage search + escort +
  navigation; task-conditioned policy or mission-parameter observation
  extension.
- Validates the Phase 9 sim cross-platform DR claim on real hardware.

## Why deferred to post-show (and stretch even then)

Phase 9 (sim) closes the cross-platform DR algorithm. Phase 18 is the
hardware validation — high cost (multiple real airframes, multiple
maintenance pipelines), and the value is mainly *external*
(generalization claim for publication / portfolio) rather than load-bearing
for the Phase 14 revenue. Sequenced as stretch because it is not on the
critical path of the program; revisit when post-show priorities are
re-evaluated.

## See Also

- [Phase 9 Multi-Platform DR (sim)](phase9_multiplatform_dr.md)
- [Phase 14 First Drone Show](phase14_drone_show.md)
- [Vision § Phase 18](../vision.md)
