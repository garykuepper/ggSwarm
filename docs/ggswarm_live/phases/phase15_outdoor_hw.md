# Phase 15: Outdoor + Extended Fault Tolerance Hardware (vision-level)

**Status:** Vision only. Detail to be fleshed out post-Phase 14.
Hardware spend gated by Phase 14 revenue.

**New capability:** outdoor flight at scale on real airframes capable of
real outdoor endurance and disturbance tolerance — beyond the Crazyflie
envelope used in Phases 10–14.

## High-level scope

- Move outdoors to a calm, open field as the production environment
  (Phase 14 already operated outdoors at the rehearsal / show scale; this
  phase opens the envelope further).
- **Likely hardware upgrade.** Crazyflies are practically indoor-only
  beyond the Phase 14 envelope; need Holybro X500 / ModalAI Seeker-class
  or similar with real outdoor endurance.
- Wind range expanded beyond Phase 6 sim DR envelope based on
  Phase 14b/14c real-world data.
- Multi-simultaneous dropout / comms link failure / sensor degradation /
  operator disconnect / battery-graceful-degradation drilled at scale on
  the new airframes.
- GPS still disabled; peer-ranging continues.

## Why deferred to post-show

Phases 6 and 9 (sim) close everything that can be closed in sim about
disturbance handling and cross-platform DR. Phase 14b/14c provide initial
outdoor + multi-drone real data on Crazyflie-scale airframes. Phase 15 is
the *post-show* expansion onto larger airframes — gated by the revenue
Phase 14 produces and by the calibration data Phases 14b/14c gather.

## See Also

- [Phase 6 Outdoor Disturbance DR (sim)](phase6_disturbance_dr.md)
- [Phase 9 Multi-Platform DR (sim)](phase9_multiplatform_dr.md)
- [Phase 14 First Drone Show](phase14_drone_show.md)
- [Vision § Phase 15](../vision.md)
