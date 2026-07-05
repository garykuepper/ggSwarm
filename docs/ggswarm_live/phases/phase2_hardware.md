# Phase 2: Hardware Transfer

**Status:** Planned. Goal list, not a concrete plan — detail gets filled
in once Phase 1 (decentralization + downwash) lands and it's clear what
the policy actually needs from real hardware.

**New capability:** the sim-trained decentralized policy flies on real
drones.

## Goals

- Bring up real hardware capable of running the trained policy
  (companion computer + flight controller stack, TBD — see
  [`../archive/phases/phase10_singledrone.md`](../archive/phases/phase10_singledrone.md)
  for prior thinking on Pegasus Simulator vs. custom PX4 vs.
  Crazyswarm2, none of it committed)
- Validate the Phase 1 policy transfers: formation-keeping behavior
  holds up outside sim
- Confirm decentralization holds up on hardware: no anchors, no ground
  station in the control loop, peer-to-peer localization and
  assignment work with real radios and real noise, not just simulated
  noise models
- Fly a basic formation on real drones as the milestone artifact

## Deliberately not in scope here

Skybrush choreography integration, drone-show-specific architecture,
FAA Part 107 / § 107.35 regulatory work, and paid-show revenue goals
are **not** part of ggSwarm anymore — that work now lives in a
separate drone-light-show project with its own algorithm, decoupled
from this research program. (See prior design in
[`../archive/architecture.md`](../archive/architecture.md) and
[`../archive/phases/phase14_drone_show.md`](../archive/phases/phase14_drone_show.md)
if that separate project wants a starting reference.)

## See Also

- [Vision](../vision.md)
- [Phase 1: Sim](phase1_sim.md)
