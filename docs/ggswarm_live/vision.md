# ggSwarm Live: Vision

*Living document, v0.3. Simplified 2026-07-03 from the original 18-phase
plan (preserved at [`archive/vision_v0.2.md`](archive/vision_v0.2.md)).
Solo pace, no timeline.*

## Vision statement

A **decentralized drone swarm** that forms formations on command with
no central controller, no fixed anchors, and no single point of
failure — and that keeps working when real aerodynamics (downwash,
wake turbulence) enter the picture, not just idealized sim physics.

The capstone (`v1.0.0-capstone`, frozen at [`../capstone/`](../capstone/))
proved the formation-control policy in an idealized, centralized sim.
ggSwarm Live's job is to remove those two simplifications — one at a
time, in sim, before ever touching hardware — and then get the result
flying on real drones.

## The plan

**Phase 1 — Sim.** Two goals, no fixed order between them:

- **Decentralization:** peer-to-peer localization, distributed slot
  assignment, multi-dropout fault tolerance, gossip-style command
  propagation — no central coordinator, no anchors.
- **Downwash physics:** real inter-drone aerodynamics in the training
  distribution, so the policy isn't learned against unrealistically
  clean physics.

See [`phases/phase1_sim.md`](phases/phase1_sim.md) for current status
(shared-scene MAPPO groundwork already built) and prior detailed design
in [`archive/`](archive/) for either goal, if useful.

**Phase 2 — Hardware.** A goal list, not a concrete plan: get the
Phase 1 policy flying on real drones and confirm the decentralization
holds up outside sim. Detail gets filled in once Phase 1 makes it clear
what hardware actually needs to support. See
[`phases/phase2_hardware.md`](phases/phase2_hardware.md).

That's the whole active plan. Capabilities from the old 18-phase plan
that aren't part of either goal (expressive shapes, animated
formations, obstacle avoidance, scale beyond 20 drones, onboard
distillation, multi-platform domain randomization, outdoor disturbance
DR) aren't scheduled — they're loose ideas in
[`backlog.md`](backlog.md), picked up only if they turn out to matter.

## Explicitly not part of this program

**The drone-light-show / revenue work is a separate project.** Earlier
planning assumed the ggSwarm decentralized-formation policy would
double as the execution layer under Skybrush choreography for paid
shows (Option B RL-overlay architecture, Part 107 / § 107.35
regulatory phases). That coupling didn't hold up — a drone light show
has its own, much simpler algorithmic needs (pre-authored choreography,
not learned formation control), and bolting the research policy onto a
revenue product distorted both. The show work is now its own project
with its own algorithm; it isn't documented here. The prior combined
design is preserved at [`archive/architecture.md`](archive/architecture.md)
and [`archive/phases/phase14_drone_show.md`](archive/phases/phase14_drone_show.md)
in case that separate project wants a starting reference.

## References

Academic references and simulator/hardware ecosystem notes live in
[`references.md`](references.md).
