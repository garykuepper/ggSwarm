# Phase 3: Decentralized Assignment + Peer Ranging

**Status:** Planned.

**New capability:** no fixed infrastructure.

## Scope

1. **Peer UWB ranging.** Replace LPS anchors with drone-to-drone distance measurements only.
2. **Distributed state estimation.** Each drone estimates its position in the swarm-relative frame via multilateration over peer ranges + IMU fusion.
3. **Distributed slot assignment.** Auction algorithm (Bertsekas-style) over the peer mesh; replaces central slot precomputation.
4. **Gossip command dissemination.** Versioned command flooding; any-drone-hears → everyone-acts.
5. **Distributed centroid consensus.** Average consensus over IMU positions, used as shape anchor.

## Backlog items folded into this phase

- **E1** Semi-decentralized slot allocation ([backlog](../backlog.md#e1))

## Keeps simple

Still 3–5 Crazyflies, still indoor, still static shapes, still single-dropout only.

## Milestone artifact

Demo video of a Crazyflie swarm holding formation with all fixed anchors
removed and only peer ranging active. Social: "No anchors, no GPS, no
ground station. The drones localize each other."

## Risk hot-spots

- Peer-range localization too noisy for tight formation error → accept larger error or add lightweight visual fiducials
- Auction convergence under realistic comms loss → stale-aware bidding, eventual-consistency assignment
- Centroid drift in relative-frame operation

## See Also

- [Vision § Phase 3](../vision.md)
