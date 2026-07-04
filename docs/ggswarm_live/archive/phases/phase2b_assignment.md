# Phase 2b: Decentralized Slot Assignment (sim only)

**Status:** Planned. Can run in parallel with Phase 2a once the localization
interface is defined.

**New capability:** slot assignment without a centralized planner — the env's
greedy-nearest `cdist` call is replaced.

## Scope

Two-step curriculum: backlog item E1 (slot-preference logits) lands first as a
learning-friendly stepping stone; full Bertsekas auction over the peer mesh
then replaces E1 as the production path. E1 is not retained as a permanent
layer — it exists to ease the policy's transition from "env hands me a slot"
to "I bid on a slot."

## Inputs from prior phase

- Phase 1c GATv2 + MAPPO checkpoint (slot-preference logits will be a new
  policy output head)
- Existing greedy-nearest assignment harness in `ggswarm_env.py:367-387` —
  reused as the env-side tie-break / conflict resolver during step 1

## Sim methodology

### Step 1 — backlog E1: slot-preference logits

1. Add per-slot preference logits as a new policy output head (one logit per
   formation slot, masked by current valid-assignment set).
2. Env collects every drone's preferences and resolves conflicts using
   preferences as tie-breakers in the same greedy-nearest harness
   (`ggswarm_env.py:367-387`).
3. Reward shaping rewards consistent preference (no thrashing) and assignment
   stability across steps.

This is *partial* decentralization: drones express intent, env still
executes. It is the curriculum step.

### Step 2 — Bertsekas auction over peer mesh

1. Each drone bids on its preferred slot weighted by cost (e.g., distance
   from current pose).
2. Bids propagate via the gossip channel (Phase 2d's versioned gossip primitive).
3. A drone outbid for its current slot picks the next-best slot.
4. The system converges to a valid assignment in O(N) rounds under healthy
   comms; longer under loss. Stale-aware bidding tolerates one-round-old
   views without producing collisions.

This is *full* decentralization: the env does not assign anything. Once the
auction is stable end-to-end, the E1 head can be removed (or the head's
logits can feed initial bid weights, deferred).

## Milestone artifact

**Step 1 milestone:** sim demo showing 8 drones with policy-driven slot
preferences, env-side tie-break, no instability or thrashing. Checkpoint
tagged for ablation evidence (the recommended `v2.0.0-phase2b-E1` checkpoint
in the plan's open-decisions list).

**Step 2 milestone:** sim demo showing 8 drones converging to a valid
permutation with no `cdist` call inside the env step. Video recorded with
`--video_prefix p2b-2`.

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| Assignment latency (steady state, healthy comms) | ≤ 5 sim steps |
| Assignment latency (under 30% link loss) | ≤ 50 sim steps |
| Conflict rate (two drones picking same slot in resolved state) | 0 |
| Regret vs. centralized greedy-nearest (steady-state formation error) | ≤ 1.10× |
| Assignment thrashing rate (changes per drone per minute) | ≤ 0.5 |

## FAA evidence produced

Feeds **architecture.md §4 Layer 2 (offboard timeout)** evidence column —
specifically the "swarm continues coordinated behavior without a central
planner" property that the safety case rests on. Conflict rate is the
headline number; auction convergence under loss is the behavior table.

## Risks

- Auction does not converge in degraded comms → stale-aware bidding,
  eventual-consistency guarantees, fallback to last-stable assignment.
- E1 stepping stone destabilizes existing checkpoint → keep Phase 1c
  checkpoint frozen, train the preference head as a new output without
  perturbing the policy core (curriculum learning rate schedule).
- Auction scales poorly to N=20+ → revisit with hierarchical clustering or
  consensus-based bundle algorithm in Phase 4.

## Decline list

- **Raft / blockchain consensus for slot assignment** — declined; see
  `consensus_mechanisms.md`. Auction is the right specialization.
- **Hungarian / centralized optimal assignment** — declined as production
  path; reintroduces a single point of failure and does not match the
  decentralized goal. Retained only as a benchmark in the regret metric.
- **Permanent E1 layer alongside auction** — declined per user's stepping-stone
  preference; E1 ships as a curriculum aid, not as a production layer.

## See Also

- [Phase 2 parent index](phase2_decentralized.md)
- [Phase 2d Consensus + Dissemination](phase2d_consensus_dissemination.md) — provides the gossip channel auction bids ride on
- [Backlog item E1](../backlog.md#e1-semi-decentralized-slot-allocation)
- [Consensus mechanisms reference](../consensus_mechanisms.md)
- [Vision § Phase 2b](../vision.md)
