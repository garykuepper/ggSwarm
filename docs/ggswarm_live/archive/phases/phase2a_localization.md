# Phase 2a: Decentralized Localization (sim only)

**Status:** Planned. Prerequisite for Phase 2c.

**New capability:** drones localize from peer UWB ranging only — ground-truth
pose is removed from the observation pipeline.

## Scope

Replace ground-truth pose with a simulated UWB peer-ranging stack and
distributed multilateration estimator. Adopt the SwarmRaft (paper)
fault-detection layer — but not its Raft transport — for honest sensor
faults. See [consensus_mechanisms.md](../consensus_mechanisms.md) for the
adopt / decline rationale.

## Inputs from prior phase

- Phase 1c shared-scene MAPPO + GATv2 checkpoint with edge-conditioned attention
- Existing `_get_observations` pipeline in `ggswarm_env.py` (the localization
  swap targets the *source* of pose; downstream obs construction is unchanged)

## Sim methodology

1. **Simulated UWB ranging.** Each pair `(i, j)` produces `d_{ij,k} = ‖x_i − x_j‖ + η`
   with calibrated noise model: range bias, Gaussian noise per Crazyflie LPS
   literature, dropout probability per link, latency window.
2. **Distributed multilateration estimator.** Each drone solves a local
   nonlinear least-squares over its received peer ranges, initialized from the
   centroid of currently-trusted peer position estimates.
3. **Residual test for fault detection.** Per-drone residual
   `e_{i,k} = min‖x − z_{i,k}‖` against the feasible region from neighbor
   ranges; threshold `μ + 3σ` calibrated offline under honest operation.
4. **Multilateration recovery.** When a drone's residual exceeds threshold,
   its position estimate is replaced by the multilateration solution over
   its non-flagged neighbors (least-squares with soft-L1 loss). This mirrors
   the SwarmRaft paper's Stage-2 recovery.
5. **No leader, no Raft.** Each drone runs the residual test on its own view
   of the gossiped peer state; we do not force a single global verdict per
   round. See `consensus_mechanisms.md` for why.

## Milestone artifact

8-drone sim demo holding formation under calibrated UWB noise with the
residual test and multilateration recovery active. Repo checkpoint of the
localization-swapped policy. Video recorded with `--video_prefix p2a-1`.

## Scorecard schema

| Metric | Target |
| :--- | :--- |
| Position RMSE vs. ground truth (steady state) | ≤ 0.10 m |
| Residual-test false-positive rate (honest operation) | ≤ 0.01 |
| Residual-test false-negative rate (injected ranging fault) | ≤ 0.05 |
| Recovery time after injected ranging fault | ≤ 1.0 s |
| Formation collapse rate (per 1k episodes) | 0 |

Calibrate `μ`, `σ`, and the noise model parameters via offline Monte Carlo
under honest operation before the residual threshold is locked.

## FAA evidence produced

Feeds **architecture.md §4 Layer 1 (RL fallback)** evidence column. The
residual-test FP/FN rates are the headline numbers for the safety case
section on sensor-fault detection; recovery-time CDF goes in the Layer 1
behavior table.

## Risks

- Calibrated UWB noise model under-represents real Crazyflie multipath →
  Phase 12 logs feed back to recalibrate, then 2a retrains.
- Multilateration ill-conditioned with too-few non-flagged peers → fall back
  to short-window IMU dead reckoning (matches paper's INS fallback note).
- False positives during transients (e.g., aggressive maneuvers) inflate the
  residual → calibrate threshold against the maneuver envelope, not just hover.

## Decline list

- **Raft transport over the verifier role** — declined; see
  `consensus_mechanisms.md`. Per-drone local verdict is sufficient for our
  cooperative threat model.
- **Byzantine-resilient ranging (n ≥ 2f+1 with adversarial bound)** — declined;
  cooperative peers, no spoofing budget.
- **Visual fiducials as a localization fallback** — deferred; would help if
  peer ranging plus IMU is insufficient on hardware, but not in scope for 2a.

## See Also

- [Phase 2 parent index](phase2_decentralized.md)
- [Phase 2c Fault Tolerance](phase2c_fault_tolerance.md) — multi-dropout extension
- [Consensus mechanisms reference](../consensus_mechanisms.md)
- [Vision § Phase 2a](../vision.md)
- SwarmRaft paper (Dev et al. 2025): residual-test + multilateration recovery
