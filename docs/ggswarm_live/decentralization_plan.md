# Decentralization Detail Plan — Peer-to-Peer Localization

This is the detailed execution plan for the localization slice of
[Phase 1 Goal A](phases/phase1_sim.md) (proper decentralization).
Authored 2026-07-04.

---

## 1. Problem statement

Today every drone's observation is built from ground-truth sim state.
In `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_marl_env.py`
(obs construction ~lines 387–449), neighbor relative positions,
`desired_pos_b`, and the GNN KNN `edge_index` all derive directly from
`self._robot.data.root_pos_w`. Slot assignment is likewise central:
formation offsets come from `formations.py` and are handed out via a
greedy nearest-slot assignment (`torch.cdist`) at reset. There is no
localization, no ranging, and no observation noise anywhere in the
code — every drone effectively knows its own and every peer's exact
position for free.

This plan closes the localization half of that gap: each drone
estimates its own position from peer ranging and onboard odometry
instead of reading it from the simulator.

## 2. SwarmRaft paper: verified purpose and what we take from it

The paper this repo's "SwarmRaft" mechanism is named after is real and
has been verified directly: Dev, Madhwal, Shevelo, Osinenko, Yanovich,
"SwarmRaft: Leveraging Consensus for Robust Drone Swarm Coordination in
GNSS-Degraded Environments," arXiv:2508.00622 (v2), *IEEE Internet of
Things Journal* 13(5):9112–9120, 2026, DOI 10.1109/JIOT.2025.3645453.
Its subject is consensus-based **position estimation** under GNSS loss:
fuse INS with peer-to-peer ranging, validate positional consistency via
a residual test, reconstruct a failed node's position by least-squares
multilateration, and replicate verdicts across the swarm via the Raft
consensus algorithm.

ggSwarm adopts the paper's estimation, residual-test, and
multilateration-recovery layers. It declines the paper's Raft
consensus transport: leader election reintroduces a single point of
failure over a lossy radio; a quorum cliff halts a small swarm on
partition or loss; consensus latency of tens to hundreds of
milliseconds cannot sit inside a ≥50 Hz estimation loop; and the
paper's adversarial-GNSS-spoofing threat model doesn't apply to a
cooperative swarm. The full adopt/decline rationale already lives in
[archive/consensus_mechanisms.md](archive/consensus_mechanisms.md) —
this doc links it rather than repeating it.

Separately, the capstone code's own "SwarmRaft" is a misnomer: it is a
dropout/alive-mask mechanism (cfg ~line 173, dropout logic in
`_pre_physics_step`) that kills a random drone mid-episode and
centrally reassigns slots. It has nothing to do with the paper. Stage 0
below renames it to `DropoutGuard`.

## 3. Design: odometry-anchored gauge

Pure peer ranging only fixes the swarm's geometry up to a global
translation, rotation, and reflection — the classic anchor-free
"gauge" problem. Without an anchor, the estimated formation can drift
or flip relative to the real one even while every pairwise distance is
correct. The design resolves this without reintroducing a central
anchor:

- **Seed.** Each drone's estimate `p̂_i` is seeded from the known
  takeoff layout at reset — standard drone-show practice, and a
  launch-time datum rather than a runtime anchor. This kills the
  reflection ambiguity from the start.
- **Propagate.** `p̂_i += (v_i + odom_noise) · dt`, i.e. simulated
  flow-deck/IMU velocity odometry.
- **Measure.** Simulated UWB pairwise ranges
  `d_ij = ‖x_i − x_j‖ + bias + N(0, σ)`, with per-link Bernoulli
  dropout and latency (ring buffer), calibrated to Crazyflie LPS/TWR
  literature.
- **Correct.** Each drone takes a few damped Gauss-Newton steps on its
  own estimate, minimizing
  `Σ_j w_ij (‖p̂_i − p̂_j^broadcast‖ − d_ij)²` against peers'
  latency-delayed broadcast estimates. This is genuinely distributed —
  no joint solve — and vectorizes as batched `[num_envs, A, A]` torch
  ops.
- **Fault detection.** Per-drone residual test, flagging when residual
  exceeds `μ + k·σ` (k = 3), with `μ`/`σ` calibrated offline by Monte
  Carlo against the trained policy's maneuver envelope, not against
  hover.
- **Recovery.** A flagged drone is re-multilaterated from its
  non-flagged peers (IRLS with Huber loss); fewer than 4 usable peers
  falls back to dead reckoning. No leader, no Raft — per-drone local
  verdicts, i.e. the paper's Stage-2 recovery without its Raft
  transport.

Gauge drift becomes a slow common-mode random walk driven by odometry
noise. Formation goals in the observation are computed relative to the
*estimated* swarm centroid, so common-mode error cancels there.
Rewards and dones keep using ground truth — legal privileged
information under CTDE.

### Implementation notes (2026-07-05)

Stages 0-4 shipped with three controller-adjudicated deviations from the
design above, each forced by a concrete failure observed during
implementation rather than a preference change:

1. **Correct against forward-predicted broadcasts**, not raw
   last-broadcast positions — the design as written above drags a
   moving swarm's translation toward stale broadcast positions. Peers
   now broadcast `(position, velocity)` and `correct()` targets
   `b_pred = p_broadcast + v_broadcast * dt`.
2. **Pre-fit innovation gating** — residuals are computed and tested
   *before* `correct()` runs (not after, as "Fault detection" implies by
   ordering), with median (not mean) aggregation across peer links and a
   hold-last-verdict rule on zero usable links.
3. **Recovery gates on odometry-jump consistency, not IRLS residual
   alone** — range residuals can't separate a genuine fault from an
   equally consistent wrong displaced position; `recover()` also rejects
   any candidate whose jump from the dead-reckoned estimate exceeds
   `recovery_jump_gate * threshold`.

Full rationale, the numbers that forced each deviation, and the pending
Isaac-side validation checklist are in the
[2026-07-05 changelog entry](status/changelog.md#2026-07-05--peer-to-peer-localization-stages-0-4-implemented-isaac-gates-pending).

## 4. Staged implementation

### Stage 0 — DropoutGuard rename — **complete**

Rename the "SwarmRaft" strings and comments in the active MARL path:
`ggswarm_marl_env.py`, `ggswarm_marl_env_cfg.py`, `scripts/skrl/play.py`,
`scripts/eval_metrics.py`. The legacy `ggswarm_env.py` is left untouched.

Gate: grep clean for "SwarmRaft" in the active path (done); 5-iteration
smoke train — **pending, Windows** (see changelog).

### Stage 1 — UWB ranging sim — **complete**

New `source/ggswarm/ggswarm/ranging.py` (`UwbRangingSim`): pure torch,
no Isaac imports, preallocated `[N, A, A]` buffers updated in place via
`.normal_()`/`.bernoulli_()` (per-step allocation ban means
`randn_like` is off the table — it allocates). New `tests/test_ranging.py`.

Gate: pytest green — noise mean ≈ bias, std ≈ σ, dropout rate matches
config, latency returns t−L values, symmetry holds, no allocations
after warmup. **Met** (7/7 tests).

### Stage 2 — Decentralized localizer — **complete**

New `source/ggswarm/ggswarm/localization.py` (`DecentralizedLocalizer`
with `propagate` / `correct` / `update_residuals` / `run_fault_test` /
`recover` / `rmse_vs` / `reset_idx`). Validated against a synthetic
8-drone scripted-trajectory pytest.

Gate: steady-state RMSE ≤ 0.10 m at default noise, gauge drift stays
within the odometry random-walk envelope, no reflection flips, and the
<4-peer fallback works. **Met in unit tests** (RMSE 0.073 m, gauge drift
0.040 m); see the three adjudicated deviations above.

### Stage 3 — Env integration, shadow mode — **code complete, Isaac validation pending**

A single `_update_localization()` at the top of `_get_observations`
(propagate → measure → correct → test → recover → log
`Metrics/loc_rmse_m`, `Metrics/loc_gauge_drift_m`). Observations still
come from ground truth, so no retraining is needed yet.
`reset_idx` clears the per-env latency ring-buffer slices;
DropoutGuard-dead drones are masked out of ranging.

Gate: `scripts/skrl/replay_gate.py` is bit-identical with
`loc_enabled=False`; a shadow replay of the existing checkpoint shows
unchanged formation metrics and loc RMSE ≤ 0.10 m; step time is stable.
**Pending, Windows** — see changelog checklist.

### Stage 4 — Residual test, recovery, fault injection (still shadow) — **code complete, Isaac validation pending**

New `scripts/calibrate_residual_threshold.py` (headless honest-noise
rollouts of the trained policy, printing `μ`/`σ` for cfg) and
`scripts/eval_localization.py` (RMSE and false-positive rate on honest
runs; false-negative rate and recovery-time CDF on fault-injected runs;
reusing `scripts/eval_metrics.py` helpers).

Gate: false-positive rate ≤ 0.01, false-negative rate ≤ 0.05, recovery
time ≤ 1.0 s. **Met in unit tests** (recovered faulted-drone error mean
0.053 m, max 0.101 m against the < 0.20 m unit-test gate); real-policy
numbers **pending, Windows** — see changelog checklist.

### Stage 5 — Observation swap + noise-anneal fine-tune

Switch to `loc_obs_source="estimate"`: KNN relative positions and GNN
edges from broadcast estimates, `desired_pos_b` from the drone's own
estimate, formation goal relative to the estimated centroid. Warm-start
from the MAPPO checkpoint; anneal the noise scale from
`loc_noise_scale_min` to `1.0` to avoid an observation-distribution
shock. If training struggles, the first knob to try is a privileged
`_get_states()` with ground-truth pose (legal under CTDE).

Gate: formation error within 2σ of the ground-truth baseline via
replay-gate comparison; the Stage 4 scorecard still holds with the
policy in the loop; zero formation collapses over 1k episodes; local
smoke test before any GCE launch.

### Stage 6 — Milestone artifact + docs

GUI demo with fault injection and visible recovery (debug overlay:
estimate-vs-truth whiskers, flagged drones shown in red). Record with
`--video --video_prefix p1loc-1`. Update the changelog, log, and
`phases/phase1_sim.md`; commit the checkpoint.

## 5. New cfg parameters

| Parameter | Default | Meaning |
| :--- | :--- | :--- |
| `loc_enabled` | `False` | Master switch for the localization pipeline |
| `loc_obs_source` | `"ground_truth"` | `"ground_truth"` \| `"estimate"` — obs pose source |
| `uwb_range_noise_std_m` | `0.10` | UWB range measurement noise std |
| `uwb_range_bias_m` | `0.05` | UWB range measurement bias |
| `uwb_link_dropout_prob` | `0.05` | Per-link Bernoulli dropout probability |
| `uwb_latency_steps` | `1` | Ranging measurement latency, in steps |
| `odom_vel_noise_std_mps` | `0.02` | Velocity odometry noise std |
| `loc_correct_iters` | `3` | Gauss-Newton correction iterations per step |
| `loc_gn_damping` | `0.5` | Gauss-Newton damping factor |
| `residual_mu` | (calibrated) | Residual-test mean, from offline calibration |
| `residual_sigma` | (calibrated) | Residual-test std, from offline calibration |
| `residual_k` | `3.0` | Residual-test threshold multiplier |
| `recovery_irls_iters` | `5` | IRLS iterations for multilateration recovery |
| `recovery_huber_delta` | `0.10` | Huber loss delta for recovery |
| `fault_inject_enabled` | — | Enable synthetic ranging-fault injection |
| `fault_bias_m` | `1.0` | Injected fault bias magnitude |
| `fault_step_min` | `200` | Earliest step a fault can be injected |
| `fault_step_max` | `350` | Latest step a fault can be injected |
| `fault_count` | `1` | Number of drones faulted per episode |
| `loc_noise_anneal_start` | `0` | Iteration noise annealing begins |
| `loc_noise_anneal_end` | `5000` | Iteration noise annealing reaches full scale |
| `loc_noise_scale_min` | `0.1` | Noise scale at the start of annealing |

Every tunable above lives in cfg, per the CLAUDE.md reward/hyperparameter
hygiene rule — none of it is a magic number in env core.

## 6. Scorecard

| Metric | Target |
| :--- | :--- |
| Position RMSE vs. ground truth (steady state) | ≤ 0.10 m |
| Residual-test false-positive rate (honest operation) | ≤ 0.01 |
| Residual-test false-negative rate (injected fault) | ≤ 0.05 |
| Recovery time after injected fault | ≤ 1.0 s |
| Formation collapses per 1k episodes | 0 |

Same targets as the archived [phase2a_localization.md](archive/phases/phase2a_localization.md)
scorecard — this plan supersedes its staging but not its numbers.

## 7. Hardware transferability (Phase 2 relevance)

Every adopted element maps 1:1 to real hardware: flow-deck/IMU odometry
plus UWB inter-drone two-way ranging needs no GNSS and no anchors, and
costs only tens of bytes per peer per cycle on a lossy mesh. The
residual test is purely local and catches real faults — UWB multipath,
flow-deck dropout, IMU drift, filter divergence — not just simulated
ones. Multilateration recovery is graceful degradation, not a hard
failure. The sim implementation is deliberately written to *be* the
hardware algorithm: per-drone Gauss-Newton correction against peer
broadcasts is companion-computer code, not a sim shortcut.

The Raft transport stays declined for flight-critical loops, for the
reasons in Section 2. Slow mission-level agreement — "start the show,"
"switch formation" — is better served by versioned gossip/CRDTs, and
the full SwarmRaft Raft layer is parked in the backlog with its own
revisit triggers.

## 8. Out of scope / sequenced after

- Distributed slot assignment (Bertsekas auction) — reset-time greedy
  assignment stays for now.
- Gossip/CRDT command propagation.
- Downwash (Goal B).
- Byzantine resilience.
- IMU-orientation noise.
- Hardware bring-up.

## 9. Risks

1. **Per-step allocation ban vs. RNG** — mitigated with preallocated
   buffers and in-place ops (`.normal_()`, `.bernoulli_()`), never
   `randn_like`/`torch.zeros` inside per-step code.
2. **Estimate-based KNN edges differ from true nearest neighbors under
   noise** — expected; the noise-anneal curriculum in Stage 5 absorbs
   it.
3. **Latency ring buffer vs. staggered resets** — `reset_idx` clears
   only the per-env slices that reset.
4. **Centralized-critic degradation** — fallback is a privileged
   `_get_states()` with ground-truth pose.
5. **Recovery ill-conditioning with few non-flagged peers** — IRLS
   damping plus a dead-reckon fallback below 4 peers.
6. **Residual false positives during aggressive maneuvers** —
   calibrate against policy rollouts, not hover.

## See Also

- [Phase 1: Sim](phases/phase1_sim.md)
- [Phase 2: Hardware](phases/phase2_hardware.md)
- [Archive: Phase 2a Localization](archive/phases/phase2a_localization.md)
- [Archive: Consensus mechanisms](archive/consensus_mechanisms.md)
- [Backlog](backlog.md)
