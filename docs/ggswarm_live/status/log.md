# ggSwarm Live Log

Rolling notes. No fixed cadence (capstone weekly cadence retired with the
April 24 deadline). Use this for in-flight observations that don't yet
warrant a phase-doc or changelog entry.

## 2026-04-30

Program kickoff. Documentation reorganized. No code changes yet — Phase 1
work has not started.

## 2026-07-04

Verified the SwarmRaft paper (arXiv:2508.00622) directly: it is
consensus-based position estimation under GNSS loss, not dropout
recovery. The capstone code's "SwarmRaft" dropout mechanism is an
unrelated misnomer, to be renamed `DropoutGuard`. Authored
[decentralization_plan.md](../decentralization_plan.md), a detailed
staged plan for peer-to-peer localization. Full SwarmRaft (the paper's
Raft transport) parked in the backlog with revisit triggers. Docs only,
no code changes.

## 2026-07-05

Stages 0-4 of `decentralization_plan.md` implemented on branch
`phase1-localization`: `DropoutGuard` rename, `UwbRangingSim`,
`DecentralizedLocalizer`, shadow-mode env integration, and the
calibration/eval scripts. Pure-torch test suite is 17/17 green on Linux
CPU; three design deviations were adjudicated during implementation
(forward-predicted broadcasts, pre-fit innovation gating, odometry-jump
recovery gate). All Isaac-side gates (smoke train, replay gate,
calibration, scorecard) are deferred to the user's Windows machine — see
the changelog for the exact commands.
