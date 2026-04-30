# ggSwarm Capstone (v1) — Frozen

**Status:** Shipped 2026-04-24. This tree is frozen. Active development
has moved to [`../ggswarm_live/`](../ggswarm_live/).

The capstone delivered a simulation-only multi-agent RL drone swarm using
Isaac Lab, SKRL PPO, and a GATv2 graph attention policy. All artifacts
referenced below describe the state of the project at submission.

A snapshot of the code is tagged `v1.0.0-capstone` on the `capstone`
branch.

## Index

- [Proposal](project/proposal.md)
- [Testing Report](project/testing_report.md)
- [MINCO + CBF Research Notes](project/MINCO_CBF_Drone_Swarm_Research.md)
- [Concepts](concepts.md)
- Phases:
  - [Phase 1: Foundation](phases/phase1_foundation.md)
  - [Phase 2: Brain Development](phases/phase2_brain_development.md)
  - [Phase 3: Muscle Refinement](phases/phase3_muscle_refinement.md)
  - [Phase 4: Stress Testing](phases/phase4_stress_testing.md)
  - [Phase 5: Showcase Prep](phases/phase5_showcase_prep.md)
  - [Phase 6: Delivery](phases/phase6_delivery.md)
- Design: [architecture](design/architecture.md), [assumptions](design/assumptions.md), [tensor contracts](design/tensor_contracts.md)
- Status: [changelog](status/changelog.md), [run history](status/run_history.md), [weekly updates](status/weekly_updates.md)
- Ops: [commands](ops/commands.md), [training workflow](ops/training_workflow.md), [GCE results sync](ops/gce_results_sync.md), [post-train analysis](ops/post_train_analysis.md)

## Deferred items

The original Phase 7 ("Post-Capstone Plan") consolidated work that was
deliberately deferred from Phases 1–6. Those items have been folded into
the ggSwarm Live program backlog at
[`../ggswarm_live/backlog.md`](../ggswarm_live/backlog.md), where they
are mapped onto the ggSwarm Live phase plan.
