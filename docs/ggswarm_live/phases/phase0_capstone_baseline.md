# Phase 0: Capstone Baseline

**Status:** Complete. Shipped April 2026.

The capstone is the foundation everything else inherits from. See the
frozen [capstone tree](../../capstone/) for the full record.

## What it delivered

- Isolated envs (1 drone per physics scene), perfect state, centrally precomputed slots
- GATv2 + shared-policy PPO under SKRL
- MINCO trajectory filter, CBF shield, dynamic slot recompute on dropout
- Simulation only (Isaac Lab 2.x)
- Snapshot tagged `v1.0.0-capstone` on the `capstone` branch

## What it explicitly did not do

- No real hardware
- No shared-scene aerodynamics (no downwash coupling between drones)
- Centralized slot precomputation, no peer ranging, no GPS-denied operation
- No obstacle generalization beyond cylindrical forest
- No light-show pipeline integration

These are the entry points for Phases 1+.

## See Also

- [Vision](../vision.md)
- [Capstone README](../../capstone/README.md)
