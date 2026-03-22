"""Shared utilities for ggSwarm training scripts.

This package centralises helpers that were previously duplicated across
eval_phase2.py, eval_phase3.py, post_train_assess.py, analyze_checkpoints.py,
bench_scale.py and skrl/play.py.

Sub-modules:
    eval_stats   -- EvalStats dataclass and per-step metric helpers
    checkpoint   -- Policy loading, GNN injection, checkpoint discovery
    scorecard    -- Pass/warn/fail thresholds, verdict logic, report generation
    gcs_sync     -- GCS -> local sync helpers (gcloud storage cp, Windows-safe)
    cli          -- Command-building helpers and lock management for run.py
    sim_helpers  -- Shared Isaac Lab simulation helpers (PHASE_REGISTRY,
                    configure_gnn_policy, override_agent_count, extract_actions,
                    configure_eval_cfg)
    eval_runner  -- PhaseCollector protocol and run_eval() shared eval boilerplate
    phases/      -- Phase-specific metric collectors (hover, phase2, phase3)
"""
