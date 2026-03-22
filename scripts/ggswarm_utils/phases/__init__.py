"""Phase-specific metric collectors for the unified eval pipeline.

Each sub-module exports a *Collector class that implements the
PhaseCollector protocol from ggswarm_utils.eval_runner.

Sub-modules:
    hover   -- HoverCollector  (task: GGS-Hover-v0)
    phase2  -- Phase2Collector (tasks: Phase 2A / 2B / 2C)
    phase3  -- Phase3Collector (tasks: Phase 3 and Phase 4)
"""
