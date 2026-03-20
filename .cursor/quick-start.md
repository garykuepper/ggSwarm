# Quick Start: Common Agent Tasks

Quick reference for common ggSwarm development tasks.

## Task 1: Add a New Reward Term to Phase 2
- Add parameter to GGSwarmMarlEnvCfg with default 0.0
- Compute term in _compute_rewards() 
- Document in docs/status/changelog.md
- See Rule 5 in project-rules.mdc for example

## Task 2: Debug Why Agents Are not Hovering
- Monitor: python scripts/run.py hover monitor
- Check: rew_scale_pos greater_equal 1.0, rew_scale_alive greater 0.1, rew_scale_ground_hit less_equal -10.0
- Fix learning rate to 5e-5 if oscillating

## Task 3: Resume Interrupted Training
Find latest checkpoint then resume training from it.

## Task 4: Inspect a Checkpoint Policy
Load checkpoint and inspect weight norms - GNN weights should be greater 0.1.

## Task 5: Tune Curriculum Learning
Configure curriculum_start_step, curriculum_end_step, alpha parameters.

See .cursor/debugging-guide.md for comprehensive troubleshooting.
