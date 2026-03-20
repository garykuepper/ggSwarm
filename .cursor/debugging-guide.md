# Debugging Guide: ggSwarm Training

Systematic diagnosis and fixes for common training issues.

## Monitor Training
python scripts/run.py phase2 monitor   # Phase 2
python scripts/run.py hover monitor    # Hover

## Common Failures

### NaN Loss After 5 Steps
Fix: Cap rew_scale_ground_hit at -10 to -20
Check learning rate in [1e-5, 1e-3]

### Agents Crash Immediately  
Fix: rew_scale_pos greater_equal 1.0, rew_scale_alive greater 0.1

### Training Plateaus (20k+ steps)
Phase 2: Increase formation reward, tune curriculum
Hover: Increase network capacity, reduce learning rate

### Agents Hover But Do not Form
Check curriculum_start_step not too early
Verify formation geometry
Ensure GNN is used with --gnn flag

## Checkpoint Inspection
Load with torch.load() and inspect weight norms.

## Logs
Path: logs/skrl/ggswarm_marl/[timestamp]/
TensorBoard: tensorboard --logdir=.../summaries --port=6006

## Phase 2 Specific
Single-agent MARL: shared_states already injected in train.py
GNN over-smoothing: Limit to 2 layers, add dropout

See project-rules.mdc and debugging-guide.md for comprehensive troubleshooting.
