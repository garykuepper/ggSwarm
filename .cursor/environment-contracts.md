# Environment Contracts: ggSwarm

Complete specification of all environment I/O to prevent shape bugs.

## Observation Contract
Shape: [num_envs, num_agents, 12] (float32)
Components: lin_vel_xyz, ang_vel_xyz, gravity_xyz, rel_pos_goal_xyz

## Action Contract  
Shape: [num_envs, num_agents, 4] (float32)
Components: thrust, moment_x/y/z

## Extras Dictionary
adj_matrix: [num_envs, num_agents, num_agents] (bool)
Diagonal must be all zeros (no self-edges)
Controlled by cfg.graph_threshold_m

## Common Contract Violations
- Shape mismatch: Verify observation has all 12 components
- Adjacency diagonal not zero: Zero it explicitly
- NaN loss: Check rew_scale_ground_hit not too negative
- Policy ignores adjacency: Convert to COO format for GNN

See debugging-guide.md for complete details.
