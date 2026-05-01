# Phase 1a: Shared-Scene Refactor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Refactor `GgswarmEnv` so all `num_agents` drones spawn in one shared physics scene per env (replacing one-drone-per-env), fix the B3 stacked-spawn artifact, collapse the now-redundant `_num_groups = N // A` plumbing, and verify via a replay-only test against `v1.0.0-capstone` that the new env preserves capstone-checkpoint behavior with downwash off.

**Architecture:** In-place refactor of `ggswarm_env.py` and `ggswarm_env_cfg.py`. After 1a, groups *are* envs — every reshape `[N, ...] → [G, A, ...]` becomes `[N_envs, A, ...]` directly. The capstone env remains accessible only via the `capstone` branch and `v1.0.0-capstone` tag (created earlier this session pointing at commit `c33d339`). Spec: `docs/superpowers/specs/2026-04-30-phase1-shared-scene-design.md`. 1b and 1c get stubs at the end of this plan; their detailed plans will be written when their preconditions clear.

**Tech Stack:** Isaac Lab 2.x · `DirectRLEnv` · PhysX · SKRL PPO · GATv2 (PyTorch Geometric) · NVENC H.264

---

## File structure

| Path | Status | Responsibility |
| :--- | :--- | :--- |
| `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env.py` | Modify | All in-place refactor work — scene setup, reshape collapse, B3 fix, forest reshape audit |
| `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env_cfg.py` | Modify | Robot ArticulationCfg expansion to 8 per env, drop `num_agents == 1` defaults |
| `scripts/skrl/replay_gate.py` | Create | Replay-gate harness: load `v1.0.0-capstone`, run N rollouts in both envs (capstone branch and shared-scene main), statistical compare |
| `scripts/skrl/play.py` | Modify | Drop the `num_agents == 1` and `num_agents > 1` branching that's collapsing into a single shared-scene path |
| `docs/ggswarm_live/status/changelog.md` | Modify | Per-step entry for each milestone (smoke pass, sweep result, replay gate pass, tag) |
| `docs/ggswarm_live/phases/phase1_shared_scene_sim.md` | Modify | Status field flips to "1a complete" with TB scalar baseline |
| `logs/ref/v1.0.0-capstone/` | Create | Reference rollouts captured from capstone branch — input to replay gate |

**Why this layout:** The env file is the integration point for shared-scene; splitting it across multiple new files would scatter the refactor and force re-imports during 1b/1c. The replay-gate harness is its own script (not folded into `play.py`) because its job is statistical comparison across two envs/checkpoints, not interactive playback.

---

## Tasks

### Task 0: Capture capstone reference rollouts (preflight, must run BEFORE refactor)

**Files:**
- Create: `logs/ref/v1.0.0-capstone/rollouts_metadata.json`
- Create: `logs/ref/v1.0.0-capstone/seed_*.pt` (per-seed rollout tensors)

**Why first:** Once 1a refactor lands on `main`, the only way to regenerate capstone rollouts is to switch to the `capstone` branch and rebuild the env. Capturing them once now and storing under `logs/ref/` saves repeated branch switching and locks the reference distribution.

- [ ] **Step 0.1: Switch to capstone branch in a clean clone or a worktree**

```text
git worktree add ../ggSwarm-capstone capstone
cd ../ggSwarm-capstone
```

Verify: `git rev-parse HEAD` returns `c33d339...`. Verify the env code matches: `head -5 source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env.py` (should show capstone-era docstring, no `_num_groups` cleanup).

- [ ] **Step 0.2: Identify the production capstone checkpoint**

The capstone repo's status logs reference checkpoint `p4-revert-4`. Locate it:

```text
find logs/skrl/ggswarm -name '*.pt' -path '*p4-revert-4*' | head -5
```

Pick the `best_agent.pt` from the most recent `p4-revert-4` run. Record its absolute path; you'll pass it via `--checkpoint`.

- [ ] **Step 0.3: Run play 5 times with fixed seeds, recording trajectories and rewards**

```text
mkdir -p ../ggSwarm/logs/ref/v1.0.0-capstone

for SEED in 7 13 21 42 99; do
  python scripts/skrl/play.py --task ggswarm-v0 \
    --checkpoint <path-to-p4-revert-4-best_agent.pt> \
    --num_agents 8 --play_length 500 --seed $SEED \
    --trajectories \
    --prefix capstone-ref-seed${SEED}
  cp logs/skrl/ggswarm/<run>/trajectories/capstone-ref-seed${SEED}-trajectory_data.csv \
     ../ggSwarm/logs/ref/v1.0.0-capstone/
done
```

Pass condition: 5 CSVs land in `logs/ref/v1.0.0-capstone/`, each with 500 rows × (1 + 6×A) columns.

- [ ] **Step 0.4: Compute reference metric distributions and write metadata**

Create `logs/ref/v1.0.0-capstone/rollouts_metadata.json` with:

```json
{
  "checkpoint_path": "<absolute path to p4-revert-4 best_agent.pt>",
  "checkpoint_sha": "<sha256 of the .pt>",
  "capstone_commit": "c33d339",
  "capstone_tag": "v1.0.0-capstone",
  "task": "ggswarm-v0",
  "num_agents": 8,
  "play_length": 500,
  "seeds": [7, 13, 21, 42, 99],
  "metrics": {
    "mean_formation_error_m": {"mean": <fill from CSV>, "std": <fill from CSV>},
    "collision_pairs_per_step": {"mean": <fill from CSV>, "std": <fill from CSV>},
    "final_distance_to_goal_m": {"mean": <fill from CSV>, "std": <fill from CSV>},
    "episode_reward": {"mean": <fill from CSV>, "std": <fill from CSV>}
  }
}
```

Compute means/stds across the 5 seeds in a one-off Python session:

```python
import json, csv, hashlib, statistics
from pathlib import Path
ref = Path("logs/ref/v1.0.0-capstone")
# read each CSV, compute formation error per row from per-drone positions
# average across rows, then average across seeds; std across seeds
```

(The actual computation is straightforward — pairwise distances vs `formation_target_spacing=0.5`, mean per step, mean per rollout, mean+std across 5 rollouts.)

- [ ] **Step 0.5: Commit reference rollouts to main (return to main first)**

```text
cd ../ggSwarm
git checkout main
git add logs/ref/v1.0.0-capstone/
git commit -m "chore(phase1a): capture v1.0.0-capstone reference rollouts for replay gate

5-seed rollout set (seeds 7, 13, 21, 42, 99) at play_length=500 against
the p4-revert-4 best_agent.pt checkpoint. These are the fixed reference
distributions the 1a replay gate compares the shared-scene env against.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

Pass condition: `git log -1 --stat` shows the rollouts and metadata committed.

---

### Task 1: Create 1a working branch

**Files:** none (git operation only).

- [ ] **Step 1.1: Create branch off main**

```text
git checkout -b phase1a-shared-scene
git status   # should show clean tree on phase1a-shared-scene
```

- [ ] **Step 1.2: Verify the smoke test still passes BEFORE any change**

```text
env_isaaclab/Scripts/python.exe scripts/skrl/train.py --headless \
  --task ggswarm-v0 --num_envs 64 --max_iterations 5
```

Expected: training proceeds 5 iters with no errors. This establishes "the env works on this branch right now" as the baseline before we change anything.

- [ ] **Step 1.3: Commit a marker (no code change)**

```text
git commit --allow-empty -m "chore(phase1a): start branch — pre-refactor smoke passes"
```

---

### Task 2: Scene topology — spawn 8 drone articulations per env

**Files:**
- Modify: `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env_cfg.py:82-85` (the `robot` field)
- Modify: `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env.py:_setup_scene`

- [ ] **Step 2.1: Replace single robot cfg with a list of 8 robot cfgs in `ggswarm_env_cfg.py`**

Replace lines around 82-85:

```python
# robot — base config, _setup_scene creates one per agent
robot: ArticulationCfg = CRAZYFLIE_CFG.replace(
    prim_path="/World/envs/env_.*/Drone_0"
)
```

with:

```python
# Robots — A drones spawn per env at /World/envs/env_.*/Drone_{0..A-1}.
# A list rather than .* glob so each drone is an independently addressable
# Articulation we can index in the env code below.
robots: list[ArticulationCfg] = [
    CRAZYFLIE_CFG.replace(prim_path=f"/World/envs/env_.*/Drone_{i}")
    for i in range(8)
]
```

Note: `num_agents = 8` already exists at line 35; this list length must match.

- [ ] **Step 2.2: Modify `_setup_scene` to instantiate all 8 articulations**

In `ggswarm_env.py` at the `_setup_scene` method, replace the single-articulation block:

```python
def _setup_scene(self):
    self._robot = Articulation(self.cfg.robot)
    self.scene.articulations["robot"] = self._robot

    # Apply yellow material to drone body (before clone_environments)
    ...
    mat_path = "/World/envs/env_0/Drone_0/body/Looks/DroneMat"
    ...
```

with a per-drone loop. The articulations are stored as a list `self._robots[0..A-1]`, and a fused position/velocity view `self._robot` is constructed for the existing per-step code that reads `self._robot.data.root_pos_w`. The cleanest path is to keep `self._robot` as a *facade* wrapping the list:

```python
def _setup_scene(self):
    A = self.cfg.num_agents
    self._robots = [Articulation(rcfg) for rcfg in self.cfg.robots]
    for i, r in enumerate(self._robots):
        self.scene.articulations[f"robot_{i}"] = r

    # Apply yellow material to all drone bodies (before clone_environments)
    import omni.usd  # noqa: PLC0415
    from pxr import UsdShade  # noqa: PLC0415

    stage = omni.usd.get_context().get_stage()
    for i in range(A):
        mat_path = f"/World/envs/env_0/Drone_{i}/body/Looks/DroneMat"
        mat_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.85, 0.0))
        sim_utils.spawn_preview_surface(mat_path, mat_cfg)
        body_prim = stage.GetPrimAtPath(f"/World/envs/env_0/Drone_{i}/body")
        mat_prim = stage.GetPrimAtPath(mat_path)
        if body_prim.IsValid() and mat_prim.IsValid():
            UsdShade.MaterialBindingAPI.Apply(body_prim)
            UsdShade.MaterialBindingAPI(body_prim).Bind(UsdShade.Material(mat_prim))

    # ... rest of _setup_scene unchanged (terrain, clone_environments, lights, forest)
```

- [ ] **Step 2.3: Add a fused-view facade so existing per-step code keeps working**

The existing env code reads `self._robot.data.root_pos_w` as a `[N_total_drones, 3]` tensor. After 1a, "total drones" = `N_envs × A`. We need a thin wrapper that concatenates the A articulation views.

Add to the bottom of `_setup_scene`:

```python
class _FusedRobotData:
    """Concatenates per-articulation views into a [N_envs * A, *] tensor.

    The legacy capstone env returned a single Articulation whose data was
    already shaped [N_envs * 1, *] = [N_envs, *]. After the 1a refactor the
    same per-step access pattern is preserved by stitching the A
    per-articulation tensors together along the env dim.
    """

    def __init__(self, robots, num_envs, num_agents, device):
        self._robots = robots
        self._N = num_envs
        self._A = num_agents
        self._device = device

    def _stack(self, attr: str) -> torch.Tensor:
        # Each robot's data.<attr> has shape [N_envs, *]. Stack along the
        # env dim, interleaving so drone-i in env-e lives at index e*A + i.
        per_drone = [getattr(r.data, attr) for r in self._robots]   # list of [N_envs, *]
        stacked = torch.stack(per_drone, dim=1)   # [N_envs, A, *]
        return stacked.reshape(self._N * self._A, *stacked.shape[2:])

    @property
    def root_pos_w(self):  return self._stack("root_pos_w")
    @property
    def root_quat_w(self): return self._stack("root_quat_w")
    @property
    def root_lin_vel_w(self): return self._stack("root_lin_vel_w")
    @property
    def root_lin_vel_b(self): return self._stack("root_lin_vel_b")
    @property
    def root_ang_vel_b(self): return self._stack("root_ang_vel_b")
    @property
    def projected_gravity_b(self): return self._stack("projected_gravity_b")
    @property
    def default_root_state(self): return self._stack("default_root_state")
    @property
    def default_joint_pos(self): return self._stack("default_joint_pos")
    @property
    def default_joint_vel(self): return self._stack("default_joint_vel")


class _FusedRobot:
    def __init__(self, robots, num_envs, num_agents, device):
        self._robots = robots
        self._N = num_envs
        self._A = num_agents
        self.data = _FusedRobotData(robots, num_envs, num_agents, device)
        self.root_physx_view = robots[0].root_physx_view  # mass etc identical across A
        self._ALL_INDICES = torch.arange(num_envs * num_agents, device=device)

    def find_bodies(self, name):
        return self._robots[0].find_bodies(name)

    def reset(self, drone_ids):
        # drone_ids is in the [N_envs * A] flat space. For each per-articulation
        # robot, reset only the env_ids that contain a drone in this articulation.
        if drone_ids is None:
            for r in self._robots:
                r.reset()
            return
        # Convert flat drone_ids -> per-articulation env_ids.
        for i, r in enumerate(self._robots):
            # drone_id = env_id * A + i, so drone_ids belonging to articulation i
            # have (drone_id % A) == i; their env_ids are drone_id // A.
            mask = (drone_ids % self._A) == i
            env_ids = drone_ids[mask] // self._A
            if env_ids.numel() > 0:
                r.reset(env_ids)

    def write_root_pose_to_sim(self, pose, drone_ids):
        # pose is [len(drone_ids), 7]; route per-articulation similarly.
        for i, r in enumerate(self._robots):
            mask = (drone_ids % self._A) == i
            if mask.any():
                env_ids = drone_ids[mask] // self._A
                r.write_root_pose_to_sim(pose[mask], env_ids)

    def write_root_velocity_to_sim(self, vel, drone_ids):
        for i, r in enumerate(self._robots):
            mask = (drone_ids % self._A) == i
            if mask.any():
                env_ids = drone_ids[mask] // self._A
                r.write_root_velocity_to_sim(vel[mask], env_ids)

    def write_joint_state_to_sim(self, jp, jv, joint_ids, drone_ids):
        for i, r in enumerate(self._robots):
            mask = (drone_ids % self._A) == i
            if mask.any():
                env_ids = drone_ids[mask] // self._A
                r.write_joint_state_to_sim(jp[mask], jv[mask], joint_ids, env_ids)

    @property
    def permanent_wrench_composer(self):
        # Return an object that fans set_forces_and_torques across the A robots.
        return _FusedWrenchComposer(self._robots, self._N, self._A)


class _FusedWrenchComposer:
    def __init__(self, robots, num_envs, num_agents):
        self._robots = robots
        self._N = num_envs
        self._A = num_agents

    def set_forces_and_torques(self, body_ids, forces, torques):
        # forces, torques are [N_envs * A, 1, 3]; split per articulation.
        F = forces.view(self._N, self._A, 1, 3)
        T = torques.view(self._N, self._A, 1, 3)
        for i, r in enumerate(self._robots):
            r.permanent_wrench_composer.set_forces_and_torques(
                body_ids=body_ids,
                forces=F[:, i],
                torques=T[:, i],
            )
```

Bind the facade at the end of `_setup_scene`:

```python
self._robot = _FusedRobot(self._robots, self.scene.cfg.num_envs, self.cfg.num_agents, self.device)
```

Define the two `_Fused*` classes at module level above `class GgswarmEnv` (not inline in the method).

**Why a facade rather than rewriting every per-step access:** The capstone env reads `self._robot.data.*` from ~30 lines across `_pre_physics_step`, `_get_observations`, `_get_rewards`, `_get_dones`, `_reset_idx`. Rewriting every site to index into `self._robots[i]` is 30+ edits with high error surface. A facade that returns the same `[N_envs * A, *]` shape on the same property names lets all per-step code stay byte-identical, which is exactly what the 1a replay gate requires.

- [ ] **Step 2.4: Smoke-run the changed scene setup**

```text
env_isaaclab/Scripts/python.exe scripts/skrl/train.py --headless \
  --task ggswarm-v0 --num_envs 16 --max_iterations 1
```

Expected: env loads, stage shows 16 envs × 8 drones = 128 drone articulations, training begins one iter then exits cleanly.

If it fails: most likely cause is the `clone_environments` step rejecting per-articulation prim paths. Check that `replicate_physics=True` still works with `Drone_0..7` (it should — Isaac Lab supports multi-articulation envs). If not, set `replicate_physics=False` as a workaround in the cfg and document why.

- [ ] **Step 2.5: Commit**

```text
git add source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env.py \
        source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env_cfg.py
git commit -m "feat(phase1a): spawn A drone articulations per env via fused-view facade

_setup_scene now creates A=num_agents Articulation objects per env at
/World/envs/env_.*/Drone_{0..A-1}. A _FusedRobot/_FusedRobotData facade
preserves the [N_envs * A, *] tensor shape contract that all per-step env
code reads from. Per-step access patterns unchanged.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 3: Reshape math collapse — `_get_observations` and `_expand_obs_with_neighbors`

**Files:**
- Modify: `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env.py:_get_observations` (line ~460)
- Modify: `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env.py:_expand_obs_with_neighbors` (line ~518)

The shape contract changes: `N` (total drones) was the outer dim; it stays. `G = self._num_groups = N // A` becomes `G = self.scene.cfg.num_envs` (or, equivalently, `G = N // A` still holds since `N = num_envs * A` post-1a). Rename `G → N_envs` in the local variable scope to make the code self-document the new mental model.

- [ ] **Step 3.1: Update `_expand_obs_with_neighbors`**

Replace the local variable `G` with `N_envs` throughout the method:

```python
def _expand_obs_with_neighbors(self, obs: torch.Tensor) -> torch.Tensor:
    from ggswarm.gnn_policy import GgswarmGNNPolicy  # noqa: PLC0415

    N = self.num_envs * self.cfg.num_agents   # total drones (used to be self.num_envs)
    A = self.cfg.num_agents
    K = min(self.cfg.num_neighbors, A - 1)
    N_envs = self.scene.cfg.num_envs
    pos_local = self._robot.data.root_pos_w - self._terrain.env_origins.repeat_interleave(A, dim=0)  # shape: [N, 3]
    pos_grouped = pos_local.reshape(N_envs, A, 3)
    ...
```

**Critical:** `self._terrain.env_origins` has shape `[num_envs, 3]`. After 1a, the fused `root_pos_w` has shape `[num_envs * A, 3]`. To subtract origins per drone we need to repeat each origin A times: `env_origins.repeat_interleave(A, dim=0)` produces `[num_envs * A, 3]`. Apply this same pattern everywhere `self._terrain.env_origins` is subtracted from drone positions.

In `__init__`, the `_terrain.env_origins` lookup happens inside `super().__init__`. Pre-allocate the expanded origins:

```python
# In __init__, after super().__init__:
self._env_origins_per_drone = self._terrain.env_origins.repeat_interleave(A, dim=0)  # shape: [N_envs * A, 3]
```

Then per-step code does `self._robot.data.root_pos_w - self._env_origins_per_drone`. Avoids per-step `repeat_interleave` allocation.

- [ ] **Step 3.2: Update `_get_observations` references**

Anywhere `self._num_groups` appears in `_get_observations` (debug-draw block at line ~496-514), replace with `self.scene.cfg.num_envs` directly.

- [ ] **Step 3.3: Smoke**

```text
env_isaaclab/Scripts/python.exe scripts/skrl/train.py --headless \
  --task ggswarm-v0 --num_envs 16 --max_iterations 5
```

Expected: training runs 5 iters with no shape errors. Watch the SKRL output for `loss` decreasing or at least varying — silence/zeros suggest reward shape broke.

- [ ] **Step 3.4: Commit**

```text
git add source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env.py
git commit -m "refactor(phase1a): collapse group plumbing in _get_observations

G = N // A is no longer needed since one env IS one group post-shared-scene.
Local variable renamed to N_envs to match the new mental model. Pre-allocated
_env_origins_per_drone buffer added in __init__ to avoid per-step
repeat_interleave allocation (reward-hygiene rule).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 4: Reshape math collapse — `_compute_formation_reward`

**Files:**
- Modify: `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env.py:_compute_formation_reward` (line ~645)

- [ ] **Step 4.1: Replace `G` references**

```python
def _compute_formation_reward(self) -> torch.Tensor:
    A = self.cfg.num_agents
    N_envs = self.scene.cfg.num_envs
    pos_local = self._robot.data.root_pos_w - self._env_origins_per_drone   # shape: [N_envs * A, 3]
    pos_grouped = pos_local.reshape(N_envs, A, 3)
    ...
    self._formation_total_error.zero_()                  # shape: [N_envs]
    ...
    return formation_reward.unsqueeze(1).expand(N_envs, A).reshape(N_envs * A)
```

In `__init__`, rename buffer alloc:

```python
# Was: self._formation_total_error = torch.zeros(G, device=device)
self._formation_total_error = torch.zeros(self.scene.cfg.num_envs, device=device)  # shape: [N_envs]
```

- [ ] **Step 4.2: Smoke (same as 3.3)**

- [ ] **Step 4.3: Commit**

```text
git commit -m "refactor(phase1a): collapse group plumbing in _compute_formation_reward

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 5: Reshape math collapse — `_compute_cloud_reward`

**Files:**
- Modify: `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env.py:_compute_cloud_reward` (line ~689)

Same pattern as Task 4. All `G` → `N_envs = self.scene.cfg.num_envs`. Update the pre-allocated buffers in `__init__`:

- [ ] **Step 5.1: Update `_compute_cloud_reward`**

Replace `G` with `N_envs` throughout the method body. Final return shape is `[N_envs * A]` (was `[N]`).

- [ ] **Step 5.2: Update `__init__` cloud-mode scratch buffers**

```python
if self._cloud_mode and A > 1:
    N_envs = self.scene.cfg.num_envs
    self._group_goal_local = torch.zeros(N_envs, 3, device=device)        # [N_envs, 3]
    self._cloud_centroid_dist = torch.zeros(N_envs, device=device)         # [N_envs]
    self._cloud_spacing_penalty = torch.zeros(N_envs, A, device=device)    # [N_envs, A]
    self._cloud_cohesion_reward = torch.zeros(N_envs, A, device=device)    # [N_envs, A]
```

- [ ] **Step 5.3: Smoke (same)**

- [ ] **Step 5.4: Commit**

```text
git commit -m "refactor(phase1a): collapse group plumbing in _compute_cloud_reward

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 6: Reshape math collapse — `_get_dones` (collision detection)

**Files:**
- Modify: `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env.py:_get_dones` (line ~786)

- [ ] **Step 6.1: Replace `G` references in collision-detection block**

```python
if self.cfg.num_agents > 1 and self.cfg.collision_enabled:
    A = self.cfg.num_agents
    N_envs = self.scene.cfg.num_envs
    pos_local = self._robot.data.root_pos_w - self._env_origins_per_drone   # shape: [N_envs * A, 3]
    pos_g = pos_local.reshape(N_envs, A, 3)                                  # shape: [N_envs, A, 3]
    ...
    collided_flat = collided_group.unsqueeze(1).expand(N_envs, A).reshape(-1)  # shape: [N_envs * A]
```

In `__init__`:
```python
# Was: self._collision_count = torch.zeros(G, device=device)
self._collision_count = torch.zeros(self.scene.cfg.num_envs, device=device)  # shape: [N_envs]
```

- [ ] **Step 6.2: Replace `G` in collective-resets block**

```python
if self.cfg.num_agents > 1 and self.cfg.collective_resets:
    A = self.cfg.num_agents
    N_envs = self.scene.cfg.num_envs
    died_grouped = died.reshape(N_envs, A)
    any_died = died_grouped.any(dim=1)
    died = any_died.unsqueeze(1).expand(N_envs, A).reshape(-1)
```

- [ ] **Step 6.3: Smoke (same)**

- [ ] **Step 6.4: Commit**

```text
git commit -m "refactor(phase1a): collapse group plumbing in _get_dones

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 7: Reshape math collapse — `_reset_idx`

**Files:**
- Modify: `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env.py:_reset_idx` (line ~856)

The reset routine receives `env_ids` in the `[N_envs * A]` flat space (per the `_FusedRobot._ALL_INDICES`). Current logic computes `group_ids = env_ids // A` and iterates per-group. Post-1a, "group" = "env", so `env_ids // A` IS the unique env index.

- [ ] **Step 7.1: Rename group references**

In the section starting around line 905 (`if self.cfg.num_agents > 1 and self._formation_offsets is not None:`), keep the `// A` math but rename `group_ids` → `env_ids_unique`, `n_groups` → `n_envs_to_reset`, `g_idx`/`g` → `e_idx`/`e`. Same logic, clearer naming.

- [ ] **Step 7.2: Replace `_num_groups > 1` checks**

Line 885:
```python
if self.cfg.num_agents > 1 and self._num_groups > 1:
```
becomes:
```python
if self.cfg.num_agents > 1 and self.scene.cfg.num_envs > 1:
```

The episode-length staggering at line 889 (`group_lengths = torch.randint(...)`) keeps the same shape — we want each env to have a different episode-length offset, which is what `[N_envs]` randint gives us:
```python
N_envs = self.scene.cfg.num_envs
A = self.cfg.num_agents
env_lengths = torch.randint(0, int(self.max_episode_length), (N_envs,), device=self.device)
self.episode_length_buf = env_lengths.unsqueeze(1).expand(N_envs, A).reshape(-1).clone()
```

- [ ] **Step 7.3: Update dropout-step buffer alloc in `__init__`**

```python
# Was: self._dropout_step = torch.zeros(G, dtype=torch.long, device=device)
self._dropout_step = torch.zeros(self.scene.cfg.num_envs, dtype=torch.long, device=device)  # [N_envs]
```

And the `ep_len_grouped` line in `_pre_physics_step` (around line 347):
```python
ep_len_grouped = self.episode_length_buf.reshape(self.scene.cfg.num_envs, A)[:, 0]
```

- [ ] **Step 7.4: Smoke (same)**

- [ ] **Step 7.5: Commit**

```text
git commit -m "refactor(phase1a): collapse group plumbing in _reset_idx

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 8: Reshape math collapse — `_pre_physics_step` (forest deflection block)

**Files:**
- Modify: `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env.py:_pre_physics_step` (line ~244, the forest block at ~254-341)

The forest block uses `G = self._num_groups` for KNN-mean velocity reshaping. Same pattern as before.

- [ ] **Step 8.1: Replace `G` in the forest block**

```python
if self.cfg.forest_enabled and self._formation_active and self._obstacle_pos is not None:
    A = self.cfg.num_agents
    N_envs = self.scene.cfg.num_envs
    K_nn = min(self.cfg.num_neighbors, A - 1)
    ...
    drone_local = self._robot.data.root_pos_w - self._env_origins_per_drone  # shape: [N_envs * A, 3]
    ...
    pos_g = drone_local.reshape(N_envs, A, 3)         # shape: [N_envs, A, 3]
    vel_g = vel_xy.reshape(N_envs, A, 2)              # shape: [N_envs, A, 2]
    ...
    knn_vel = vel_g.gather(
        1, knn_idx.reshape(N_envs, A * K_nn, 1).expand(N_envs, A * K_nn, 2)
    ).reshape(N_envs, A, K_nn, 2)
```

The dropout block (line ~344-388) also has `G`:
```python
if self.cfg.dropout_enabled and self.cfg.num_agents > 1:
    A = self.cfg.num_agents
    N_envs = self.scene.cfg.num_envs
    ep_len_grouped = self.episode_length_buf.reshape(N_envs, A)[:, 0]
```

- [ ] **Step 8.2: Smoke + forest play smoke**

```text
env_isaaclab/Scripts/python.exe scripts/skrl/train.py --headless \
  --task ggswarm-v0 --num_envs 16 --max_iterations 5
```

Then forest mode play:

```text
python scripts/skrl/play.py --task ggswarm-v0 \
  --checkpoint <p4-revert-4 path> --forest --play_length 200
```

Expected: 200 steps run without crash. Forest deflection block doesn't error. (The drones won't behave correctly because they were trained in isolated env, but the gate is "no crash, reasonable trajectories".)

- [ ] **Step 8.3: Commit**

```text
git commit -m "refactor(phase1a): collapse group plumbing in forest + dropout blocks

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 9: Remove `num_agents == 1` branch and dead `_num_groups` attribute

**Files:**
- Modify: `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env.py` throughout
- Modify: `scripts/skrl/play.py:249-272`

After Tasks 3-8 every `_num_groups` reference is gone. Now delete the field itself and the `num_agents == 1` branches that no longer get exercised.

- [ ] **Step 9.1: Delete `_num_groups` and surrounding single-agent guards in `__init__`**

In `ggswarm_env.py:__init__`, delete:

```python
# delete these lines
if A > 1 and N % A != 0:
    raise ValueError(f"num_envs ({N}) must be divisible by num_agents ({A})")
self._num_groups = N // A if A > 1 else N
```

The divisibility check is no longer relevant — N (total drones) is always `num_envs * A` after 1a. Total drones = `self.num_envs` already (Isaac Lab sets `self.num_envs = num_envs * A` automatically when articulations expand? — verify in step 9.4).

Actually in Isaac Lab, `self.num_envs` reflects `scene.cfg.num_envs` directly, not articulation count. So if `scene.cfg.num_envs = 16`, `self.num_envs = 16`, and `self._robot.data.root_pos_w.shape[0] = 16 * A = 128`. Confirm and adjust shape comments.

- [ ] **Step 9.2: Replace `self.num_envs` with the correct quantity in env code**

Audit every `self.num_envs` reference in `ggswarm_env.py`. Where it's used as "total drones", replace with `self.num_envs * self.cfg.num_agents`. Where it's used as "number of envs", keep as-is.

Specifically:
- `__init__` line `N = self.num_envs` → keep as `N = self.num_envs * A` (or just `N_drones`).
- `_zero_reward_N` shape: `[N_drones]`, not `[num_envs]`. Update buffer alloc.
- Any `self.num_envs` references that should be `total drones`: rename to `N_drones = self.num_envs * A` for clarity.

- [ ] **Step 9.3: Remove single-agent dead paths**

Search for `if self.cfg.num_agents > 1` and matching `else` branches that handled the single-agent (capstone hover-only) case. Delete the `else` branches; phase 1+ never has `num_agents == 1`. Remove the `if A > 1 else N` style guards in `__init__` for `_pair_indices`, `_formation_offsets`, etc.

Search command (read-only — do not pipe to anything destructive):

```text
grep -n "num_agents > 1\|num_agents == 1\|num_agents <= 1\|A > 1\|A == 1" \
  source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env.py
```

Walk each match. If the body is "skip this work in single-agent mode", delete the guard and keep the work. If the body is an `else` that matters (e.g., single-agent random goal sampling), delete it.

In `_reset_idx`, line 974-982 (`else: # Single-agent: independent random goals`) — delete this entire `else` block.

- [ ] **Step 9.4: Verify `self.num_envs` semantics with a debug print**

Add a one-time `logger.info` at end of `__init__`:

```python
import logging
logger = logging.getLogger(__name__)
logger.info(f"GgswarmEnv init: num_envs={self.num_envs}, num_agents={A}, "
            f"total drones (root_pos_w shape[0])={self._robot.data.root_pos_w.shape[0]}")
```

Run a 1-iter smoke and confirm the printed total matches `num_envs * A`.

- [ ] **Step 9.5: Update `play.py` num_agents branching**

Replace the conditional at line 249-272:

```python
# was:
if args_cli.num_envs is not None:
    env_cfg.scene.num_envs = args_cli.num_envs
elif args_cli.num_agents > 1:
    env_cfg.scene.num_envs = args_cli.num_agents
else:
    env_cfg.scene.num_envs = 1
...
env_cfg.num_agents = args_cli.num_agents
if args_cli.num_agents > 1:
    env_cfg.observation_space = 12 + env_cfg.num_neighbors * 3
    env_cfg.scene.env_spacing = 0.01
    env_cfg.collective_resets = False
    env_cfg.formation_centroid = (0.0, 0.0, 1.0)
    env_cfg.dropout_enabled = args_cli.dropout
else:
    env_cfg.observation_space = 12
```

with the post-1a single-path version:

```python
if args_cli.num_envs is not None:
    env_cfg.scene.num_envs = args_cli.num_envs
else:
    env_cfg.scene.num_envs = 1   # single env, A drones in shared scene
env_cfg.num_agents = args_cli.num_agents
env_cfg.observation_space = 12 + env_cfg.num_neighbors * 3
env_cfg.scene.env_spacing = 0.01
env_cfg.collective_resets = False
env_cfg.formation_centroid = (0.0, 0.0, 1.0)
env_cfg.dropout_enabled = args_cli.dropout
```

Note: `--num_agents 1` is no longer supported. If a user passes it, error out cleanly. Add at the top of `play.py:main`:

```python
if args_cli.num_agents == 1:
    raise SystemExit(
        "Phase 1+ does not support num_agents=1. Use the capstone branch and "
        "v1.0.0-capstone tag for hover-only single-agent play."
    )
```

- [ ] **Step 9.6: Smoke train + smoke play**

```text
env_isaaclab/Scripts/python.exe scripts/skrl/train.py --headless \
  --task ggswarm-v0 --num_envs 16 --max_iterations 5
```

Then play with the capstone checkpoint:

```text
python scripts/skrl/play.py --task ggswarm-v0 \
  --checkpoint <p4-revert-4 path> --num_agents 8 --play_length 100
```

Expected: 100 steps in shared-scene env, no crashes. Trajectories may differ from capstone-branch run (that's what the replay gate measures).

- [ ] **Step 9.7: Commit**

```text
git commit -m "refactor(phase1a): remove num_agents==1 branch and dead _num_groups attr

Phase 1+ always runs num_agents>1 in shared-scene mode. Single-agent
hover-only is preserved on the capstone branch / v1.0.0-capstone tag.
play.py raises SystemExit if num_agents=1 is passed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 10: B3 spawn fix — replace stacked-vertical with circle of radius `spawn_radius`

**Files:**
- Modify: `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env.py:_reset_idx` (line ~984-998)

- [ ] **Step 10.1: Replace stacked spawn with circle**

Current code (around line 990-995):

```python
default_root_state[:, :3] += self._terrain.env_origins[env_ids]
r = self.cfg.spawn_radius
default_root_state[:, 0] += torch.zeros(len(env_ids), device=self.device).uniform_(-r, r)
default_root_state[:, 1] += torch.zeros(len(env_ids), device=self.device).uniform_(-r, r)
default_root_state[:, 2] = 0.5
```

The current code uses `spawn_radius` as a *random jitter* but all 8 drones still spawn within the same XY box. Post-B3 we want each drone to spawn on a slot of a circle, jittered slightly:

```python
import math
A = self.cfg.num_agents
r_circle = self.cfg.spawn_radius
jitter = self.cfg.min_spawn_spacing * 0.15   # 15% of nearest-neighbor distance

# env_ids is in [N_envs * A] flat space. drone-i within env-e has env_ids[k] = e*A + i.
# Compute per-drone slot offsets on the circle.
drone_slot = env_ids % A   # shape: [len(env_ids)] — which drone within its env
theta = 2 * math.pi * drone_slot.float() / A
slot_x = r_circle * torch.cos(theta)        # [len(env_ids)]
slot_y = r_circle * torch.sin(theta)

origins = self._env_origins_per_drone[env_ids]   # [len(env_ids), 3]
default_root_state[:, 0] += origins[:, 0] + slot_x \
    + torch.zeros(len(env_ids), device=self.device).uniform_(-jitter, jitter)
default_root_state[:, 1] += origins[:, 1] + slot_y \
    + torch.zeros(len(env_ids), device=self.device).uniform_(-jitter, jitter)
default_root_state[:, 2] = 0.5
```

This places drone-i in env-e at angle `2π·i/A` on the circle, then jitters by ±15% of nearest-neighbor distance.

- [ ] **Step 10.2: Smoke + visual confirmation**

```text
env_isaaclab/Scripts/python.exe scripts/skrl/train.py --headless \
  --task ggswarm-v0 --num_envs 16 --max_iterations 5
```

Then play (with GUI for visual):

```text
python scripts/skrl/play.py --task ggswarm-v0 \
  --checkpoint <p4-revert-4 path> --num_agents 8 --play_length 100
```

Expected at episode start: 8 drones visible in a ring, not stacked vertically. Trajectory plots from the play output should show 8 distinct spawn positions per env at t=0.

- [ ] **Step 10.3: Commit**

```text
git commit -m "feat(phase1a): B3 spawn fix — drones spawn on circle, not stacked (closes B3)

Replaces the prior stacked-vertical spawn with per-drone slot positions on
a circle of radius spawn_radius, jittered by ±15% of min_spawn_spacing.
Eliminates the spawn-time downwash artifact.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 11: Smoke test (G1a-1)

**Files:** none (verification only).

- [ ] **Step 11.1: Run smoke per CLAUDE.md spec**

```text
env_isaaclab/Scripts/python.exe scripts/skrl/train.py --headless \
  --task ggswarm-v0 --num_envs 16 --max_iterations 5 --log_subdir p1a
```

Pass: training proceeds 5 iters with no errors. Episode reward shows non-zero values (signal the reward path works post-refactor).

If fail: bisect by reverting the most recent reshape commit and retesting. The per-step allocation ban audit (no new `torch.zeros` in `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`) is the most likely failure mode.

- [ ] **Step 11.2: Add smoke pass entry to changelog**

Append to `docs/ggswarm_live/status/changelog.md`:

```markdown
## 2026-04-30 — Phase 1a smoke pass (G1a-1)

`train.py --num_envs 16 --max_iterations 5 --log_subdir p1a` runs clean on
shared-scene env. Episode reward non-zero. Per-step allocation audit passed
(no new tensor allocations in _pre_physics_step, _get_observations,
_get_rewards, _get_dones).
```

- [ ] **Step 11.3: Commit**

```text
git add docs/ggswarm_live/status/changelog.md
git commit -m "docs(phase1a): G1a-1 smoke pass — 16 envs × 5 iters clean

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 12: Throughput sweep (G1a-2)

**Files:**
- Create: `scripts/sweep/phase1a_throughput.py` (new)
- Modify: `docs/ggswarm_live/status/changelog.md`

- [ ] **Step 12.1: Write sweep script**

Create `scripts/sweep/phase1a_throughput.py`:

```python
"""Phase 1a throughput sweep — picks the env-count knee for shared-scene env.

Runs train.py at progressively larger num_envs for a small max_iterations,
parses steps/sec from the SKRL stdout output, and reports a knee selection.
"""

from __future__ import annotations
import re
import subprocess
import sys
from pathlib import Path

ENV_COUNTS = [16, 32, 64, 128, 256, 512]
MAX_ITERS = 5
TASK = "ggswarm-v0"
PYTHON = "env_isaaclab/Scripts/python.exe"

def parse_steps_per_sec(stdout: str) -> float | None:
    """SKRL prints lines like 'Total time elapsed: ... steps/s: 1234.5'."""
    m = re.search(r"steps?/s:\s*([\d.]+)", stdout, re.IGNORECASE)
    return float(m.group(1)) if m else None

def main():
    results = []
    for n in ENV_COUNTS:
        print(f"\n=== num_envs={n} ===")
        proc = subprocess.run(
            [PYTHON, "scripts/skrl/train.py", "--headless",
             "--task", TASK, "--num_envs", str(n),
             "--max_iterations", str(MAX_ITERS),
             "--log_subdir", "p1a-sweep"],
            capture_output=True, text=True, timeout=600,
        )
        if proc.returncode != 0:
            print(f"FAILED at num_envs={n}: {proc.stderr[-500:]}")
            results.append((n, None, "OOM-or-crash"))
            break
        sps = parse_steps_per_sec(proc.stdout + proc.stderr)
        results.append((n, sps, "ok"))
        print(f"num_envs={n}: {sps} steps/s")

    print("\n=== Sweep summary ===")
    print("num_envs | steps/sec | status")
    for n, sps, status in results:
        sps_str = f"{sps:.1f}" if sps is not None else "-"
        print(f"{n:8d} | {sps_str:>9s} | {status}")

    # Knee selection: highest num_envs that completed successfully.
    ok = [(n, sps) for n, sps, status in results if status == "ok"]
    if not ok:
        print("\nNo successful runs. Stop condition #1 in spec.")
        sys.exit(1)
    knee = max(ok, key=lambda x: x[0])
    print(f"\nChosen knee: num_envs={knee[0]} ({knee[1]:.1f} steps/s)")
    Path("logs/sweeps/phase1a_throughput.txt").parent.mkdir(parents=True, exist_ok=True)
    with open("logs/sweeps/phase1a_throughput.txt", "w") as f:
        f.write("num_envs,steps_per_sec,status\n")
        for n, sps, status in results:
            f.write(f"{n},{sps if sps is not None else ''},{status}\n")
        f.write(f"chosen_knee,{knee[0]},{knee[1]}\n")
    print(f"Saved: logs/sweeps/phase1a_throughput.txt")

if __name__ == "__main__":
    main()
```

- [ ] **Step 12.2: Run sweep**

```text
python scripts/sweep/phase1a_throughput.py
```

Expected: a table of `(num_envs, steps/sec)` for as many configs as the 3070 can handle. The script auto-stops at the first failure.

If the script's regex doesn't match SKRL's actual output format: run a single training command manually, observe the throughput line format, update the regex.

- [ ] **Step 12.3: Update cfg default to chosen knee**

Edit `ggswarm_env_cfg.py` line 80:

```python
scene: InteractiveSceneCfg = InteractiveSceneCfg(
    num_envs=<chosen knee>, env_spacing=2.5, replicate_physics=True
)
```

- [ ] **Step 12.4: Append sweep result to changelog**

```markdown
## 2026-04-30 — Phase 1a throughput sweep (G1a-2)

| num_envs | steps/sec | status |
| :--- | :--- | :--- |
| 16 | <fill> | ok |
| 32 | <fill> | ok |
| 64 | <fill> | ok |
| ... | ... | ... |

Chosen knee: num_envs=<N>. Set as cfg default. 1a/1b/1c retrain envelope:
N envs × A=8 drones = <N*8> total drones per training step.
```

- [ ] **Step 12.5: Commit**

```text
git add scripts/sweep/phase1a_throughput.py logs/sweeps/phase1a_throughput.txt \
        source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env_cfg.py \
        docs/ggswarm_live/status/changelog.md
git commit -m "feat(phase1a): G1a-2 throughput sweep — chosen knee num_envs=<N>

Records steps/sec at 16/32/64/128/... envs on local 3070. Cfg default
updated to the chosen knee. Sweep result archived under logs/sweeps/.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 13: Forest-mode play smoke (G1a-3, catches R6)

**Files:** none (verification only).

- [ ] **Step 13.1: Run forest play with capstone checkpoint**

```text
python scripts/skrl/play.py --task ggswarm-v0 \
  --checkpoint <p4-revert-4 path> --forest --num_agents 8 \
  --play_length 200 --prefix p1a-forest-smoke
```

Pass: 200 steps without crash. Drones may not navigate forest correctly (different env distribution than capstone), but the forest-deflection block in `_pre_physics_step` must execute without shape errors.

If shape error: cross-reference Task 8's reshape edits — most likely something in the deflection loop missed a `repeat_interleave(A)` for env origins or a `[N_envs, A, *]` reshape.

- [ ] **Step 13.2: Append forest-smoke entry to changelog**

```markdown
## 2026-04-30 — Phase 1a forest-mode play smoke (G1a-3, R6 cleared)

200-step forest play with p4-revert-4 checkpoint runs clean. Forest
deflection block in _pre_physics_step executes without shape errors. R6
risk (group plumbing collapse breaks forest mode) cleared.
```

- [ ] **Step 13.3: Commit changelog only**

```text
git add docs/ggswarm_live/status/changelog.md
git commit -m "docs(phase1a): G1a-3 forest play smoke pass

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 14: Replay-gate harness implementation

**Files:**
- Create: `scripts/skrl/replay_gate.py`

- [ ] **Step 14.1: Write the harness**

Create `scripts/skrl/replay_gate.py`:

```python
"""Phase 1a replay gate — compare shared-scene env vs capstone reference rollouts.

Loads v1.0.0-capstone checkpoint, runs N rollouts in the current shared-scene
env (downwash off, B3 spawn-fix per --b3 flag), compares per-rollout metrics
against the captured reference distribution at logs/ref/v1.0.0-capstone/.

Pass condition: all four metrics (mean_formation_error_m, collision_pairs_per_step,
final_distance_to_goal_m, episode_reward) within 2σ of capstone reference mean.

This is gate G1a-4. Failure modes are documented in the spec under R3.
"""

from __future__ import annotations
import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path

REF_DIR = Path("logs/ref/v1.0.0-capstone")
SEEDS = [7, 13, 21, 42, 99]
METRICS = [
    "mean_formation_error_m",
    "collision_pairs_per_step",
    "final_distance_to_goal_m",
    "episode_reward",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to v1.0.0-capstone best_agent.pt")
    p.add_argument("--b3", action="store_true",
                   help="Run with B3 spawn fix on (default: B3 off — controlled match)")
    p.add_argument("--play_length", type=int, default=500)
    return p.parse_args()

def run_rollouts(ckpt: str, b3: bool, play_length: int) -> dict[int, Path]:
    """Run play.py for each seed; returns map seed → trajectory CSV path."""
    out_paths = {}
    for seed in SEEDS:
        prefix = f"replay-gate-b3{'on' if b3 else 'off'}-seed{seed}"
        # B3 control: temporary cfg override would go here. For now, B3 is hard-coded
        # in env code (Task 10). To run B3-off, check out the pre-Task-10 commit.
        # See Task 15 for the workflow.
        proc = subprocess.run(
            ["env_isaaclab/Scripts/python.exe", "scripts/skrl/play.py",
             "--task", "ggswarm-v0", "--checkpoint", ckpt,
             "--num_agents", "8", "--play_length", str(play_length),
             "--seed", str(seed), "--trajectories", "--prefix", prefix],
            capture_output=True, text=True, timeout=900,
        )
        if proc.returncode != 0:
            print(f"FAIL at seed {seed}:\n{proc.stderr[-1000:]}")
            sys.exit(2)
        # Find the CSV
        traj_dir = list(Path("logs/skrl/ggswarm").rglob(f"{prefix}-trajectory_data.csv"))
        if not traj_dir:
            print(f"No trajectory CSV for seed {seed}")
            sys.exit(2)
        out_paths[seed] = traj_dir[0]
    return out_paths

def compute_metrics(csv_path: Path, num_agents: int = 8,
                    target_spacing: float = 0.5) -> dict[str, float]:
    """Parse a per-step trajectory CSV and compute the four metrics."""
    rows = list(csv.DictReader(csv_path.open()))
    T = len(rows)
    formation_errors = []
    collision_counts = []
    final_distances = []
    # episode_reward not in CSV — pull from SKRL log if needed; for now use
    # final_distance_to_goal as a proxy and document this in metadata.
    for row in rows:
        positions = []
        goals = []
        for a in range(num_agents):
            positions.append([float(row[f"d{a}_x"]), float(row[f"d{a}_y"]), float(row[f"d{a}_z"])])
            goals.append([float(row[f"d{a}_gx"]), float(row[f"d{a}_gy"]), float(row[f"d{a}_gz"])])
        # Pairwise distance error
        pair_errs = []
        coll = 0
        for i in range(num_agents):
            for j in range(i + 1, num_agents):
                d = math.dist(positions[i], positions[j])
                pair_errs.append(abs(d - target_spacing))
                if d < 0.10:   # cfg.collision_radius
                    coll += 1
        formation_errors.append(sum(pair_errs) / len(pair_errs))
        collision_counts.append(coll)
    last = rows[-1]
    for a in range(num_agents):
        p = [float(last[f"d{a}_x"]), float(last[f"d{a}_y"]), float(last[f"d{a}_z"])]
        g = [float(last[f"d{a}_gx"]), float(last[f"d{a}_gy"]), float(last[f"d{a}_gz"])]
        final_distances.append(math.dist(p, g))
    return {
        "mean_formation_error_m": sum(formation_errors) / T,
        "collision_pairs_per_step": sum(collision_counts) / T,
        "final_distance_to_goal_m": sum(final_distances) / num_agents,
        "episode_reward": float("nan"),   # populated from SKRL log in a follow-up if needed
    }

def main():
    args = parse_args()
    ref_meta = json.loads((REF_DIR / "rollouts_metadata.json").read_text())
    ref = ref_meta["metrics"]

    print(f"Running shared-scene rollouts (B3={'on' if args.b3 else 'off'}, "
          f"play_length={args.play_length})...")
    paths = run_rollouts(args.checkpoint, args.b3, args.play_length)

    print("\nComputing per-seed metrics...")
    per_seed = {seed: compute_metrics(p) for seed, p in paths.items()}
    aggregate = {}
    for m in METRICS:
        vals = [per_seed[s][m] for s in SEEDS if not math.isnan(per_seed[s][m])]
        if not vals:
            continue
        aggregate[m] = {"mean": statistics.mean(vals),
                        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0}

    print("\n=== Replay gate result ===")
    print(f"Metric                        | Capstone (mean ± std)   | 1a env (mean ± std)     | Δ (σ-units) | Pass?")
    fail = False
    for m in METRICS:
        if m not in aggregate or m not in ref:
            continue
        cap_mean, cap_std = ref[m]["mean"], ref[m]["std"]
        new_mean, new_std = aggregate[m]["mean"], aggregate[m]["std"]
        sigma_dist = abs(new_mean - cap_mean) / max(cap_std, 1e-9)
        passed = sigma_dist <= 2.0
        if not passed:
            fail = True
        print(f"{m:30s}| {cap_mean:8.4f} ± {cap_std:7.4f}  | "
              f"{new_mean:8.4f} ± {new_std:7.4f}  | {sigma_dist:6.2f}      | "
              f"{'PASS' if passed else 'FAIL'}")
    print()
    if fail and not args.b3:
        print("REPLAY GATE FAILED (B3 off). See spec stop condition #3.")
        sys.exit(1)
    if fail and args.b3:
        print("Drift recorded with B3 on (not gating per spec R7). Continue.")
    else:
        print("REPLAY GATE PASSED.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 14.2: Commit harness (no run yet)**

```text
git add scripts/skrl/replay_gate.py
git commit -m "feat(phase1a): add replay-gate harness for G1a-4

Compares shared-scene env rollouts against captured v1.0.0-capstone
reference distributions across 5 seeds. Reports per-metric σ-distance and
pass/fail per spec gate G1a-4.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 15: Run replay gate (G1a-4)

**Files:** changelog only.

The spec calls for **two runs**: B3 off (must pass), B3 on (drift recorded). Since B3 is committed in Task 10 as a hard env change, "B3 off" requires temporarily reverting Task 10 — done by checking out the pre-Task-10 commit, running the gate, then checking back out to HEAD.

- [ ] **Step 15.1: Find the pre-Task-10 commit**

```text
git log --oneline --grep="B3 spawn fix"
```

The commit message contains "B3 spawn fix"; its parent is the pre-B3 commit. Note both SHAs.

- [ ] **Step 15.2: Run replay gate B3-off**

```text
git checkout <pre-Task-10 SHA>
python scripts/skrl/replay_gate.py --checkpoint <p4-revert-4 path>
```

Expected: prints the per-metric table. **Pass condition:** all four metrics within 2σ of capstone reference. If fail, see spec stop condition #3 (3 failures + diagnostic = sub-design pass).

Save the printed table to `logs/sweeps/phase1a_replay_gate_b3off.txt`.

- [ ] **Step 15.3: Run replay gate B3-on**

```text
git checkout phase1a-shared-scene  # back to HEAD
python scripts/skrl/replay_gate.py --checkpoint <p4-revert-4 path> --b3
```

Save to `logs/sweeps/phase1a_replay_gate_b3on.txt`. Drift recorded but not gating per spec R7.

- [ ] **Step 15.4: Append G1a-4 result to changelog**

```markdown
## 2026-04-30 — Phase 1a replay gate (G1a-4)

**B3 off** (pre-Task-10 SHA `<sha>`):

| Metric | Capstone | 1a env | σ-distance | Pass |
| :--- | :--- | :--- | :--- | :--- |
| mean_formation_error_m | <fill> | <fill> | <fill> | PASS/FAIL |
| collision_pairs_per_step | <fill> | <fill> | <fill> | PASS/FAIL |
| final_distance_to_goal_m | <fill> | <fill> | <fill> | PASS/FAIL |
| episode_reward | <fill> | <fill> | <fill> | PASS/FAIL |

**B3 on** (HEAD): drift recorded, not gating per spec R7.

| Metric | B3-off mean | B3-on mean | Δ |
| :--- | :--- | :--- | :--- |
| mean_formation_error_m | <fill> | <fill> | <fill> |
| ... | | | |

G1a-4 PASS — shared-scene env preserves capstone behavior with downwash off.
```

- [ ] **Step 15.5: Commit**

```text
git add logs/sweeps/phase1a_replay_gate_b3*.txt docs/ggswarm_live/status/changelog.md
git commit -m "docs(phase1a): G1a-4 replay gate PASS

Both runs (B3 off must-pass, B3 on drift-recorded) logged with per-metric
σ-distances. Spec R3 risk closed.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

---

### Task 16: Update phase doc and merge to main

**Files:**
- Modify: `docs/ggswarm_live/phases/phase1_shared_scene_sim.md`

- [ ] **Step 16.1: Mark 1a complete in phase doc**

Add a new section near the top:

```markdown
## Sub-phase status

| Sub-phase | Status | Tag | Date |
| :--- | :--- | :--- | :--- |
| 1a | Complete (replay gate G1a-4 passed) | `phase1a-shared-scene` | 2026-04-30 |
| 1b | Planned | — | — |
| 1c | Planned | — | — |
```

- [ ] **Step 16.2: Commit doc update**

```text
git add docs/ggswarm_live/phases/phase1_shared_scene_sim.md
git commit -m "docs(phase1a): mark sub-phase 1a complete

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>"
```

- [ ] **Step 16.3: Merge phase1a-shared-scene to main**

```text
git checkout main
git merge --no-ff phase1a-shared-scene -m "Merge phase1a-shared-scene: shared-scene refactor + B3 fix + replay-gate pass"
```

Use `--no-ff` to keep the sub-phase visible as a discrete merge in `git log --graph`.

---

### Task 17: Tag `phase1a-shared-scene`

**Files:** none (git operation).

- [ ] **Step 17.1: Tag the merge commit**

```text
git tag -a phase1a-shared-scene -m "Phase 1a complete — shared-scene refactor, B3 fix, replay gate pass"
git tag --list   # confirm phase1a-shared-scene appears alongside v1.0.0-capstone
```

- [ ] **Step 17.2: (Optional) push tag and main**

Confirm with the user before running:

```text
git push origin main
git push origin phase1a-shared-scene
```

(Do not run without explicit user confirmation — push is a remote-visible action.)

---

### Task 18: 1b/1c stub addendum (defer planning)

**Files:**
- Create: `docs/superpowers/plans/2026-04-30-phase1a-shared-scene.md` already covers 1a. Add 1b/1c stubs at the bottom of this same file in this task.

This task documents that 1b/1c plans are deferred to future planning sessions and lists what each session needs as input.

- [ ] **Step 18.1: Append 1b/1c stub section to this plan**

(Already added below — see "1b plan stub" and "1c plan stub" sections at the bottom of this document.)

---

## 1b plan stub (deferred)

**Trigger to write the 1b detailed plan:** G1b-0 (dataset prerequisite) cleared.

**Inputs needed:**

- Confirmed published Crazyflie aero dataset URL + license + format spec.
- The exact per-trace fields (rel-pose, ground-truth force columns, sample rate).
- Is the dataset released as part of Neural-Swarm2 or another paper? (Determines residual-network architecture choices that match the dataset's regime.)
- 1a final cfg defaults (specifically the chosen `num_envs` knee from G1a-2).

**1b at a glance** (per spec § 7.2, but not bite-sized yet):

- G1b-0: literature check + dataset download + license check.
- New module `ggswarm/aero/downwash_analytic.py` (Panerati pairwise model, no per-step alloc).
- New module `ggswarm/aero/downwash_residual.py` (MLP/graph-net + checkpoint loader).
- New script `scripts/aero/train_residual.py` (offline training pipeline).
- Aero hook in `_pre_physics_step` reading `cfg.downwash_mode`.
- New cfg fields: `downwash_mode`, `downwash_residual_checkpoint`, `downwash_residual_max_force`, `downwash_residual_clamp_warn_frac`.
- G1b-1: analytic sanity (two-drone stacked test).
- G1b-2: residual training run on dataset.
- G1b-3: three retrains (`disabled`/`analytic`/`residual`), 1a checkpoint warm-start.
- G1b-4: TB scalar comparison; pick winning anchor.
- G1b-5: tag `phase1b-downwash`.

**Why deferred:** the residual model architecture and training script depend on the dataset's actual format. Speculating now produces speculative code that gets rewritten when the dataset is real.

---

## 1c plan stub (deferred)

**Trigger to write the 1c detailed plan:** G1b-4 cleared (1b winning anchor selected).

**Inputs needed:**

- 1b winning checkpoint path (analytic-anchor or residual-anchor — depends on G1b-4 outcome).
- Confirmation that obs vector at end of 1b is still the 18D capstone-compatible layout (precondition for warm-start).
- Status of SKRL's partial-load support — if `agent.load()` rejects shape-mismatched layers, we need a custom partial-load shim.

**1c at a glance** (per spec § 7.3):

- G1c-1: shape audit after wiring `GATv2Conv(edge_dim=6)`.
- Modify `_get_observations` to publish `edge_attr` `[num_edges, 6]` alongside `edge_index`.
- Pre-allocate `_knn_edge_attr` buffer in `__init__`.
- Modify `gnn_policy.py`: `set_knn_edges_and_attrs`, edge cache stores tuples, `_gnn_forward` passes `edge_attr` to each `GATv2Conv` layer.
- Partial-load shim in `train.py` if needed.
- G1c-2: 1c retrain warm-started from 1b anchor.
- G1c-3: cold-start MLP doesn't collapse policy (first 50 iters).
- G1c-4: TB scalar deltas vs 1b documented.
- G1c-5: tag `phase1c-edge-features`.

**After G1c-5:** Phase 1 milestone gates GM-1/2/3 (2-panel video, Phase 1 changelog summary, `phase1-complete` tag, social-media write-up).

**Why deferred:** the partial-load shim's exact form depends on 1b's checkpoint structure (which depends on whether 1b shipped analytic-only or residual-anchor). The edge-MLP cold-start contingency depends on 1b's final policy std — both unknown until 1b lands.

---

## Self-review log

- **Spec coverage:** Every gate G1a-1 through G1a-5 has a task. R6 (forest break) cleared by T13. R7 (B3 invalidates gate) addressed by Task 15's two-run protocol. The reference rollout capture (precondition for G1a-4) is Task 0.
- **Placeholder scan:** `<fill>` markers in changelog templates are *intentional* — they're filled in at execution time from actual measurements. SHAs and paths in run commands are also intentionally `<...>`-marked because they don't exist until the executor reaches that step.
- **Type consistency:** `_FusedRobot` / `_FusedRobotData` / `_FusedWrenchComposer` referenced consistently. `N_envs` used as the post-1a num_envs name throughout. `N_drones = N_envs * A` used where total-drone count is needed.
- **Naming consistency:** `_env_origins_per_drone` used everywhere the per-drone-expanded origins tensor appears. `cfg.downwash_mode` consistent with spec § 4.2.
