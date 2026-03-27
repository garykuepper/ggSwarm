# CLAUDE.md — ggSwarm Project Rules

This file is automatically loaded by Claude Code at session start.
All rules below are MANDATORY unless marked SHOULD.

---

## Shell / Environment

- **Shell is PowerShell on Windows.** Use Unix shell syntax in Bash tool calls (forward slashes, `/dev/null`), but observe the PowerShell rules below for any commands the user runs locally.
- No `&&` chaining in PowerShell — use `;` or separate tool calls.
- No `head` in PowerShell — use `Select-Object -First N`.
- No `rg` as a shell command — use the Grep tool.
- `gcloud storage rsync` corrupts paths on Windows — use `gcloud storage cp`.
- Escape `$` as `\$` in `gcloud compute ssh --command="..."`, or wrap in single quotes.

---

## Project Architecture

- **Framework:** Isaac Lab + SKRL PPO (single-agent, not MAPPO)
- **Env:** `DirectRLEnv` with 1 Crazyflie per env, shared policy across all envs
- **CTDE:** Centralized Training (shared PPO), Decentralized Execution (each drone runs same policy independently)
- **Config class:** `GgswarmEnvCfg` in `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env_cfg.py`
- **Env class:** `GgswarmEnv` in `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_env.py`
- **Task ID:** `ggswarm-v0`
- **PPO config:** `agents/skrl_ppo_cfg.yaml` (quadcopter-proven hyperparams)
- **Viz utilities:** `source/ggswarm/ggswarm/viz/` (trajectory_plots.py, nvenc_recorder.py)

---

## GCE Training Operations

- Instance: `isaacsim`, zone: `us-central1-a`, project: `gg-swarm`
- Repo path on VM: `~/ggSwarm`
- Python on VM: `~/ggSwarm/env_isaaclab/bin/python`
- GCS bucket: `gs://gg-swarm-training-logs` (only valid bucket)
- **Launch pattern:**

  ```text
  gcloud compute ssh isaacsim --zone=us-central1-a --project=gg-swarm \
    --command='cd ~/ggSwarm && git pull origin main && \
    nohup ~/ggSwarm/env_isaaclab/bin/python scripts/skrl/train.py \
    --headless --task ggswarm-v0 --num_envs 4096 \
    --max_iterations 500 --log_subdir p2a \
    > ~/train_ggswarm_<label>.log 2>&1 &'
  ```

- **Log naming:** Use serialized run labels: `train_ggswarm_p2a-1.log`, `train_ggswarm_p2a-2.log`, etc. The label must match `--log_subdir` phase and increment per run.
- After training, manually push logs to GCS:

  ```text
  gcloud compute ssh isaacsim ... --command='gcloud storage cp -r \
    ~/ggSwarm/logs/skrl/ggswarm/<subdir>/<run> \
    gs://gg-swarm-training-logs/logs/skrl/ggswarm/<subdir>/'
  ```

- **GCE is for training only.** Play, video, trajectories, tensorboard must run locally.
- **Follow the log:**

  ```text
  gcloud compute ssh isaacsim --zone=us-central1-a --project=gg-swarm \
    --command="tail -f /home/gkuep/train_ggswarm_<label>_*.log"
  ```

---

## Training Commands

```text
# Train (local)
env_isaaclab/Scripts/python.exe scripts/skrl/train.py --headless \
  --task ggswarm-v0 --num_envs 512 --max_iterations 500 --log_subdir p2a

# Play (local, with GUI)
python scripts/skrl/play.py --task ggswarm-v0 \
  --checkpoint <path/to/best_agent.pt>

# Play + record video (NVENC H.264)
python scripts/skrl/play.py --task ggswarm-v0 \
  --checkpoint <path> --video --video_prefix p2a-1

# Play + trajectory plots
python scripts/skrl/play.py --task ggswarm-v0 \
  --checkpoint <path> --trajectories --play_length 500

# TensorBoard
tensorboard --logdir logs/skrl/ggswarm
```

---

## Video Recording (MANDATORY)

- Always pass `--video_prefix <run_label>` when recording video.
- The run label must match the label used in `docs/status/run_history.md` for traceability.
- Videos use NVENC H.264 hardware encoding via `ggswarm.viz.nvenc_recorder`.
- Default resolution: 1920×1080 (set in `GgswarmEnvCfg.viewer`).

---

## Reward + Hyperparameter Hygiene (MANDATORY)

- Every reward term must have a scale in `GgswarmEnvCfg` and (if applicable) a sigma/threshold parameter.
- Planned-but-disabled reward terms must be `0.0` in cfg (not commented out).
- Any reward change must be logged in `docs/status/changelog.md` with rationale.
- No magic numbers in env core — all tunable constants must be cfg parameters.

## Tensor Shape Contracts (MANDATORY)

- Any function manipulating drone tensors must add a shape comment on first access: `# shape: [num_envs, dim]`
- Required for: `pos_w`, `quat_w`, `lin_vel_b`, `ang_vel_b`, `actions`, `rewards`, and intermediate tensors.

## Per-Step Allocation Ban (MANDATORY)

- No `torch.zeros`, `torch.ones`, `torch.empty`, `torch.full`, or `tensor.clone()` inside: `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones`.
- All scratch buffers must be pre-allocated in `__init__` and reused.

## Logging Policy (MANDATORY)

- No `print()` inside per-step environment code — use `logger.*`.
- One-time init → `logger.info`; per-episode diagnostics → `logger.debug`; unexpected recoverable → `logger.warning`.
- `print()` allowed in `scripts/` and demos for CLI output.

---

## Documentation + Status (SHOULD)

| Document | Update trigger |
| :--- | :--- |
| `docs/status/changelog.md` | Every reward change, phase transition, or major bug fix |
| `docs/status/weekly_updates.md` | Weekly snapshot (Tuesdays) |
| `docs/phases/` | Phase start/completion |

## Markdown Hygiene (MANDATORY)

- GFM tables: header → separator (`| :--- |`) → body.
- Every fenced code block must be surrounded by blank lines on both sides.

## Spelling / Terminology (SHOULD)

- Maintain spelling exceptions in `cspell.json`.

---

## Hard Deadline: Apr 24, 2026 (MANDATORY)

Capstone Festival presentation and all submissions due **Apr 24, 2026**.

### Decision authority

Gary makes all scope-cut calls. The assistant **must not propose extending any deadline** — only propose what to cut.

### Timeline references

- Authoritative source: `docs/project/proposal.md` § 7.
- Current status: `docs/status/weekly_updates.md`.

## Post-Training Review (MANDATORY)

After every training run, before any reward change or relaunch:

1. Check TB scalars (reward, episode length, policy std).
2. Run play with `--trajectories` to verify behavior visually.
3. Log results in `docs/status/changelog.md`.

## Smoke Test Before GCE (MANDATORY)

Run a short local test before any GCE launch:

```text
env_isaaclab/Scripts/python.exe scripts/skrl/train.py --headless \
  --task ggswarm-v0 --num_envs 64 --max_iterations 5
```

Confirm: no errors, training progresses, correct task loads.

## Type Annotation Completeness (SHOULD)

- All new public functions and dataclass fields must carry full type annotations.
- All new public classes and functions must have at minimum a one-line docstring.

## Cyclomatic Complexity Cap (SHOULD)

- New functions must stay at or below cyclomatic complexity **15**.
