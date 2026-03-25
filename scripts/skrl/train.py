# Copyright (c) 2022-2026, ggSwarm Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# Argument parsing for command-line configuration
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument(
    "--video_length",
    type=int,
    default=200,
    help="Length of the recorded video (in steps).",
)
parser.add_argument(
    "--video_interval",
    type=int,
    default=2000,
    help="Interval between video recordings (in steps).",
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--num_agents",
    type=int,
    default=None,
    help="Number of agents (override env config).",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent",
    type=str,
    default=None,
    help=(
        "Name of the RL agent configuration entry point. Defaults to None, in which case the argument "
        "--algorithm is used to determine the default agent configuration entry point."
    ),
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed",
    action="store_true",
    default=False,
    help="Run training with multiple GPUs or nodes.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Path to model checkpoint to resume training.",
)
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--no_progress",
    action="store_true",
    default=False,
    help="Disable periodic training progress and ETA logs.",
)
parser.add_argument(
    "--progress_interval_s",
    type=float,
    default=10.0,
    help="Seconds between periodic progress/ETA updates.",
)
parser.add_argument(
    "--eta_window_s",
    type=float,
    default=120.0,
    help="Rolling window in seconds for throughput-based ETA estimation.",
)
parser.add_argument(
    "--export_io_descriptors",
    action="store_true",
    default=False,
    help="Export IO descriptors.",
)
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)
parser.add_argument(
    "--ray-proc-id",
    "-rid",
    type=int,
    default=None,
    help="Automatically configured by Ray integration, otherwise None.",
)
parser.add_argument(
    "--gnn",
    action="store_true",
    default=False,
    help="Use the custom GATv2 GNN policy instead of the default MLP.",
)
parser.add_argument(
    "--early_stop_step",
    type=int,
    default=40000,
    help=(
        "Step at which to run a health check on TensorBoard scalars. "
        "If key metrics indicate a failed run (e.g. thrust collapsed, "
        "ground_hit rising), training is stopped early. 0 = disabled. "
        "Default: 20000."
    ),
)
parser.add_argument(
    "--no_early_stop",
    action="store_true",
    default=False,
    help="Disable the automatic early-stop health check.",
)
parser.add_argument(
    "--action_telemetry_steps",
    type=int,
    default=None,
    help=(
        "If set, overrides env_cfg.action_telemetry_max_env_steps for TensorBoard "
        "diagnostics (first N env steps). Use for short local runs; omit for GCE "
        "(cfg default 0)."
    ),
)
# Append AppLauncher-specific CLI arguments
AppLauncher.add_app_launcher_args(parser)
# Parse the combined arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import logging
import os
import random
import re
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from threading import Event, Thread
from typing import Any

import gymnasium as gym
import skrl
from packaging import version

# Enable TF32 precision on Ada Lovelace (L4, RTX 40xx, etc.) for ~2x speedup
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# check for minimum supported skrl version
SKRL_VERSION = "1.4.3"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.skrl import SkrlVecEnvWrapper

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# Initialize logger for this module
logger = logging.getLogger(__name__)

import ggSwarm.tasks  # noqa: F401


def _format_duration(seconds: float) -> str:
    """Format duration in HH:MM:SS."""
    total_seconds = max(0, int(seconds))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _latest_checkpoint_step(checkpoint_dir: Path) -> int:
    """Return the highest checkpoint step from files like agent_10000.pt."""
    if not checkpoint_dir.exists():
        return 0

    max_step = 0
    for ckpt in checkpoint_dir.glob("agent_*.pt"):
        match = re.match(r"agent_(\d+)\.pt$", ckpt.name)
        if match:
            max_step = max(max_step, int(match.group(1)))
    return max_step


def _latest_event_step(log_dir: Path) -> int:
    """Return the latest scalar step from TensorBoard event files."""
    event_files = sorted(log_dir.glob("events.out.tfevents.*"), key=lambda p: p.stat().st_mtime)
    if not event_files:
        return 0

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return 0

    max_step = 0
    for event_path in event_files[-2:]:
        try:
            accumulator = EventAccumulator(str(event_path))
            accumulator.Reload()
            scalar_tags = accumulator.Tags().get("scalars", [])
            for tag in scalar_tags:
                scalars = accumulator.Scalars(tag)
                if scalars:
                    max_step = max(max_step, int(scalars[-1].step))
        except Exception:
            continue
    return max_step


def _check_training_health(log_dir: Path, check_step: int) -> tuple[bool, str]:
    """Check TensorBoard scalars for known failure patterns.

    Args:
        log_dir: Path to the training run log directory.
        check_step: The step at which to evaluate health.

    Returns:
        Tuple of (healthy, reason). If healthy is False, reason explains why.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        return True, "tensorboard not available; skipping health check"

    try:
        ea = EventAccumulator(str(log_dir))
        ea.Reload()
    except Exception:
        return True, "could not load TFEvents; skipping health check"

    available = set(ea.Tags().get("scalars", []))
    failures: list[str] = []

    # Check 1: thrust_val_mean — should be approaching 0.5 (hover), not collapsed
    tag = "Info / thrust_val_mean"
    if tag in available:
        events = ea.Scalars(tag)
        # Find the value closest to check_step
        recent = [e for e in events if e.step >= check_step * 0.8]
        if recent:
            val = recent[-1].value
            if val < 0.25:
                failures.append(
                    f"thrust_val_mean={val:.3f} at step {recent[-1].step} "
                    f"(expected >0.25 for hover; policy may be cutting thrust)"
                )

    # Check 2: mean_dist_to_goal — should not be diverging (> 2m = flying away)
    # Note: ground_hit_rate check removed — PD10 showed a normal "exploration dip"
    # where crashes spike at 20k then recover by 50k. Only truly pathological
    # runs (PD13/PD14: dist_to_goal > 6m) need to be caught.
    tag = "Info / mean_dist_to_goal"
    if tag in available:
        events = ea.Scalars(tag)
        recent = [e for e in events if e.step >= check_step * 0.8]
        if recent:
            recent_val = sum(e.value for e in recent) / len(recent)
            if recent_val > 2.0:
                failures.append(
                    f"mean_dist_to_goal={recent_val:.3f} at step {recent[-1].step} "
                    f"(> 2.0m — drones flying away from goal)"
                )

    if failures:
        return False, "; ".join(failures)
    return True, "all checks passed"


def _progress_monitor(
    *,
    stop_event: Event,
    checkpoint_dir: Path,
    log_dir: Path,
    total_timesteps: int,
    progress_target: int,
    poll_interval_s: float,
    eta_window_s: float,
    early_stop_step: int = 0,
) -> None:
    """Emit periodic progress and throughput-based ETA logs.

    Args:
        progress_target: Target for progress reporting (iterations or timesteps).
                        When max_iterations is set, this is the iteration count.
                        Otherwise, this equals total_timesteps.
        early_stop_step: Step at which to run health check. 0 = disabled.
    """
    start_time = time.time()
    history: deque[tuple[float, int]] = deque()
    last_reported_step = -1
    last_emitted_step = -1
    last_heartbeat_time = start_time
    heartbeat_interval_s = max(30.0, poll_interval_s * 3.0)
    health_checked = False
    # Determine unit label based on whether we're tracking iterations or timesteps
    is_iteration_tracking = (progress_target != total_timesteps)
    unit_label = "iters" if is_iteration_tracking else "steps"

    while not stop_event.wait(timeout=max(1.0, poll_interval_s)):
        now = time.time()
        event_step = _latest_event_step(log_dir)
        checkpoint_step = _latest_checkpoint_step(checkpoint_dir)
        current_step = max(event_step, checkpoint_step)
        elapsed = now - start_time

        # Keep a rolling history of progress points for stable ETA.
        if current_step > 0 and current_step != last_reported_step:
            history.append((now, current_step))
            last_reported_step = current_step
        while history and (now - history[0][0]) > max(10.0, eta_window_s):
            history.popleft()

        rolling_sps = None
        if len(history) >= 2:
            dt = history[-1][0] - history[0][0]
            ds = history[-1][1] - history[0][1]
            if dt > 0 and ds > 0:
                rolling_sps = ds / dt

        percent = (100.0 * current_step / max(1, progress_target))
        if rolling_sps and current_step < progress_target:
            remaining = progress_target - current_step
            eta = _format_duration(remaining / rolling_sps)
            sps_txt = f"{rolling_sps:,.1f}"
        else:
            eta = "unknown"
            sps_txt = "unknown"

        is_new_progress = current_step != last_emitted_step
        is_heartbeat_due = (now - last_heartbeat_time) >= heartbeat_interval_s
        if not is_new_progress and not is_heartbeat_due:
            continue

        # Once training has reached or exceeded the target, emit one final line then stop.
        # Heartbeat spam after 100% clutters the log with no useful information.
        if current_step >= progress_target and not is_new_progress:
            continue

        # --- Early-stop health check (runs once at early_stop_step) ---
        if (
            early_stop_step > 0
            and not health_checked
            and current_step >= early_stop_step
        ):
            health_checked = True
            healthy, reason = _check_training_health(log_dir, early_stop_step)
            if healthy:
                print(
                    f"[HEALTH] Step {current_step:,}: {reason}",
                    flush=True,
                )
            else:
                print(
                    f"[HEALTH] EARLY STOP at step {current_step:,}: {reason}",
                    flush=True,
                )
                # Write a marker file for post-training detection
                kill_file = log_dir / "EARLY_STOP"
                kill_file.write_text(
                    f"Early stop triggered at step {current_step}\n"
                    f"Reason: {reason}\n"
                )
                # Signal the main thread to stop
                stop_event.set()
                # Send SIGTERM to own process — SKRL's runner.run() is blocking,
                # but the signal handler will interrupt it cleanly.
                import os as _os
                import signal as _signal
                _os.kill(_os.getpid(), _signal.SIGTERM)
                return

        status_suffix = ""
        if not is_new_progress:
            status_suffix = " | status=waiting_for_next_metrics"

        # Flush stderr first so that any tqdm bar output that slipped through is
        # separated from our structured [PROGRESS] line.
        sys.stderr.flush()
        print(
            "[PROGRESS] "
            f"{current_step:,}/{progress_target:,} ({percent:5.1f}%) | "
            f"elapsed={_format_duration(elapsed)} | "
            f"{unit_label}_per_sec={sps_txt} | eta={eta}"
            f"{status_suffix}",
            flush=True,
        )
        last_emitted_step = current_step
        last_heartbeat_time = now


def patch_mappo_single_agent_shared_states(env: Any) -> Any:
    """Patch env reset/step to inject MAPPO shared critic states.

    SKRL's ``SequentialTrainer.single_agent_train`` does not populate
    ``infos['shared_states']`` / ``infos['shared_next_states']``. MAPPO expects those keys.
    This patch keeps the wrapper type unchanged and monkey-patches ``reset`` and ``step``.
    """
    shared_states: Any = None
    original_reset = env.reset
    original_step = env.step

    def patched_reset(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        nonlocal shared_states
        obs, info = original_reset(*args, **kwargs)
        shared_states = env.state()
        if shared_states is not None and isinstance(info, dict):
            info = dict(info)
            info["shared_states"] = shared_states
            info["shared_next_states"] = shared_states
        return obs, info

    def patched_step(actions: Any) -> tuple[Any, Any, Any, Any, Any]:
        nonlocal shared_states
        obs, rewards, terminated, truncated, info = original_step(actions)
        shared_next_states = env.state()
        if (
            shared_states is not None
            and shared_next_states is not None
            and isinstance(info, dict)
        ):
            info = dict(info)
            info["shared_states"] = shared_states
            info["shared_next_states"] = shared_next_states
        shared_states = shared_next_states
        return obs, rewards, terminated, truncated, info

    env.reset = patched_reset
    env.step = patched_step
    return env


# Determine the agent configuration entry point based on algorithm or explicit argument
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


# Decorator to load the task and agent configuration using Hydra
# It automatically merges default configs with the entry point specified
@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # Suppress tqdm progress bars in headless training — they interleave with [PROGRESS] lines.
    if args_cli.headless:
        os.environ.setdefault("TQDM_DISABLE", "1")

    # Configure environment overrides from CLI
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.action_telemetry_steps is not None and hasattr(
        env_cfg, "action_telemetry_max_env_steps"
    ):
        env_cfg.action_telemetry_max_env_steps = int(args_cli.action_telemetry_steps)
    if hasattr(env_cfg, "num_agents") and args_cli.num_agents is not None:
        env_cfg.num_agents = args_cli.num_agents
        env_cfg.possible_agents = [f"drone_{i}" for i in range(env_cfg.num_agents)]
        env_cfg.action_spaces = {agent: 4 for agent in env_cfg.possible_agents}
        env_cfg.observation_spaces = {agent: 12 for agent in env_cfg.possible_agents}

    # Check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # Set the GPU device ID for multi-GPU (distributed) training
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # Set training duration and ML framework backend.
    # max_iterations is a count of rollout collections (skrl "timesteps" = rollout collections,
    # NOT environment steps). Pass it directly — do NOT multiply by rollouts.
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    # Override model with GNN if requested
    if args_cli.gnn:
        logger.info("Using GGSwarmGNNPolicy (GATv2) for training.")
        policy_cfg = agent_cfg["models"]["policy"]
        policy_cfg["class"] = "GGSwarmGNNPolicy"
        # GNN architecture (Rule 14): inject here so YAML MLP path never passes these to gaussian_model.
        hidden_ch = int(policy_cfg.setdefault("hidden_channels", 128))
        num_heads = int(policy_cfg.setdefault("num_heads", 2))
        init_ls = float(policy_cfg.get("initial_log_std", -0.5))
        logger.info(
            "GNN policy: GGSwarmGNNPolicy | hidden_channels=%s | num_heads=%s | "
            "initial_log_std=%s",
            hidden_ch,
            num_heads,
            init_ls,
        )
        # Disable the default 'network' key as GGSwarmGNNPolicy defines its own architecture
        if "network" in policy_cfg:
            del policy_cfg["network"]
            
        # Monkey-patch SKRL Runner to recognize the custom class string
        original_component = Runner._component

        def custom_component(self, name: str):
            if name.lower() == "ggswarmgnnpolicy":
                from ggSwarm.tasks.direct.ggswarm_marl.agents.skrl_gnn_policy import GGSwarmGNNPolicy
                return GGSwarmGNNPolicy
            return original_component(self, name)

        Runner._component = custom_component

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # Initialize seeds for both the agent (skrl) and the environment (Isaac Lab)
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # Define the base log root and the specific run directory (timestamped)
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # Run name includes timestamp, algorithm, and framework
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f"_{agent_cfg['agent']['experiment']['experiment_name']}"

    # Store the final paths back into the agent configuration
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # Dump simulation and agent configurations to the log directory for reproducibility
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # Get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # Set the IO descriptors export flag if requested (supported for manager-based envs)
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # Set the log directory for the environment
    env_cfg.log_dir = log_dir

    # Create the Isaac Lab environment through Gymnasium
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # Some algorithms (like standard PPO) require a single-agent observation/action space
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # Wrap environment with Gym's RecordVideo to save MP4s of the agent's behavior
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    start_time = time.time()

    # Wrap environment for skrl compatibility (handles device and data format conversion)
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

    # MAPPO + single agent: SKRL uses single_agent_train, which omits shared_states in infos.
    if algorithm == "mappo" and getattr(env, "num_agents", 0) == 1:
        logger.info(
            "Wrapping env for MAPPO single-agent: injecting shared_states / shared_next_states "
            "into infos (required by skrl MAPPO.record_transition)."
        )
        env = patch_mappo_single_agent_shared_states(env)

    # Initialize the skrl Runner to manage the training loop
    # Docs: https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    # Inject adj_matrix from env extras into GNN policy during act().
    # Must come after Runner (which creates the agent) and before training starts.
    if args_cli.gnn:
        from ggswarm_utils.sim_helpers import patch_mappo_gnn_adj_matrix  # noqa: PLC0415

        patch_mappo_gnn_adj_matrix(runner.agent, env)
        logger.info("Patched MAPPO.act() to inject adj_matrix into GNN policy inputs.")

    # Load agent weights if a checkpoint was provided
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # Execute the training process with optional periodic progress/ETA updates.
    monitor_stop_event: Event | None = None
    monitor_thread: Thread | None = None
    total_timesteps = int(agent_cfg["trainer"]["timesteps"])
    checkpoint_dir = Path(log_dir) / "checkpoints"
    # When max_iterations is provided, track progress in iteration-space (not timestep-space).
    # Otherwise, fall back to timestep-space for backward compatibility.
    progress_target = args_cli.max_iterations if args_cli.max_iterations else total_timesteps
    early_stop_step = 0 if args_cli.no_early_stop else args_cli.early_stop_step
    if not args_cli.no_progress:
        monitor_stop_event = Event()
        monitor_thread = Thread(
            target=_progress_monitor,
            kwargs={
                "stop_event": monitor_stop_event,
                "checkpoint_dir": checkpoint_dir,
                "log_dir": Path(log_dir),
                "total_timesteps": total_timesteps,
                "progress_target": progress_target,
                "poll_interval_s": args_cli.progress_interval_s,
                "eta_window_s": args_cli.eta_window_s,
                "early_stop_step": early_stop_step,
            },
            daemon=True,
        )
        monitor_thread.start()
        early_stop_msg = (
            f", early_stop_step={early_stop_step}" if early_stop_step > 0
            else ", early_stop=disabled"
        )
        print(
            "[INFO] Progress reporting enabled "
            f"(interval={args_cli.progress_interval_s:.1f}s, "
            f"eta_window={args_cli.eta_window_s:.1f}s"
            f"{early_stop_msg})."
        )

    early_stop_file = Path(log_dir) / "EARLY_STOP"
    try:
        runner.run()
    finally:
        if monitor_stop_event is not None:
            monitor_stop_event.set()
        if monitor_thread is not None:
            monitor_thread.join(timeout=2.0)

    if early_stop_file.exists():
        reason = early_stop_file.read_text().strip()
        print(f"\n[EARLY STOP] Training terminated early.\n{reason}")
        print(f"Training time: {round(time.time() - start_time, 2)} seconds")
        # Exit with code 2 so train_and_push.sh can distinguish early stop from success
        env.close()
        simulation_app.close()
        sys.exit(2)

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # Close the simulator environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
