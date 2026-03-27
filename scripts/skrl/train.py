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
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
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
parser.add_argument(
    "--num_envs", type=int, default=None, help="Number of environments to simulate."
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
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
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
parser.add_argument(
    "--log_subdir",
    type=str,
    default=None,
    help="Subdirectory under experiment directory (e.g. phase2a, phase2b).",
)
parser.add_argument(
    "--max_iterations", type=int, default=4800, help="RL Policy training iterations."
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
import time
from datetime import datetime

import gymnasium as gym
import skrl
from packaging import version

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

import ggswarm.tasks  # noqa: F401

# Determine the agent configuration entry point based on algorithm or explicit argument
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = (
        "skrl_cfg_entry_point"
        if algorithm in ["ppo"]
        else f"skrl_{algorithm}_cfg_entry_point"
    )
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


# Decorator to load the task and agent configuration using Hydra
# It automatically merges default configs with the entry point specified
@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict
):
    """Train with skrl agent."""
    # Configure environment overrides from CLI
    env_cfg.scene.num_envs = (
        args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    )
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # Check for invalid combination of CPU device with distributed training
    if (
        args_cli.distributed
        and args_cli.device is not None
        and "cpu" in args_cli.device
    ):
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # Set the GPU device ID for multi-GPU (distributed) training
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"

    # Set training duration and ML framework backend
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = (
            args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
        )
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # Initialize seeds for both the agent (skrl) and the environment (Isaac Lab)
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = (
        args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    )
    env_cfg.seed = agent_cfg["seed"]

    # Define the base log root and the specific run directory (timestamped)
    log_root_path = os.path.join(
        "logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]
    )
    if args_cli.log_subdir:
        log_root_path = os.path.join(log_root_path, args_cli.log_subdir)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")

    # Run name includes timestamp, algorithm, and framework
    log_dir = (
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        + f"_{algorithm}_{args_cli.ml_framework}"
    )
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
    resume_path = (
        retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None
    )

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
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

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

    # Initialize the skrl Runner to manage the training loop
    # Docs: https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    # Load agent weights if a checkpoint was provided
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # Execute the training process
    runner.run()

    print(f"Training time: {round(time.time() - start_time, 2)} seconds")

    # Close the simulator environment
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
