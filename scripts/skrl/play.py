# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to play a checkpoint of an RL agent from skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="Play a checkpoint of an RL agent from skrl."
)
parser.add_argument(
    "--video", action="store_true", default=False, help="Record videos during training."
)
parser.add_argument(
    "--video_length",
    type=int,
    default=500,
    help="Length of the recorded video (in steps). 500 = 10s at dt=0.02.",
)
parser.add_argument(
    "--video_prefix",
    type=str,
    default="ggswarm",
    help="Filename prefix for video (e.g. p2b-3).",
)
parser.add_argument(
    "--num_agents", type=int, default=8,
    help="Drones per swarm group. >1 enables SwarmWrapper for formation.",
)
parser.add_argument(
    "--policy", type=str, default="gnn", choices=["mlp", "gnn"],
    help="Policy architecture: gnn (default, GATv2) or mlp.",
)
parser.add_argument(
    "--play_length",
    type=int,
    default=500,
    help="Number of steps to play. 500 = 10s at dt=0.02.",
)
parser.add_argument(
    "--trajectories",
    action="store_true",
    default=True,
    help="Record and plot drone trajectories.",
)
parser.add_argument(
    "--formation",
    type=str,
    default=None,
    help="Formation shape at play time: polygon, grid, triangle, letter_G, etc.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
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
    "--checkpoint", type=str, default=None, help="Path to model checkpoint."
)
parser.add_argument(
    "--seed", type=int, default=None, help="Seed used for the environment"
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
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
    "--real-time",
    action="store_true",
    default=False,
    help="Run in real-time, if possible.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True
# Sync play_length and video_length — use the larger value for both
if args_cli.video:
    max_len = max(args_cli.play_length, args_cli.video_length)
    args_cli.play_length = max_len
    args_cli.video_length = max_len

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import os
import random
import time

import gymnasium as gym
import skrl
import torch
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

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import ggswarm.tasks  # noqa: F401

# config shortcuts
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


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
    experiment_cfg: dict,
):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    elif args_cli.num_agents > 1:
        env_cfg.scene.num_envs = args_cli.num_agents  # one swarm group for play
    else:
        env_cfg.scene.num_envs = 1
    env_cfg.sim.device = (
        args_cli.device if args_cli.device is not None else env_cfg.sim.device
    )

    # Extend episode length to match play_length so no mid-play resets
    if args_cli.play_length:
        env_cfg.episode_length_s = args_cli.play_length * env_cfg.decimation * env_cfg.sim.dt + 1.0

    # Apply num_agents and expand observation space for formation
    env_cfg.num_agents = args_cli.num_agents
    if args_cli.num_agents > 1:
        env_cfg.observation_space = 12 + env_cfg.num_neighbors * 3
        env_cfg.scene.env_spacing = 0.01  # drones visually in same space
        env_cfg.collective_resets = False  # no group teleport during play
        env_cfg.formation_centroid = (0.0, 0.0, 1.0)  # hover over origin during play
    else:
        env_cfg.observation_space = 12

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = (
        args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    )
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join(
        "logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"]
    )
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print(
                "[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task."
            )
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path,
            run_dir=f".*_{algorithm}_{args_cli.ml_framework}",
            other_dirs=["checkpoints"],
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(
        args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None
    )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # Override formation shape at play time (--formation arg)
    if args_cli.formation and args_cli.num_agents > 1:
        from ggswarm.formations import get_formation  # noqa: PLC0415

        base = env.unwrapped
        A = base.cfg.num_agents
        spacing = base.cfg.formation_target_spacing
        import math  # noqa: PLC0415
        radius = spacing / (2 * math.sin(math.pi / A))
        new_offsets = get_formation(args_cli.formation, A, radius=radius, spacing=spacing, size=2.0)
        base._formation_offsets = new_offsets.to(base.device)
        print(f"[INFO] Formation override: {args_cli.formation} ({A} agents)")

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording (NVENC H.264 hardware encoder)
    if args_cli.video:
        from ggswarm.viz.nvenc_recorder import NvencRecorder

        video_folder = os.path.join(log_dir, "videos", "play")
        env = NvencRecorder(
            env,
            video_folder=video_folder,
            video_length=args_cli.video_length,
            name_prefix=args_cli.video_prefix,
        )
        print(
            f"[INFO] Recording video to {video_folder} (NVENC H.264, {args_cli.video_length} steps)"
        )

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(
        env, ml_framework=args_cli.ml_framework
    )  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["close_environment_at_exit"] = False
    experiment_cfg["agent"]["experiment"][
        "write_interval"
    ] = 0  # don't log to TensorBoard
    experiment_cfg["agent"]["experiment"][
        "checkpoint_interval"
    ] = 0  # don't generate checkpoints

    if args_cli.policy == "gnn":
        # GNN policy: manually create agent with GATv2 model
        from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
        from skrl.memories.torch import RandomMemory
        from skrl.resources.preprocessors.torch import RunningStandardScaler

        from ggswarm.gnn_policy import GgswarmGNNPolicy

        memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=env.device)
        gnn_model = GgswarmGNNPolicy(
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            num_neighbors=env_cfg.num_neighbors,
            num_agents=args_cli.num_agents,
        )
        GgswarmGNNPolicy.init_edge_cache(memory_size=24, num_envs=env.num_envs)
        models = {"policy": gnn_model, "value": gnn_model}

        ppo_cfg = PPO_DEFAULT_CONFIG.copy()
        ppo_cfg.update({
            "state_preprocessor": RunningStandardScaler,
            "state_preprocessor_kwargs": {"size": env.observation_space, "device": env.device},
            "value_preprocessor": RunningStandardScaler,
            "value_preprocessor_kwargs": {"size": 1, "device": env.device},
        })

        agent = PPO(
            models=models,
            memory=memory,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device,
            cfg=ppo_cfg,
        )

        class _GNNRunner:
            """Minimal runner interface for GNN play compatibility."""
            def __init__(self, a):
                self.agent = a

        runner = _GNNRunner(agent)
        print("[INFO] Using GATv2 GNN policy for play")
    else:
        runner = Runner(env, experiment_cfg)

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    runner.agent.load(resume_path)
    # set agent to evaluation mode
    runner.agent.set_running_mode("eval")

    # Debug: verify preprocessor stats loaded correctly
    if hasattr(runner.agent, '_state_preprocessor') and runner.agent._state_preprocessor is not None:
        sp = runner.agent._state_preprocessor
        if hasattr(sp, 'running_mean'):
            print(f"[DEBUG] state_preprocessor running_mean: {sp.running_mean.mean():.4f}, "
                  f"running_var: {sp.running_variance.mean():.4f}, "
                  f"count: {sp.current_count.item():.0f}")
        else:
            print("[WARN] state_preprocessor has no running_mean — stats may not be loaded!")
    else:
        print("[WARN] No state_preprocessor found on agent!")

    # Trajectory recording buffers
    traj_pos: list[torch.Tensor] = []
    traj_quat: list[torch.Tensor] = []
    traj_goal: list[torch.Tensor] = []

    # reset environment
    obs, _ = env.reset()
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            outputs = runner.agent.act(obs, timestep=0, timesteps=0)
            # - multi-agent (deterministic) actions
            if hasattr(env, "possible_agents"):
                actions = {
                    a: outputs[-1][a].get("mean_actions", outputs[0][a])
                    for a in env.possible_agents
                }
            # - single-agent (deterministic) actions
            else:
                actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping
            obs, _, _, _, _ = env.step(actions)

        # Record trajectory data (first swarm group)
        if args_cli.trajectories:
            base_env = env.unwrapped
            A = base_env.cfg.num_agents
            pos = base_env._robot.data.root_pos_w[:A].detach().cpu().clone()    # [A, 3]
            quat = base_env._robot.data.root_quat_w[:A].detach().cpu().clone()  # [A, 4]
            goal = base_env._desired_pos_w[:A].detach().cpu().clone()            # [A, 3]
            # Mask out dead drones with NaN so plots skip them cleanly
            if base_env.cfg.dropout_enabled:
                dead = ~base_env._agent_alive[:A].cpu()
                pos[dead] = float("nan")
                quat[dead] = float("nan")
                goal[dead] = float("nan")
            traj_pos.append(pos)
            traj_quat.append(quat)
            traj_goal.append(goal)

        timestep += 1
        if args_cli.video and timestep >= args_cli.video_length:
            break
        if args_cli.play_length and timestep >= args_cli.play_length:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Generate trajectory plots
    if args_cli.trajectories and traj_pos:
        euler_fn = None
        try:
            from isaaclab.utils.math import euler_xyz_from_quat

            euler_fn = euler_xyz_from_quat
        except ImportError:
            print(
                "[WARN] euler_xyz_from_quat not available; attitude subplot will be empty."
            )

        from ggswarm.viz.trajectory_plots import generate_trajectory_plots

        traj_dir = os.path.join(log_dir, "trajectories")
        base_env = env.unwrapped
        A = base_env.cfg.num_agents
        env_origins = base_env._terrain.env_origins[:A].cpu() if A > 1 else None
        generate_trajectory_plots(
            traj_pos,
            traj_quat,
            out_dir=traj_dir,
            agent_names=[f"drone_{i}" for i in range(A)],
            euler_fn=euler_fn,
            goal_data=traj_goal if traj_goal else None,
            env_origins=env_origins,
            target_spacing=base_env.cfg.formation_target_spacing if A > 1 else None,
            centroid=base_env.cfg.formation_centroid if A > 1 else None,
            collision_radius=base_env.cfg.collision_radius if A > 1 else None,
        )

        # Save trajectory data as CSV for post-processing
        import csv  # noqa: PLC0415

        pos = torch.stack(traj_pos)  # [T, A, 3]
        goal = torch.stack(traj_goal) if traj_goal else None  # [T, A, 3]
        if env_origins is not None:
            pos = pos - env_origins.unsqueeze(0)
            if goal is not None:
                goal = goal - env_origins.unsqueeze(0)
        T = pos.shape[0]

        csv_path = os.path.join(traj_dir, "trajectory_data.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            header = ["step"]
            for a in range(A):
                header += [f"d{a}_x", f"d{a}_y", f"d{a}_z"]
                if goal is not None:
                    header += [f"d{a}_gx", f"d{a}_gy", f"d{a}_gz"]
            writer.writerow(header)
            for t in range(T):
                row = [t]
                for a in range(A):
                    p = pos[t, a].tolist()
                    row += [f"{p[0]:.4f}", f"{p[1]:.4f}", f"{p[2]:.4f}"]
                    if goal is not None:
                        g = goal[t, a].tolist()
                        row += [f"{g[0]:.4f}", f"{g[1]:.4f}", f"{g[2]:.4f}"]
                writer.writerow(row)
        print(f"[INFO] Trajectory CSV saved: {csv_path}")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
    # Force exit — Isaac Sim viewer can linger after close
    import sys  # noqa: PLC0415, E402

    sys.exit(0)
