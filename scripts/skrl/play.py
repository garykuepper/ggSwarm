# Copyright (c) 2022-2026, ggSwarm Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

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
import os
import sys
import traceback

LOG_PATH = r"c:\Users\gkuep\Code\isaaclab\ggSwarm\play_error.log"

def log_error(exctype, value, tb):
    try:
        with open(LOG_PATH, "w") as f:
            traceback.print_exception(exctype, value, tb, file=f)
    except Exception:
        pass
    sys.__excepthook__(exctype, value, tb)

sys.excepthook = log_error

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Play a checkpoint of an RL agent from skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Stop after N environment steps (debugging/sanity check).",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument(
    "--num_agents",
    type=int,
    default=None,
    help="Number of agents (override env config). Use 4 to match training if trained with 4 agents.",
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
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
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
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument(
    "--gnn",
    action="store_true",
    default=False,
    help="Use the custom GATv2 GNN policy instead of the default MLP.",
)
parser.add_argument(
    "--hover_debug",
    action="store_true",
    default=False,
    help=(
        "Bypass the policy and send a constant hover-like action "
        "(action_z=0, torques=0). Useful to sanity-check force application."
    ),
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
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

import ggSwarm.tasks  # noqa: F401

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict

from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
if args_cli.agent is None:
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"
else:
    agent_cfg_entry_point = args_cli.agent
    algorithm = agent_cfg_entry_point.split("_cfg")[0].split("skrl_")[-1].lower()


@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, experiment_cfg: dict):
    """Play with skrl agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if hasattr(env_cfg, "num_agents") and args_cli.num_agents is not None:
        env_cfg.num_agents = args_cli.num_agents
        env_cfg.possible_agents = [f"drone_{i}" for i in range(env_cfg.num_agents)]
        env_cfg.action_spaces = {agent: 4 for agent in env_cfg.possible_agents}
        env_cfg.observation_spaces = {agent: 12 for agent in env_cfg.possible_agents}

    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

        # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    experiment_cfg["seed"] = args_cli.seed if args_cli.seed is not None else experiment_cfg["seed"]
    env_cfg.seed = experiment_cfg["seed"]

    # specify directory for logging experiments (load checkpoint)
    log_root_path = os.path.join("logs", "skrl", experiment_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    # get checkpoint path
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("skrl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = os.path.abspath(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(
            log_root_path, run_dir=f".*_{algorithm}_{args_cli.ml_framework}", other_dirs=["checkpoints"]
        )
    log_dir = os.path.dirname(os.path.dirname(resume_path))

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # get environment (step) dt for real-time evaluation
    try:
        dt = env.step_dt
    except AttributeError:
        dt = env.unwrapped.step_dt

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    experiment_cfg["trainer"]["checkpoint_interval"] = 0  # don't generate checkpoints

    # Override model with GNN if requested
    if args_cli.gnn:
        print("[INFO] Using GGSwarmGNNPolicy (GATv2) for playback.")
        experiment_cfg["models"]["policy"]["class"] = "GGSwarmGNNPolicy"
        # Disable the default 'network' key as GGSwarmGNNPolicy defines its own architecture
        if "network" in experiment_cfg["models"]["policy"]:
            del experiment_cfg["models"]["policy"]["network"]

        # Monkey-patch SKRL Runner to recognize the custom class string
        original_component = Runner._component

        def custom_component(self, name: str):
            if name.lower() == "ggswarmgnnpolicy":
                from ggSwarm.tasks.direct.ggswarm_marl.agents.skrl_gnn_policy import GGSwarmGNNPolicy

                return GGSwarmGNNPolicy
            return original_component(self, name)

        Runner._component = custom_component

    runner = Runner(env, experiment_cfg)

    def bcb(msg):
        with open(r"c:\Users\gkuep\Code\isaaclab\ggSwarm\breadcrumbs.txt", "a") as f:
            f.write(f"{time.time()}: {msg}\n")
            f.flush()
            os.fsync(f.fileno())
    
    bcb("Runner created")

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    # Get the agent(s) from the runner
    # skrl runners can have 'agent' (single-agent) or 'agents' (multi-agent)
    agent = getattr(runner, "agent", None)
    if agent is None and hasattr(runner, "agents"):
        agent = runner.agents[0] # assuming shared policy for all agents in MAPPO

    if agent is None:
        print("[ERROR] Could not find agent in runner. Dumping attributes to play_error.log")
        with open(LOG_PATH, "a") as f:
            f.write(f"\nRunner Attributes: {dir(runner)}\n")
            if hasattr(runner, "agents"):
                f.write(f"Runner.agents: {runner.agents}\n")
        # Let it fail with the original attribute error so we get a traceback too
        agent = runner.agent 

    bcb("Loading checkpoint file")
    # Load checkpoint
    checkpoint = torch.load(resume_path, map_location=agent.device)
    bcb("Checkpoint file loaded")
    def _load_policy_state_dict(state_dict: dict, source: str) -> bool:
        """Load a policy `state_dict` into the right network module.

        skrl's `Runner`/agent objects don't always expose the policy as `agent.policy`.
        We try common locations and load into the first candidate that accepts
        the provided keys.
        """
        candidates: list[object] = []

        def _maybe_add(x: object):
            if isinstance(x, torch.nn.Module):
                candidates.append(x)

        # Most direct: agent.policy
        if hasattr(agent, "policy"):
            _maybe_add(getattr(agent, "policy"))

        # Common: agent.models["policy"]
        if hasattr(agent, "models"):
            models = getattr(agent, "models")
            if isinstance(models, dict):
                # Prefer keys that look like policy, but fall back to all modules in the dict.
                policy_like = [m for k, m in models.items() if "policy" in str(k).lower()]
                for m in (policy_like if policy_like else list(models.values())):
                    _maybe_add(m)
            else:
                if hasattr(models, "policy"):
                    _maybe_add(getattr(models, "policy"))

        # Last resort: scan any torch.nn.Modules stored on the agent object.
        for name in dir(agent):
            try:
                v = getattr(agent, name)
            except Exception:
                continue
            if isinstance(v, torch.nn.Module):
                _maybe_add(v)
            elif isinstance(v, dict):
                for vv in v.values():
                    _maybe_add(vv)

        for m in candidates:
            try:
                m.load_state_dict(state_dict)
                bcb(f"Policy state dict loaded from: {source}")
                return True
            except Exception:
                continue
        return False

    if "policy" in checkpoint:
        if not _load_policy_state_dict(checkpoint["policy"], "checkpoint['policy']"):
            raise KeyError("Could not load policy from checkpoint['policy']")
    elif "policy_0" in checkpoint:  # some multi-agent formats
        if not _load_policy_state_dict(checkpoint["policy_0"], "checkpoint['policy_0']"):
            raise KeyError("Could not load policy from checkpoint['policy_0']")
    else:
        # skrl multi-agent checkpoints may be stored per-agent, e.g.
        # {"drone_0": {"policy": <state_dict>, ...}, "drone_1": {...}, ...}
        loaded = False
        if isinstance(checkpoint, dict):
            for k, v in checkpoint.items():
                if isinstance(v, dict) and "policy" in v:
                    loaded = _load_policy_state_dict(v["policy"], f"checkpoint['{k}']['policy']")
                    if loaded:
                        break

        if not loaded:
            # Diagnostic: Print all keys if no known policy key is found
            bcb(f"MISSING POLICY KEY. Keys: {list(checkpoint.keys())}")
            # Attempt to find ANY key containing 'policy'
            policy_key = next((k for k in checkpoint.keys() if "policy" in k.lower()), None)
            if policy_key and isinstance(checkpoint[policy_key], dict):
                loaded = _load_policy_state_dict(checkpoint[policy_key], f"checkpoint['{policy_key}']")
            if not loaded:
                bcb("No policy-like keys found! Stopping to prevent Critic mismatch.")
                raise KeyError(f"Could not find policy in checkpoint. Available keys: {list(checkpoint.keys())}")

    bcb("Finalizing policy load")
    print("[INFO] Policy loaded successfully (Value function skipped for scalability).")

    # MAPPO with separate=True creates one agent per drone; we only loaded into agents[0].
    # Broadcast the loaded policy to all agents so drones 1..N use the same trained weights
    # (e.g. when playing with 10 agents but trained with 4).
    if hasattr(runner, "agents") and len(runner.agents) > 1:
        def _get_policy_module(a):
            if hasattr(a, "policy") and isinstance(getattr(a, "policy"), torch.nn.Module):
                return getattr(a, "policy")
            if hasattr(a, "models") and isinstance(getattr(a, "models"), dict):
                for k, m in getattr(a, "models").items():
                    if "policy" in str(k).lower() and isinstance(m, torch.nn.Module):
                        return m
            return None

        src_policy = _get_policy_module(agent)
        if src_policy is not None:
            state_dict = src_policy.state_dict()
            for other in runner.agents[1:]:
                dst_policy = _get_policy_module(other)
                if dst_policy is not None:
                    try:
                        dst_policy.load_state_dict(state_dict)
                    except Exception:
                        pass  # skip if architecture differs
            print(f"[INFO] Broadcast policy to {len(runner.agents)} agents (trained policy now used by all drones).")

    # set agent(s) to evaluation mode
    agent.set_running_mode("eval")
    if hasattr(runner, "agents"):
        for a in runner.agents[1:]:
            a.set_running_mode("eval")
    bcb("Evaluation mode set")

    # reset environment
    bcb("Resetting environment")
    obs, _ = env.reset()
    bcb("Environment reset complete")
    timestep = 0
    total_steps = args_cli.video_length if args_cli.video else (args_cli.max_steps or 0)
    print("[INFO] Starting simulation loop (Ctrl+C to stop). First step with --video can take 30–60s.")
    if args_cli.hover_debug:
        print("[INFO] hover_debug enabled: bypassing policy actions (action_z=0, torques=0).")
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()

        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            if args_cli.hover_debug:
                # action shape contract: each agent expects [num_envs, 4]
                hover_action = torch.zeros(
                    env.num_envs,
                    4,
                    device=env.device if hasattr(env, "device") else agent.device,
                    dtype=torch.float32,
                )
                actions = {a: hover_action for a in env.possible_agents} if hasattr(env, "possible_agents") else hover_action
            elif hasattr(env, "possible_agents") and hasattr(runner, "agents") and len(runner.agents) > 1:
                # MAPPO separate agents: each agent produces actions for its drone
                actions = {}
                for i, ag in enumerate(runner.agents):
                    if i < len(env.possible_agents):
                        out = ag.act(obs, timestep=0, timesteps=0)
                        aid = env.possible_agents[i]
                        last = out[-1]
                        if isinstance(last, dict) and aid in last:
                            actions[aid] = last[aid].get("mean_actions", last[aid]) if isinstance(last[aid], dict) else last[aid]
                        elif isinstance(last, dict):
                            actions[aid] = last.get("mean_actions", out[0])
                        else:
                            actions[aid] = last if isinstance(last, torch.Tensor) else out[0]
            else:
                outputs = agent.act(obs, timestep=0, timesteps=0)
                # - multi-agent (single agent produces all actions)
                if hasattr(env, "possible_agents"):
                    actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in env.possible_agents}
                # - single-agent (deterministic) actions
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])
            # env stepping (with --video, first step triggers render and can be slow)
            obs, _, _, _, _ = env.step(actions)
        timestep += 1

        # progress so user knows it's not hung
        if timestep == 1:
            print("[INFO] Step 1 done (video capture active).")
        elif total_steps > 0 and timestep % max(1, total_steps // 10) == 0:
            print(f"[INFO] Step {timestep} / {total_steps}")

        # exit the play loop after recording one video
        if args_cli.video and timestep == args_cli.video_length:
            break

        # optional hard stop for non-video runs
        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
