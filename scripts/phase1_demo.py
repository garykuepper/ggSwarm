# Copyright (c) 2022-2026, ggSwarm Developers.
# All rights reserved.
#
# SPDX-License-Identifier: MIT

"""Script to an environment with random action agent."""

import argparse

import ggSwarm.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

from isaaclab.app import AppLauncher

import isaaclab_tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

# add argparse arguments
parser = argparse.ArgumentParser(description="Random agent for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=None,
    help="Number of environments to simulate.",
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


def main():
    """Random actions agent with Isaac Lab environment."""
    print("[DEBUG] random_agent.py main start")
    # create environment configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    print(f"[DEBUG] Creating environment: {args_cli.task}")
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)
    print("[DEBUG] Environment created")

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()
    # simulate environment
    while simulation_app.is_running():
        # run everything in inference mode
        with torch.inference_mode():
            # sample actions
            unwrapped_env = env.unwrapped
            if hasattr(unwrapped_env, "possible_agents"):
                # Multi-agent handling
                actions = {}
                for agent in unwrapped_env.possible_agents:
                    # action_space(agent) is a bound method in DirectMARLEnv
                    space = unwrapped_env.action_space(agent)
                    actions[agent] = (
                        2
                        * torch.rand(
                            (unwrapped_env.num_envs, *space.shape),
                            device=unwrapped_env.device,
                        )
                        - 1
                    )
            else:
                # Single-agent handling
                actions = (
                    2
                    * torch.rand(
                        (unwrapped_env.num_envs, *env.action_space.shape),
                        device=unwrapped_env.device,
                    )
                    - 1
                )
            # apply actions
            obs, rewards, dones, truncated, info = env.step(actions)

            # Extract and print the Adjacency Matrix as evidence for Phase 1
            if "adj_matrix" in info:
                # Get the matrix for the first environment instance
                adj = info["adj_matrix"][0].cpu().numpy()
                # Subtraction of self-loops is no longer needed as they are removed in the env
                num_connected = (adj > 0).sum()
                print(f"[Phase 1 Evidence] Drones connected (Distance < 2.0m): {num_connected // 2} pairs")
                # Uncomment to print the full matrix:
                # print(adj)
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
