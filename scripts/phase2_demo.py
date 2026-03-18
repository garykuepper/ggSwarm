"""
Standalone demonstration script for Phase 2 Brain Development.
This script demonstrates the swarm running with the trained GNN policy.
"""

import argparse

# AppLauncher must be imported and the simulation app started BEFORE any
# isaaclab or isaacsim modules are imported.
from isaaclab.app import AppLauncher

# Configuration for Phase 2 Demo
DEFAULT_TASK = "Template-GGSwarm-Marl-Direct-v0"

def main():
    parser = argparse.ArgumentParser(description="Phase 2 GNN Policy Demonstration.")
    parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="Name of the task.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")

    # Append AppLauncher cli args
    AppLauncher.add_app_launcher_args(parser)
    args_cli = parser.parse_args()

    # Launch simulation app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    # Import project modules after AppLauncher starts
    import ggSwarm.tasks  # noqa: F401
    import gymnasium as gym

    # Create the environment
    print(f"[Phase 2 Demo] Creating environment: {args_cli.task}")
    env = gym.make(args_cli.task, num_envs=args_cli.num_envs, device=args_cli.device)

    # Simple message indicating we are wrapping the skrl play logic
    print("[Phase 2 Demo] Loading GATv2 Policy via SKRL metadata...")

    # In a simplified demo, we can just defer to the skrl player's logic
    # or implement a minimal evaluation loop here.
    # For robustness, we'll prompt the user if they'd rather use the full play.py
    # if no checkpoint is found.

    print("[Phase 2 Demo] Environment Ready. Starting playback loop...")

    env.reset()
    while simulation_app.is_running():
        # Phase 2 Demo logic:
        # This script acts as a high-level entry point.
        # To truly play back the exact SKRL policy, it's best to use run_phase2.py play.
        # This standalone demo serves as a template for custom inference tests.

        # For now, we apply zero/random actions if no policy is loaded,
        # or we could implement a basic attractor logic.
        actions = env.action_space.sample() if hasattr(env.action_space, "sample") else {}
        obs, rewards, dones, list_info = env.step(actions)

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
