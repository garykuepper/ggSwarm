"""Evaluate hover stability metrics for the `GGS-Hover-v0` task."""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass

import torch
from isaaclab.app import AppLauncher


@dataclass
class HoverEvalStats:
    """Running averages for hover evaluation metrics."""

    airborne_ratio_sum: float = 0.0
    airborne_ratio_count: int = 0
    altitude_error_sum: float = 0.0
    altitude_error_count: int = 0
    position_error_sum: float = 0.0
    position_error_count: int = 0
    mean_speed_sum: float = 0.0
    mean_speed_count: int = 0
    mean_ang_speed_sum: float = 0.0
    mean_ang_speed_count: int = 0
    ground_hit_rate_sum: float = 0.0
    ground_hit_rate_count: int = 0

    def update(
        self,
        *,
        airborne_ratio: torch.Tensor,
        altitude_error: torch.Tensor,
        position_error: torch.Tensor,
        mean_speed: torch.Tensor,
        mean_ang_speed: torch.Tensor,
        ground_hit_rate: torch.Tensor,
    ) -> None:
        self.airborne_ratio_sum += float(airborne_ratio.item())
        self.airborne_ratio_count += 1
        self.altitude_error_sum += float(altitude_error.item())
        self.altitude_error_count += 1
        self.position_error_sum += float(position_error.item())
        self.position_error_count += 1
        self.mean_speed_sum += float(mean_speed.item())
        self.mean_speed_count += 1
        self.mean_ang_speed_sum += float(mean_ang_speed.item())
        self.mean_ang_speed_count += 1
        self.ground_hit_rate_sum += float(ground_hit_rate.item())
        self.ground_hit_rate_count += 1

    def summarize(self) -> dict[str, float]:
        return {
            "airborne_ratio": self.airborne_ratio_sum / max(1, self.airborne_ratio_count),
            "mean_altitude_error_m": self.altitude_error_sum
            / max(1, self.altitude_error_count),
            "mean_position_error_m": self.position_error_sum
            / max(1, self.position_error_count),
            "mean_speed_mps": self.mean_speed_sum / max(1, self.mean_speed_count),
            "mean_ang_speed_rps": self.mean_ang_speed_sum / max(1, self.mean_ang_speed_count),
            "ground_hit_rate": self.ground_hit_rate_sum / max(1, self.ground_hit_rate_count),
        }


def main() -> None:
    """Run hover evaluation with deterministic metrics output."""
    parser = argparse.ArgumentParser(
        description="Evaluate hover metrics for ggSwarm hover baseline."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="GGS-Hover-v0",
        help="Gym task ID to evaluate.",
    )
    parser.add_argument(
        "--algorithm",
        type=str,
        default="MAPPO",
        choices=["PPO", "IPPO", "MAPPO"],
        help="Algorithm used for training (affects checkpoint discovery).",
    )
    parser.add_argument(
        "--ml_framework",
        type=str,
        default="torch",
        choices=["torch", "jax", "jax-numpy"],
        help="The ML framework used for training the skrl agent.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (.pt). If omitted, auto-discovers latest.",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Number of parallel environments to simulate.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for environment reset randomness.",
    )

    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import skrl
    from packaging import version
    from skrl.utils.runner.torch import Runner

    from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg
    from isaaclab.envs import ManagerBasedRLEnvCfg, multi_agent_to_single_agent
    from isaaclab_rl.skrl import SkrlVecEnvWrapper

    import ggSwarm.tasks  # noqa: F401
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.utils import get_checkpoint_path
    from isaaclab_tasks.utils.hydra import hydra_task_config

    if version.parse(skrl.__version__) < version.parse("1.4.3"):
        raise RuntimeError(
            f"Unsupported skrl version: {skrl.__version__}. Install skrl>=1.4.3"
        )

    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = (
        "skrl_cfg_entry_point"
        if algorithm in ["ppo"]
        else f"skrl_{algorithm}_cfg_entry_point"
    )

    @hydra_task_config(args_cli.task, agent_cfg_entry_point)
    def _eval(
        env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
        cfg: dict,
    ) -> None:
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = (
            args_cli.device if args_cli.device is not None else env_cfg.sim.device
        )

        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        cfg["seed"] = args_cli.seed
        env_cfg.seed = cfg["seed"]

        log_root_path = os.path.join("logs", "skrl", cfg["agent"]["experiment"]["directory"])
        log_root_path = os.path.abspath(log_root_path)
        if args_cli.checkpoint is not None:
            resume_path = os.path.abspath(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(
                log_root_path,
                run_dir=f".*_{algorithm}_{args_cli.ml_framework}",
                other_dirs=["checkpoints"],
            )

        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
            env = multi_agent_to_single_agent(env)
        base_env = env.unwrapped
        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

        cfg["trainer"]["checkpoint_interval"] = 0
        runner = Runner(env, cfg)
        agent = getattr(runner, "agent", None)
        if agent is None:
            agents_attr = getattr(runner, "agents", None)
            if isinstance(agents_attr, (list, tuple)) and agents_attr:
                agent = agents_attr[0]
        if agent is None:
            raise RuntimeError("Could not resolve skrl agent for evaluation")

        checkpoint = torch.load(resume_path, map_location=agent.device)
        agent.load(resume_path) if hasattr(agent, "load") else None

        if isinstance(checkpoint, dict) and "policy" in checkpoint and hasattr(
            agent, "policy"
        ):
            agent.policy.load_state_dict(checkpoint["policy"])

        agent.set_running_mode("eval")
        max_episode_length = getattr(base_env, "max_episode_length", None)
        if max_episode_length is None:
            raise RuntimeError("Could not determine max_episode_length from env")

        total_steps = int(args_cli.num_episodes) * int(max_episode_length)
        stats = HoverEvalStats()
        obs, _ = env.reset()
        steps = 0

        while simulation_app.is_running() and steps < total_steps:
            with torch.inference_mode():
                outputs = agent.act(obs, timestep=0, timesteps=0)
                if hasattr(base_env, "possible_agents"):
                    actions = {
                        a: outputs[-1][a].get("mean_actions", outputs[0][a])
                        for a in base_env.possible_agents
                    }
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])

                obs, _, _, _, _ = env.step(actions)

                # shape: [num_envs, num_agents, 3]
                pos_w = base_env.robot.data.root_pos_w.view(
                    base_env.num_envs, base_env.cfg.num_agents, 3
                )
                # shape: [num_envs, num_agents, 3]
                lin_vel_b = base_env.robot.data.root_lin_vel_b.view(
                    base_env.num_envs, base_env.cfg.num_agents, 3
                )
                # shape: [num_envs, num_agents, 3]
                ang_vel_b = base_env.robot.data.root_ang_vel_b.view(
                    base_env.num_envs, base_env.cfg.num_agents, 3
                )
                # shape: [num_envs, num_agents, 3]
                desired_pos_w = base_env._desired_pos_w

                # shape: [num_envs, num_agents]
                z = pos_w[:, :, 2]
                airborne_ratio = (z >= float(base_env.cfg.hover_reward_min_height)).float().mean()
                altitude_error = torch.abs(desired_pos_w[:, :, 2] - z).mean()
                position_error = torch.norm(desired_pos_w - pos_w, dim=-1).mean()
                mean_speed = torch.norm(lin_vel_b, dim=-1).mean()
                mean_ang_speed = torch.norm(ang_vel_b, dim=-1).mean()
                ground_hit_rate = (z < float(base_env.cfg.ground_hit_height)).float().mean()

                stats.update(
                    airborne_ratio=airborne_ratio,
                    altitude_error=altitude_error,
                    position_error=position_error,
                    mean_speed=mean_speed,
                    mean_ang_speed=mean_ang_speed,
                    ground_hit_rate=ground_hit_rate,
                )

            steps += 1

        summary = stats.summarize()
        pass_airborne = summary["airborne_ratio"] > 0.95
        pass_ground = summary["ground_hit_rate"] < 0.01
        pass_hover = summary["mean_altitude_error_m"] < 0.20
        overall_pass = pass_airborne and pass_ground and pass_hover

        print("\n=== Hover Evaluation Summary ===")
        print(f"checkpoint: {resume_path}")
        print(f"episodes: {args_cli.num_episodes}")
        print(f"airborne_ratio: {summary['airborne_ratio']:.6f}")
        print(f"mean_altitude_error_m: {summary['mean_altitude_error_m']:.6f}")
        print(f"mean_position_error_m: {summary['mean_position_error_m']:.6f}")
        print(f"mean_speed_mps: {summary['mean_speed_mps']:.6f}")
        print(f"mean_ang_speed_rps: {summary['mean_ang_speed_rps']:.6f}")
        print(f"ground_hit_rate: {summary['ground_hit_rate']:.6f}")
        print(f"pass: {overall_pass}")

        env.close()

    _eval()
    simulation_app.close()


if __name__ == "__main__":
    main()
