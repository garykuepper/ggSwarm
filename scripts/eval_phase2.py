"""Evaluate Phase 2 formation metrics for a trained MAPPO/GATv2 policy."""

from __future__ import annotations

import argparse
import os
import random
import sys
from dataclasses import dataclass

import torch
from isaaclab.app import AppLauncher


@dataclass
class EvalStats:
    formation_error_sum: float = 0.0
    formation_error_count: int = 0
    separation_event_sum: float = 0.0
    separation_event_count: int = 0
    mean_speed_sum: float = 0.0
    mean_speed_count: int = 0
    altitude_error_sum: float = 0.0
    altitude_error_count: int = 0
    ground_hit_rate_sum: float = 0.0
    ground_hit_rate_count: int = 0
    airborne_ratio_sum: float = 0.0
    airborne_ratio_count: int = 0
    # New metrics
    mean_roll_deg_sum: float = 0.0
    mean_roll_deg_count: int = 0
    mean_pitch_deg_sum: float = 0.0
    mean_pitch_deg_count: int = 0
    orientation_violation_rate_sum: float = 0.0
    orientation_violation_rate_count: int = 0
    altitude_std_sum: float = 0.0
    altitude_std_count: int = 0
    episode_survival_steps_sum: float = 0.0
    episode_survival_steps_count: int = 0

    def update(
        self,
        *,
        formation_error: torch.Tensor,
        separation_event_rate: torch.Tensor,
        mean_speed: torch.Tensor,
        mean_altitude_error: torch.Tensor,
        ground_hit_rate: torch.Tensor,
        airborne_ratio: torch.Tensor,
        mean_roll_deg: torch.Tensor,
        mean_pitch_deg: torch.Tensor,
        orientation_violation_rate: torch.Tensor,
        altitude_std: torch.Tensor,
        episode_survival_steps: torch.Tensor,
    ) -> None:
        self.formation_error_sum += float(formation_error.item())
        self.formation_error_count += 1
        self.separation_event_sum += float(separation_event_rate.item())
        self.separation_event_count += 1
        self.mean_speed_sum += float(mean_speed.item())
        self.mean_speed_count += 1
        self.altitude_error_sum += float(mean_altitude_error.item())
        self.altitude_error_count += 1
        self.ground_hit_rate_sum += float(ground_hit_rate.item())
        self.ground_hit_rate_count += 1
        self.airborne_ratio_sum += float(airborne_ratio.item())
        self.airborne_ratio_count += 1
        self.mean_roll_deg_sum += float(mean_roll_deg.item())
        self.mean_roll_deg_count += 1
        self.mean_pitch_deg_sum += float(mean_pitch_deg.item())
        self.mean_pitch_deg_count += 1
        self.orientation_violation_rate_sum += float(orientation_violation_rate.item())
        self.orientation_violation_rate_count += 1
        self.altitude_std_sum += float(altitude_std.item())
        self.altitude_std_count += 1
        self.episode_survival_steps_sum += float(episode_survival_steps.item())
        self.episode_survival_steps_count += 1

    def summarize(self) -> dict[str, float]:
        return {
            "mean_formation_error_m": self.formation_error_sum
            / max(1, self.formation_error_count),
            "separation_event_rate": self.separation_event_sum
            / max(1, self.separation_event_count),
            "mean_speed_mps": self.mean_speed_sum / max(1, self.mean_speed_count),
            "mean_altitude_error_m": self.altitude_error_sum
            / max(1, self.altitude_error_count),
            "ground_hit_rate": self.ground_hit_rate_sum
            / max(1, self.ground_hit_rate_count),
            "airborne_ratio": self.airborne_ratio_sum / max(1, self.airborne_ratio_count),
            "mean_roll_deg": self.mean_roll_deg_sum / max(1, self.mean_roll_deg_count),
            "mean_pitch_deg": self.mean_pitch_deg_sum / max(1, self.mean_pitch_deg_count),
            "orientation_violation_rate": self.orientation_violation_rate_sum
            / max(1, self.orientation_violation_rate_count),
            "altitude_std_m": self.altitude_std_sum / max(1, self.altitude_std_count),
            "mean_episode_survival_steps": self.episode_survival_steps_sum
            / max(1, self.episode_survival_steps_count),
        }


def _pairwise_mean_abs_spacing_error(
    pos_w: torch.Tensor, target_dist: float
) -> torch.Tensor:
    """Compute mean |d_ij - target| over i<j, averaged over envs.

    Args:
        pos_w: Agent world positions.
        target_dist: Desired inter-agent spacing in meters.

    Returns:
        Scalar tensor: mean formation error (meters).
    """
    # shape: [num_envs, num_agents, 3]
    diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
    # shape: [num_envs, num_agents, num_agents]
    dist = torch.norm(diff, dim=-1)
    num_agents = dist.shape[-1]
    mask = torch.triu(
        torch.ones((num_agents, num_agents), device=dist.device, dtype=torch.bool),
        diagonal=1,
    )
    per_env = torch.abs(dist[:, mask] - target_dist).mean(dim=1)
    return per_env.mean()


def _compute_orientation_metrics(
    proj_grav_b: torch.Tensor, orientation_threshold_deg: float = 45.0
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute roll, pitch, and orientation violation rate from projected gravity.

    Args:
        proj_grav_b: Projected gravity in body frame [num_envs, num_agents, 3].
                     When level, z-component is -1.0.
        orientation_threshold_deg: Threshold in degrees for violation detection.

    Returns:
        mean_roll_deg: Mean absolute roll across all steps and agents (scalar).
        mean_pitch_deg: Mean absolute pitch across all steps and agents (scalar).
        violation_rate: Fraction of steps where roll or pitch exceeds threshold (scalar).
    """
    # Extract gravity components in body frame
    # grav_x and grav_y indicate roll/pitch; grav_z indicates how level the drone is
    grav_x = proj_grav_b[:, :, 0]  # [num_envs, num_agents]
    grav_y = proj_grav_b[:, :, 1]
    grav_z = proj_grav_b[:, :, 2]

    # Compute roll and pitch from gravity vector
    # atan2(y, -z) gives roll; atan2(x, -z) gives pitch (in radians)
    roll_rad = torch.atan2(grav_y, -grav_z)
    pitch_rad = torch.atan2(grav_x, -grav_z)

    # Convert to degrees
    roll_deg = torch.abs(roll_rad * 180.0 / torch.pi)
    pitch_deg = torch.abs(pitch_rad * 180.0 / torch.pi)

    # Mean roll and pitch
    mean_roll = roll_deg.mean()
    mean_pitch = pitch_deg.mean()

    # Violation rate: fraction where either roll or pitch exceeds threshold
    threshold_rad = orientation_threshold_deg * torch.pi / 180.0
    violations = (torch.abs(roll_rad) > threshold_rad) | (torch.abs(pitch_rad) > threshold_rad)
    violation_rate = violations.float().mean()

    return mean_roll, mean_pitch, violation_rate


def _compute_altitude_metrics(
    pos_w: torch.Tensor, desired_pos_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute altitude standard deviation and survival steps.

    Args:
        pos_w: Current positions [num_envs, num_agents, 3].
        desired_pos_w: Desired positions [num_envs, num_agents, 3].

    Returns:
        altitude_std: Standard deviation of Z position across all agents (scalar).
        mean_altitude_over_time: Average altitude (scalar).
    """
    altitude = pos_w[:, :, 2]  # [num_envs, num_agents]
    altitude_std = altitude.std()
    mean_altitude = altitude.mean()
    return altitude_std, mean_altitude


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate Phase 2 formation metrics for ggSwarm."
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Template-GGSwarm-Marl-Direct-v0",
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
        "--num_agents",
        type=int,
        default=None,
        help="Override number of agents per environment.",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes.",
    )
    parser.add_argument(
        "--airborne_height_margin",
        type=float,
        default=0.2,
        help="Height margin above min_height to count as airborne/stable.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed for environment reset randomness.",
    )
    parser.add_argument(
        "--gnn",
        action="store_true",
        default=True,
        help="Use GGSwarmGNNPolicy (GATv2).",
    )

    AppLauncher.add_app_launcher_args(parser)
    args_cli, hydra_args = parser.parse_known_args()
    sys.argv = [sys.argv[0]] + hydra_args

    # launch omniverse app
    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import skrl
    from packaging import version

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

    if args_cli.ml_framework.startswith("torch"):
        from skrl.utils.runner.torch import Runner
    else:
        raise RuntimeError("Only torch eval is supported for now")

    # config shortcuts: use MAPPO entry point by default
    algorithm = args_cli.algorithm.lower()
    agent_cfg_entry_point = (
        "skrl_cfg_entry_point"
        if algorithm in ["ppo"]
        else f"skrl_{algorithm}_cfg_entry_point"
    )

    @hydra_task_config(args_cli.task, agent_cfg_entry_point)
    def _eval(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, cfg: dict):
        # Override env sizing
        if args_cli.num_envs is not None:
            env_cfg.scene.num_envs = args_cli.num_envs
        env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
        if hasattr(env_cfg, "num_agents") and args_cli.num_agents is not None:
            env_cfg.num_agents = args_cli.num_agents
            env_cfg.possible_agents = [f"drone_{i}" for i in range(env_cfg.num_agents)]
            env_cfg.action_spaces = {agent: 4 for agent in env_cfg.possible_agents}
            env_cfg.observation_spaces = {agent: 12 for agent in env_cfg.possible_agents}

        # Seed handling
        if args_cli.seed == -1:
            args_cli.seed = random.randint(0, 10000)
        cfg["seed"] = args_cli.seed
        env_cfg.seed = cfg["seed"]

        # Configure model class for evaluation
        if args_cli.gnn:
            cfg["models"]["policy"]["class"] = "GGSwarmGNNPolicy"
            if "network" in cfg["models"]["policy"]:
                del cfg["models"]["policy"]["network"]

            original_component = Runner._component

            def custom_component(self, name: str):
                if name.lower() == "ggswarmgnnpolicy":
                    from ggSwarm.tasks.direct.ggswarm_marl.agents.skrl_gnn_policy import (
                        GGSwarmGNNPolicy,
                    )

                    return GGSwarmGNNPolicy
                return original_component(self, name)

            Runner._component = custom_component

        # Resolve checkpoint
        task_name = args_cli.task.split(":")[-1]
        train_task_name = task_name.replace("-Play", "")
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
        run_name = os.path.basename(
            os.path.dirname(os.path.dirname(resume_path)).rstrip("/\\")
        )

        # Ensure eval writer outputs (if any) stay under logs/skrl.
        cfg["agent"]["experiment"]["directory"] = log_root_path
        cfg["agent"]["experiment"]["experiment_name"] = run_name

        # Create isaac environment
        env = gym.make(args_cli.task, cfg=env_cfg)
        if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
            env = multi_agent_to_single_agent(env)

        # Keep a handle to the base env for metrics
        base_env = env.unwrapped

        # Wrap for skrl
        env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)

        # No checkpoint generation during eval
        cfg["trainer"]["checkpoint_interval"] = 0

        runner = Runner(env, cfg)
        agent = getattr(runner, "agent", None)
        # Some SKRL runners expose a per-agent list instead of a single `agent`
        agents_attr = getattr(runner, "agents", None)
        if agent is None and isinstance(agents_attr, (list, tuple)) and agents_attr:
            agent = agents_attr[0]
        if agent is None:
            raise RuntimeError("Could not resolve skrl agent for evaluation")

        checkpoint = torch.load(resume_path, map_location=agent.device)

        def _load_policy_state_dict(state_dict: dict, source: str) -> bool:
            """Load a policy `state_dict` into the right network module.

            skrl's `Runner`/agent objects don't always expose the policy as
            `agent.policy`. We try common locations and load into the first
            candidate that accepts the provided keys.
            """
            candidates: list[object] = []

            def _maybe_add(x: object) -> None:
                if isinstance(x, torch.nn.Module):
                    candidates.append(x)

            if hasattr(agent, "policy"):
                _maybe_add(getattr(agent, "policy"))

            if hasattr(agent, "models"):
                models = getattr(agent, "models")
                if isinstance(models, dict):
                    policy_like = [
                        m for k, m in models.items() if "policy" in str(k).lower()
                    ]
                    for m in (policy_like if policy_like else list(models.values())):
                        _maybe_add(m)
                else:
                    if hasattr(models, "policy"):
                        _maybe_add(getattr(models, "policy"))

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
                    print(f"[EVAL] Policy state dict loaded from: {source}")
                    return True
                except Exception:
                    continue
            return False

        if "policy" in checkpoint:
            if not _load_policy_state_dict(checkpoint["policy"], "checkpoint['policy']"):
                raise KeyError("Could not load policy from checkpoint['policy']")
        elif "policy_0" in checkpoint:
            if not _load_policy_state_dict(
                checkpoint["policy_0"], "checkpoint['policy_0']"
            ):
                raise KeyError("Could not load policy from checkpoint['policy_0']")
        else:
            loaded = False
            if isinstance(checkpoint, dict):
                for k, v in checkpoint.items():
                    if isinstance(v, dict) and "policy" in v:
                        loaded = _load_policy_state_dict(
                            v["policy"], f"checkpoint['{k}']['policy']"
                        )
                        if loaded:
                            break
            if not loaded:
                raise KeyError(
                    f"Could not find policy in checkpoint. Available keys: {list(checkpoint.keys())}"
                )

        agent.set_running_mode("eval")

        # Determine episode length in steps
        max_episode_length = getattr(base_env, "max_episode_length", None)
        if max_episode_length is None:
            raise RuntimeError("Could not determine max_episode_length from env")

        total_steps = int(args_cli.num_episodes) * int(max_episode_length)

        stats = EvalStats()
        obs, _ = env.reset()
        steps = 0
        altitude_history = []
        episode_step_count = 0

        while simulation_app.is_running() and steps < total_steps:
            with torch.inference_mode():
                outputs = agent.act(obs, timestep=0, timesteps=0)
                if hasattr(env, "possible_agents") and hasattr(base_env, "possible_agents"):
                    actions = {
                        a: outputs[-1][a].get("mean_actions", outputs[0][a])
                        for a in base_env.possible_agents
                    }
                else:
                    actions = outputs[-1].get("mean_actions", outputs[0])

                obs, _, _, _, _ = env.step(actions)

                # Metrics from base env tensors
                # shape: [num_envs, num_agents, 3]
                pos_w = base_env.robot.data.root_pos_w.view(
                    base_env.num_envs, base_env.cfg.num_agents, 3
                )
                formation_error = _pairwise_mean_abs_spacing_error(
                    pos_w, float(base_env.cfg.target_formation_dist)
                )

                diff = pos_w.unsqueeze(2) - pos_w.unsqueeze(1)
                dist = torch.norm(diff, dim=-1)
                eye = torch.eye(base_env.cfg.num_agents, device=dist.device).unsqueeze(0)
                separation_event_rate = (
                    ((dist < float(base_env.cfg.min_separation_dist)) & (1.0 - eye).bool())
                    .any(dim=2)
                    .any(dim=1)
                    .float()
                    .mean()
                )

                # shape: [num_envs, num_agents, 3]
                lin_vel_b = base_env.robot.data.root_lin_vel_b.view(
                    base_env.num_envs, base_env.cfg.num_agents, 3
                )
                mean_speed = torch.norm(lin_vel_b, dim=-1).mean()

                # shape: [num_envs, num_agents]
                desired_altitude = base_env._desired_pos_w[:, :, 2]
                current_altitude = pos_w[:, :, 2]
                mean_altitude_error = torch.abs(current_altitude - desired_altitude).mean()
                ground_hit_rate = (current_altitude < float(base_env.cfg.min_height)).any(
                    dim=1
                ).float().mean()
                airborne_threshold = float(base_env.cfg.min_height) + float(
                    args_cli.airborne_height_margin
                )
                airborne_ratio = (current_altitude > airborne_threshold).float().mean()

                # New orientation metrics
                proj_grav_b = base_env.robot.data.projected_gravity_b.view(
                    base_env.num_envs, base_env.cfg.num_agents, 3
                )
                mean_roll_deg, mean_pitch_deg, orientation_violation_rate = (
                    _compute_orientation_metrics(proj_grav_b)
                )

                # Altitude std deviation tracking
                altitude_std, _ = _compute_altitude_metrics(pos_w, base_env._desired_pos_w)
                altitude_history.append(current_altitude.cpu())

                # Track episode survival (steps until termination)
                episode_step_count += 1
                if episode_step_count >= max_episode_length or ground_hit_rate > 0:
                    episode_survival = torch.tensor(episode_step_count, dtype=torch.float32)
                    episode_step_count = 0
                else:
                    episode_survival = torch.tensor(episode_step_count, dtype=torch.float32)

                stats.update(
                    formation_error=formation_error,
                    separation_event_rate=separation_event_rate,
                    mean_speed=mean_speed,
                    mean_altitude_error=mean_altitude_error,
                    ground_hit_rate=ground_hit_rate,
                    airborne_ratio=airborne_ratio,
                    mean_roll_deg=mean_roll_deg,
                    mean_pitch_deg=mean_pitch_deg,
                    orientation_violation_rate=orientation_violation_rate,
                    altitude_std=altitude_std,
                    episode_survival_steps=episode_survival,
                )

            steps += 1
            if steps % max(1, total_steps // 10) == 0:
                s = stats.summarize()
                print(
                    "[EVAL] "
                    f"step={steps}/{total_steps} "
                    f"mean_formation_error={s['mean_formation_error_m']:.3f}m "
                    f"separation_event_rate={s['separation_event_rate']:.3f} "
                    f"mean_speed={s['mean_speed_mps']:.3f}m/s "
                    f"mean_altitude_error={s['mean_altitude_error_m']:.3f}m "
                    f"ground_hit_rate={s['ground_hit_rate']:.3f} "
                    f"airborne_ratio={s['airborne_ratio']:.3f} "
                    f"mean_roll={s['mean_roll_deg']:.1f}° "
                    f"mean_pitch={s['mean_pitch_deg']:.1f}° "
                    f"orientation_violation={s['orientation_violation_rate']:.3f}"
                )

        summary = stats.summarize()
        print("\n=== Phase 2 Evaluation Summary ===")
        print(f"checkpoint: {resume_path}")
        print(f"episodes: {args_cli.num_episodes}")
        print(f"mean_formation_error_m: {summary['mean_formation_error_m']:.6f}")
        print(f"separation_event_rate: {summary['separation_event_rate']:.6f}")
        print(f"mean_speed_mps: {summary['mean_speed_mps']:.6f}")
        print(f"mean_altitude_error_m: {summary['mean_altitude_error_m']:.6f}")
        print(f"ground_hit_rate: {summary['ground_hit_rate']:.6f}")
        print(f"airborne_ratio: {summary['airborne_ratio']:.6f}")
        print(f"\n=== NEW: Orientation & Stability Metrics ===")
        print(f"mean_roll_deg: {summary['mean_roll_deg']:.3f}")
        print(f"mean_pitch_deg: {summary['mean_pitch_deg']:.3f}")
        print(f"orientation_violation_rate: {summary['orientation_violation_rate']:.6f}")
        print(f"altitude_std_m: {summary['altitude_std_m']:.6f}")
        print(f"mean_episode_survival_steps: {summary['mean_episode_survival_steps']:.1f}")

        env.close()

    _eval()
    simulation_app.close()


if __name__ == "__main__":
    main()

