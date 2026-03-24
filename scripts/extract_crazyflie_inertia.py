"""Extract Crazyflie moment of inertia from Isaac Lab sim and compute PD gains.

Boots a single-env headless sim, reads the PhysX inertia tensor from the root
body, and prints recommended PD attitude controller gains for critical or
slightly overdamped response.

Usage:
    python scripts/extract_crazyflie_inertia.py --headless \
        --task Template-GGSwarm-Marl-HoverStability-v0 --num_envs 1
"""

from __future__ import annotations

import argparse
import math
import sys

import torch
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--task",
    type=str,
    default="Template-GGSwarm-Marl-HoverStability-v0",
    help="Gym task id (default: hover-stability).",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Parallel envs (1 is sufficient for inertia extraction).",
)

AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402

import ggSwarm.tasks  # noqa: F401, E402
import isaaclab_tasks  # noqa: F401, E402
from isaaclab.envs import DirectMARLEnv  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402


def _compute_gains(
    I: float,
    kp: float,
    max_tilt: float,
    target_zeta: float,
    max_moment_budget: float | None = None,
) -> dict[str, float]:
    """Compute PD gains for a target damping ratio.

    Args:
        I: Moment of inertia (kg*m^2).
        kp: Proportional gain (Nm/rad).
        max_tilt: Maximum tilt angle (rad).
        target_zeta: Target damping ratio (1.0 = critical, >1.0 = overdamped).
        max_moment_budget: If provided, verify total moment fits within budget.

    Returns:
        Dict with computed gain values and dynamics.
    """
    omega_n = math.sqrt(kp / I)
    kd = 2.0 * target_zeta * math.sqrt(kp * I)
    t_settle = 4.0 / (target_zeta * omega_n) if target_zeta * omega_n > 0 else float("inf")

    # Worst-case moment: full tilt error + peak angular velocity during step response
    peak_omega = max_tilt * omega_n  # peak ang_vel for critically damped step response
    max_total_moment = kp * max_tilt + kd * peak_omega

    return {
        "kp": kp,
        "kd": kd,
        "omega_n": omega_n,
        "zeta": target_zeta,
        "t_settle_s": t_settle,
        "peak_omega_rad_s": peak_omega,
        "max_total_moment_Nm": max_total_moment,
    }


@hydra_task_config(args_cli.task, "skrl_mappo_cfg_entry_point")
def main(env_cfg, agent_cfg: dict) -> None:
    """Extract inertia and compute recommended PD gains."""
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42

    env = gym.make(args_cli.task, cfg=env_cfg)
    base = env.unwrapped
    if not isinstance(base, DirectMARLEnv):
        raise TypeError(f"Expected DirectMARLEnv, got {type(base).__name__}")

    robot = base.robot

    # --- Extract mass ---
    masses = robot.root_physx_view.get_masses()  # shape: [num_instances, num_bodies]
    total_mass = float(masses[0].sum().item())
    body_idx = base._body_indices
    body_mass = float(masses[0, body_idx].item())

    # --- Extract inertia ---
    inertias = robot.root_physx_view.get_inertias()
    print(f"\n  [DEBUG] inertias shape: {inertias.shape}")
    print(f"  [DEBUG] body_idx: {body_idx}")

    # Handle different possible shapes
    if inertias.dim() == 3:
        # shape: [num_instances, num_bodies, 9]
        body_inertia = inertias[0, body_idx].flatten()
    elif inertias.dim() == 2:
        # shape: [num_instances, 9] (root body only)
        body_inertia = inertias[0].flatten()
    else:
        body_inertia = inertias.flatten()

    print(f"  [DEBUG] body_inertia shape: {body_inertia.shape}")
    print(f"  [DEBUG] body_inertia values: {body_inertia}")

    Ixx = float(body_inertia[0])
    Iyy = float(body_inertia[4])
    Izz = float(body_inertia[8])

    # Also check off-diagonal elements
    Ixy = float(body_inertia[1])
    Ixz = float(body_inertia[2])
    Iyz = float(body_inertia[5])

    # --- Current config values ---
    kp_att = float(env_cfg.kp_att)
    kd_att = float(env_cfg.kd_att)
    kp_yaw = float(env_cfg.kp_yaw)
    max_moment = float(env_cfg.max_moment)
    max_tilt = float(env_cfg.max_tilt_angle)

    # --- Current damping analysis ---
    I_roll = Ixx  # roll axis
    I_pitch = Iyy  # pitch axis
    I_yaw = Izz  # yaw axis

    omega_n_current = math.sqrt(kp_att / I_roll) if I_roll > 0 else 0
    zeta_current = kd_att / (2.0 * math.sqrt(kp_att * I_roll)) if I_roll > 0 else 0

    # --- Print results ---
    sep = "=" * 70
    print(f"\n{sep}")
    print("  CRAZYFLIE INERTIA EXTRACTION & PD GAIN ANALYSIS")
    print(sep)

    print(f"\n  Mass:")
    print(f"    Total mass:       {total_mass:.6f} kg")
    print(f"    Main body mass:   {body_mass:.6f} kg")
    print(f"    Robot weight:     {total_mass * 9.81:.4f} N")

    print(f"\n  Inertia tensor (main body, body_idx={body_idx}):")
    print(f"    Ixx = {Ixx:.8f} kg*m^2")
    print(f"    Iyy = {Iyy:.8f} kg*m^2")
    print(f"    Izz = {Izz:.8f} kg*m^2")
    print(f"    Ixy = {Ixy:.8f}  Ixz = {Ixz:.8f}  Iyz = {Iyz:.8f}")

    print(f"\n  Current PD gains:")
    print(f"    kp_att = {kp_att}")
    print(f"    kd_att = {kd_att}")
    print(f"    kp_yaw = {kp_yaw}")
    print(f"    max_moment = {max_moment} Nm")
    print(f"    max_tilt_angle = {max_tilt} rad ({math.degrees(max_tilt):.1f}deg)")

    print(f"\n  Current dynamics (roll axis, I={I_roll:.8f}):")
    print(f"    omega_n = {omega_n_current:.2f} rad/s")
    print(f"    zeta = {zeta_current:.4f} ({'underdamped' if zeta_current < 1 else 'overdamped' if zeta_current > 1 else 'critical'})")
    if zeta_current > 0 and omega_n_current > 0:
        t_s = 4.0 / (zeta_current * omega_n_current)
        print(f"    settling time = {t_s:.4f} s ({t_s / 0.02:.0f} control steps @ 50Hz)")
    max_p_moment = kp_att * max_tilt
    print(f"    max P-moment (30deg error) = {max_p_moment:.6f} Nm")
    print(f"    max angular accel = {max_moment / I_roll:.1f} rad/s^2")

    # --- Recommended gains ---
    print(f"\n{sep}")
    print("  RECOMMENDED PD GAINS")
    print(sep)

    for zeta_target in (1.0, 1.1, 1.2):
        gains = _compute_gains(I_roll, kp_att, max_tilt, zeta_target)
        label = {1.0: "critical", 1.1: "slight overdamp", 1.2: "overdamped"}[zeta_target]
        print(f"\n  --- zeta = {zeta_target} ({label}), kp_att = {kp_att} ---")
        print(f"    kd_att = {gains['kd']:.6f}")
        print(f"    omega_n = {gains['omega_n']:.2f} rad/s")
        print(f"    settling = {gains['t_settle_s']:.4f} s ({gains['t_settle_s'] / 0.02:.0f} steps)")
        print(f"    peak ang_vel = {gains['peak_omega_rad_s']:.2f} rad/s")
        print(f"    worst-case moment = {gains['max_total_moment_Nm']:.6f} Nm")
        if gains["max_total_moment_Nm"] > max_moment:
            headroom = max_moment / gains["max_total_moment_Nm"]
            print(f"    WARNING: EXCEEDS max_moment={max_moment}! ({headroom:.0%} budget)")
            # Suggest reduced kp that fits
            kp_reduced = 0.7 * max_moment / max_tilt
            gains_r = _compute_gains(I_roll, kp_reduced, max_tilt, zeta_target)
            print(f"    -> kp_att={kp_reduced:.6f} fits: kd={gains_r['kd']:.6f}, "
                  f"worst={gains_r['max_total_moment_Nm']:.6f} Nm, "
                  f"settle={gains_r['t_settle_s']:.4f}s")
        else:
            headroom = 1.0 - gains["max_total_moment_Nm"] / max_moment
            print(f"    OK: Fits in max_moment={max_moment} ({headroom:.0%} headroom)")

    # Yaw gain recommendation
    print(f"\n  --- Yaw (I_yaw = {I_yaw:.8f}) ---")
    # For yaw P-only: kp_yaw * max_yaw_rate <= max_moment
    max_yaw_rate = float(env_cfg.max_yaw_rate)
    max_yaw_moment = kp_yaw * max_yaw_rate
    print(f"    Current kp_yaw = {kp_yaw}, max yaw moment = {max_yaw_moment:.6f} Nm")
    print(f"    Max angular accel (yaw) = {max_moment / I_yaw:.1f} rad/s^2")

    print(f"\n{sep}\n")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
