"""Reference quadcopter example (derived from Isaac Lab demo)."""

"""Launch Isaac Sim Simulator first."""

import argparse

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Reference quadcopter example (Isaac Lab demo).")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext

from isaaclab_assets import CRAZYFLIE_CFG  # isort:skip


def main() -> None:
    """Run a small quadcopter simulation for reference/visual inspection."""
    sim_cfg = sim_utils.SimulationCfg(dt=0.005, device=args_cli.device)
    sim = SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[0.5, 0.5, 1.0], target=[0.0, 0.0, 0.5])

    # Ground plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)

    # Lighting
    cfg = sim_utils.DistantLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    cfg.func("/World/Light", cfg)

    # Robot
    robot_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Crazyflie")
    robot_cfg.spawn.func("/World/Crazyflie", robot_cfg.spawn, translation=robot_cfg.init_state.pos)
    robot = Articulation(robot_cfg)

    # Play the simulator
    sim.reset()

    # Fetch relevant params to make the quadcopter hover in place
    prop_body_ids = robot.find_bodies("m.*_prop")[0]
    robot_mass = robot.root_physx_view.get_masses().sum()
    gravity = torch.tensor(sim.cfg.gravity, device=sim.device).norm()

    while simulation_app.is_running():
        # Periodic reset
        if int(sim.get_physics_time() * 200.0) % 2000 == 0:
            joint_pos, joint_vel = robot.data.default_joint_pos, robot.data.default_joint_vel
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.write_root_pose_to_sim(robot.data.default_root_state[:, :7])
            robot.write_root_velocity_to_sim(robot.data.default_root_state[:, 7:])
            robot.reset()

        forces = torch.zeros(robot.num_instances, 4, 3, device=sim.device)
        torques = torch.zeros_like(forces)

        drone_wt = robot_mass * gravity

        # Deterministic thrust variation (sinusoidal wobble)
        sin_wave = torch.sin(torch.tensor(sim.get_physics_time() * 10.0, device=sim.device))
        thrust_variation = (drone_wt * 0.1 * sin_wave) - (drone_wt * 0.1)
        forces[..., 2] = (drone_wt / 4.0) + thrust_variation

        robot.permanent_wrench_composer.set_forces_and_torques(
            forces=forces,
            torques=torques,
            body_ids=prop_body_ids,
        )
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim.get_physics_dt())

        # Camera follow
        root_pos = robot.data.root_pos_w[0].cpu().tolist()
        camera_offset = [0.5, 0.5, 1.0]
        sim.set_camera_view(
            eye=[
                root_pos[0] + camera_offset[0],
                root_pos[1] + camera_offset[1],
                root_pos[2] + camera_offset[2],
            ],
            target=root_pos,
        )


if __name__ == "__main__":
    main()
    simulation_app.close()

