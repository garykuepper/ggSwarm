"""Phase 1a probe: confirm the multi-drone-per-env regex Articulation pattern.

Spawns 2 origins × 3 drones each (Drone_0/1/2 under Origin_0/1) using
CRAZYFLIE_CFG.spawn.func directly, then constructs ONE Articulation with
prim_path "/World/Origin_.*/Drone_.*". After a few sim steps, prints:

  - num_instances (should be 6)
  - body_names (per-articulation body roster)
  - find_bodies("body") result
  - data.root_pos_w with each drone placed at a known offset, so we can read
    the flattening order directly from the tensor.

This decides whether the row-major reshape `[num_envs, A, 3]` math the
existing Phase 1a plan assumes is valid for env-major order, drone-major
order, or something else.

Throwaway — delete after the design question is answered.
"""

from __future__ import annotations

import argparse
import sys

# Unbuffered stdout: Isaac Sim shutdown can hang on Windows; we want every
# print visible in real time even if the process needs to be killed.
sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.sim.utils import find_matching_prim_paths  # noqa: E402

from isaaclab_assets import CRAZYFLIE_CFG  # noqa: E402  isort: skip


NUM_ORIGINS = 2
NUM_DRONES = 3


def design_scene():
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    cfg = sim_utils.DomeLightCfg(intensity=2000.0)
    cfg.func("/World/Light", cfg)

    # Two origins (proxy for two envs)
    origin_offsets = [(0.0, 0.0, 0.0), (10.0, 0.0, 0.0)]
    for i, offset in enumerate(origin_offsets):
        sim_utils.create_prim(f"/World/Origin_{i}", "Xform", translation=offset)

    # Spawn each drone individually under each origin using a known XYZ offset
    # so we can read flattening order from the resulting tensor.
    spawn_cfg = CRAZYFLIE_CFG.spawn
    for o in range(NUM_ORIGINS):
        for d in range(NUM_DRONES):
            prim_path = f"/World/Origin_{o}/Drone_{d}"
            # XY offset encodes drone identity within an origin: x = d * 1.0
            # Origin offset already moves the whole pack 10m in x for o=1.
            spawn_cfg.func(
                prim_path,
                spawn_cfg,
                translation=(float(d) * 1.0, 0.0, 1.0),
                orientation=(1.0, 0.0, 0.0, 0.0),
            )

    # Single Articulation with regex matching all NUM_ORIGINS * NUM_DRONES drones.
    art_cfg = CRAZYFLIE_CFG.replace(prim_path="/World/Origin_.*/Drone_.*")
    art_cfg.spawn = None  # already spawned above; don't try to re-spawn
    drone = Articulation(cfg=art_cfg)
    return drone, origin_offsets


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=1 / 100))
    drone, origin_offsets = design_scene()

    matching = find_matching_prim_paths("/World/Origin_.*/Drone_.*")
    print("\n=== USD-stage match order ===")
    for i, p in enumerate(matching):
        print(f"  [{i}] {p}")

    sim.reset()  # triggers _initialize_impl on the Articulation

    print("\n=== Articulation summary ===")
    print(f"num_instances : {drone.num_instances}")
    print(f"num_bodies    : {drone.num_bodies}")
    print(f"body_names    : {drone.body_names}")
    body_idx, body_names = drone.find_bodies("body")
    print(f"find_bodies('body') -> idx={body_idx}, names={body_names}")

    # Step a few times so transforms settle, then read root_pos_w.
    for _ in range(3):
        sim.step()
        drone.update(sim.get_physics_dt())

    print("\n=== root_pos_w (one row per instance) ===")
    pos = drone.data.root_pos_w  # [num_instances, 3]
    print(f"shape: {tuple(pos.shape)}")
    for i in range(pos.shape[0]):
        x, y, z = pos[i].tolist()
        # Each spawned drone has translation (d * 1.0, 0, 1) within its origin,
        # plus origin_offset for o=1 adding (10, 0, 0). So expected x =
        # origin_offset.x + d. Reverse-engineer (origin_idx, drone_idx) from x.
        ox = round(x / 10.0) if x >= 5.0 else 0
        dx = round(x - ox * 10.0)
        print(f"  inst {i}: x={x:6.3f} y={y:6.3f} z={z:6.3f}  -> origin={ox}, drone={dx}")

    expected_env_major = [(o, d) for o in range(NUM_ORIGINS) for d in range(NUM_DRONES)]
    expected_drone_major = [(o, d) for d in range(NUM_DRONES) for o in range(NUM_ORIGINS)]
    actual = []
    for i in range(pos.shape[0]):
        x = pos[i, 0].item()
        ox = round(x / 10.0) if x >= 5.0 else 0
        dx = round(x - ox * 10.0)
        actual.append((ox, dx))

    print("\n=== Order verdict ===")
    print(f"actual:        {actual}")
    print(f"env-major:     {expected_env_major}")
    print(f"drone-major:   {expected_drone_major}")
    if actual == expected_env_major:
        print("RESULT: env-major ordering — plan's reshape math is valid.")
    elif actual == expected_drone_major:
        print("RESULT: drone-major ordering — plan needs a permutation index.")
    else:
        print("RESULT: unexpected order — design must adapt.")

    simulation_app.close()


if __name__ == "__main__":
    main()
