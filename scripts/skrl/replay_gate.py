"""Phase 1a replay gate (G1a-4).

Loads the capstone p4-revert-4 actor weights into the MAPPO env's shared
GNN actor, runs N rollouts in `ggswarm-marl-v0` (shared PhysX scene, A=8
drones per env), and compares per-rollout metrics against the captured
`logs/ref/v1.0.0-capstone/rollouts_metadata.json` reference distribution.

Pass = the three CSV-derivable metrics (mean_slot_error_m,
collision_pairs_per_step, final_distance_to_goal_m) within 2σ of the
capstone reference. Per the spec, episode_reward is not used (play.py
doesn't surface it).

Capstone checkpoint format (verified): top-level dict with `policy`,
`value`, `optimizer`, `state_preprocessor`, `value_preprocessor` keys.
Both `policy` and `value` are state_dicts of the same single-agent
GgswarmGNNPolicy module. We load `policy` into the MAPPO actor and the
state_preprocessor stats into the MAPPO state_preprocessor (per drone
obs normalization). Value head + value_preprocessor are inference-time
no-ops (only used during training updates).
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase 1a replay gate")
parser.add_argument("--checkpoint", required=True, help="Capstone p4-revert-4 best_agent.pt path")
parser.add_argument("--play_length", type=int, default=500, help="Steps per rollout")
parser.add_argument("--num_envs", type=int, default=1, help="Envs to run in parallel")
parser.add_argument("--seeds", type=str, default="7,13,21,42,99", help="Comma-separated seeds to evaluate")
parser.add_argument("--ref_dir", type=str, default="logs/ref/v1.0.0-capstone", help="Reference rollouts directory")
parser.add_argument("--sigma_tol", type=float, default=2.0, help="Pass tolerance")
parser.add_argument(
    "--forest",
    action="store_true",
    help="G1a-3 forest-mode smoke: enable forest_enabled cfg, "
    "skip metric comparison; pass = no shape errors over play_length steps",
)
parser.add_argument(
    "--traj_dir",
    type=str,
    default="logs/replay_gate/trajectories",
    help="Directory to write per-seed trajectory CSVs (format: step, d{i}_{x,y,z,gx,gy,gz}). Use --no_traj to disable.",
)
parser.add_argument("--no_traj", action="store_true", help="Disable per-seed trajectory CSV recording")
parser.add_argument(
    "--prefix", type=str, default=None, help="Trajectory CSV filename prefix; defaults to checkpoint stem"
)
parser.add_argument(
    "--video", action="store_true", help="Record an NVENC H.264 video of the rollout (one file per seed)."
)
parser.add_argument(
    "--video_length", type=int, default=None, help="Steps to record per seed (defaults to --play_length)."
)
parser.add_argument(
    "--video_prefix", type=str, default=None, help="Video filename prefix (defaults to checkpoint run dir name)."
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# Default: GUI on (visual playback). Pass --headless for the metrics-only gate run.
# Video requires camera-rendered frames.
if args.video:
    args.enable_cameras = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import ggswarm.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from ggswarm.checkpoint_utils import init_state_preprocessor, load_actor_weights  # noqa: E402
from ggswarm.gnn_policy import GgswarmGNNPolicy  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

NUM_AGENTS = 8
COLLISION_RADIUS_M = 0.10
METRIC_KEYS = ("mean_slot_error_m", "collision_pairs_per_step", "final_distance_to_goal_m")


def build_env(
    num_envs: int,
    play_length: int,
    forest: bool = False,
    video: bool = False,
    video_folder: str | None = None,
    video_length: int = 500,
    video_prefix: str | None = None,
):
    """Build the MARL env, optionally wrapped with NvencRecorder for video."""
    env_cfg = parse_env_cfg("ggswarm-marl-v0", num_envs=num_envs, use_fabric=True)
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = play_length * env_cfg.decimation * env_cfg.sim.dt + 1.0
    env_cfg.formation_centroid = (0.0, 0.0, 1.0)
    env_cfg.dropout_enabled = False
    env_cfg.forest_enabled = forest
    env = gym.make(
        "ggswarm-marl-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if video else None,
    )
    if video:
        from ggswarm.viz.nvenc_recorder import NvencRecorder  # noqa: PLC0415

        env = NvencRecorder(
            env,
            video_folder=video_folder,
            video_length=video_length,
            name_prefix=video_prefix or "replay_gate",
        )
        print(f"[INFO] Recording video to {video_folder} (NVENC H.264, {video_length} steps, prefix={video_prefix})")
    return SkrlVecEnvWrapper(env, ml_framework="torch"), env_cfg


def per_seed_metrics(positions: torch.Tensor, goals: torch.Tensor) -> dict[str, float]:
    """positions, goals: [T, A, 3]. Returns the three CSV-derivable metrics."""
    diff = positions - goals
    slot_err = diff.norm(dim=-1).mean().item()
    final_dist = diff[-1].norm(dim=-1).mean().item()
    coll = 0.0
    A = positions.shape[1]
    for t in range(positions.shape[0]):
        p = positions[t]
        for i in range(A):
            for j in range(i + 1, A):
                if (p[i] - p[j]).norm().item() < COLLISION_RADIUS_M:
                    coll += 1
    coll /= positions.shape[0]
    return {
        "mean_slot_error_m": slot_err,
        "final_distance_to_goal_m": final_dist,
        "collision_pairs_per_step": coll,
    }


def _run_seed_rollout(
    env, actor, state_preprocessor, seed: int, play_length: int, device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one rollout at the given seed, return [T, A, 3] position + goal tensors."""
    env.unwrapped.seed(seed)
    obs_dict, _ = env.reset()

    positions = torch.zeros(play_length, NUM_AGENTS, 3, device=device)
    goals = torch.zeros(play_length, NUM_AGENTS, 3, device=device)
    env_origin = env.unwrapped._terrain.env_origins[0:1].repeat(NUM_AGENTS, 1)

    with torch.no_grad():
        for t in range(play_length):
            actions = {}
            for agent_id in env.possible_agents:
                o = obs_dict[agent_id]
                if state_preprocessor is not None:
                    o = state_preprocessor(o)
                mean_action, _, _ = actor.compute({"states": o}, role="policy")
                actions[agent_id] = mean_action.clamp(-1.0, 1.0)
            obs_dict, _, _, _, _ = env.step(actions)

            positions[t] = env.unwrapped._robot.data.root_pos_w[:NUM_AGENTS] - env_origin
            goals[t] = env.unwrapped._desired_pos_w[:NUM_AGENTS] - env_origin
    return positions, goals


def _write_trajectory_csv(traj_dir: Path, seed: int, positions: torch.Tensor, goals: torch.Tensor) -> None:
    """Same column layout as the capstone reference rollouts."""
    csv_path = traj_dir / f"seed{seed}-trajectory_data.csv"
    cols = ["step"]
    for d in range(NUM_AGENTS):
        cols += [f"d{d}_x", f"d{d}_y", f"d{d}_z", f"d{d}_gx", f"d{d}_gy", f"d{d}_gz"]
    pos_cpu = positions.cpu()
    goal_cpu = goals.cpu()
    with csv_path.open("w") as f:
        f.write(",".join(cols) + "\n")
        for t in range(positions.shape[0]):
            row = [str(t)]
            for d in range(NUM_AGENTS):
                row += [
                    f"{pos_cpu[t, d, 0]:.4f}",
                    f"{pos_cpu[t, d, 1]:.4f}",
                    f"{pos_cpu[t, d, 2]:.4f}",
                    f"{goal_cpu[t, d, 0]:.4f}",
                    f"{goal_cpu[t, d, 1]:.4f}",
                    f"{goal_cpu[t, d, 2]:.4f}",
                ]
            f.write(",".join(row) + "\n")
    print(f"  wrote {csv_path}")


def _aggregate_and_report(
    per_seed: dict[int, dict[str, float]], ref: dict[str, dict[str, float]], sigma_tol: float, out_path: Path
) -> bool:
    """Print the result table and save the CSV in one pass. Returns True on fail."""
    seeds = sorted(per_seed.keys())
    aggregate = {
        k: {
            "mean": statistics.mean([per_seed[s][k] for s in seeds]),
            "std": statistics.stdev([per_seed[s][k] for s in seeds]) if len(seeds) > 1 else 0.0,
        }
        for k in METRIC_KEYS
    }

    print("\n=== Replay gate result ===")
    print(
        f"{'metric':30s} | {'capstone (mean +/- std)':25s} | {'1a env (mean +/- std)':25s} | {'sigma-dist':10s} | pass"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fail = False
    with out_path.open("w") as f:
        f.write("metric,capstone_mean,capstone_std,marl_mean,marl_std,sigma_distance,pass\n")
        for k in METRIC_KEYS:
            cap_mean, cap_std = ref[k]["mean"], ref[k]["std"]
            new_mean, new_std = aggregate[k]["mean"], aggregate[k]["std"]
            sigma_dist = abs(new_mean - cap_mean) / max(cap_std, 1e-9)
            passed = sigma_dist <= sigma_tol
            if not passed:
                fail = True
            print(
                f"{k:30s} | {cap_mean:8.4f} +/- {cap_std:7.4f}  | "
                f"{new_mean:8.4f} +/- {new_std:7.4f}  | {sigma_dist:10.2f} | "
                f"{'PASS' if passed else 'FAIL'}"
            )
            f.write(f"{k},{cap_mean:.6f},{cap_std:.6f},{new_mean:.6f},{new_std:.6f},{sigma_dist:.6f},{passed}\n")
    print(f"\nSaved: {out_path}")
    return fail


def _resolve_video_paths() -> tuple[str | None, str | None]:
    if not args.video:
        return None, None
    video_prefix = args.video_prefix or Path(args.checkpoint).parent.parent.name
    video_folder = str(Path(args.checkpoint).parent.parent / "videos" / "replay_gate")
    Path(video_folder).mkdir(parents=True, exist_ok=True)
    return video_folder, video_prefix


def _resolve_traj_dir() -> Path | None:
    if args.no_traj:
        return None
    prefix = args.prefix or Path(args.checkpoint).parent.parent.name
    traj_dir = Path(args.traj_dir) / prefix
    traj_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Recording trajectory CSVs to {traj_dir}/")
    return traj_dir


def main() -> int:
    seeds = [int(s) for s in args.seeds.split(",")]
    ref_meta = json.loads(Path(args.ref_dir, "rollouts_metadata.json").read_text())
    ref = ref_meta["metrics"]

    video_folder, video_prefix = _resolve_video_paths()
    env, env_cfg = build_env(
        args.num_envs,
        args.play_length,
        forest=args.forest,
        video=args.video,
        video_folder=video_folder,
        video_length=args.video_length or args.play_length,
        video_prefix=video_prefix,
    )
    device = env.device
    if args.forest:
        print(
            f"[INFO] Forest-mode smoke (G1a-3): forest_enabled=True, "
            f"play_length={args.play_length}, no metric comparison"
        )
        seeds = seeds[:1]

    agent0 = env.possible_agents[0]
    actor = GgswarmGNNPolicy(
        observation_space=env.observation_space(agent0),
        action_space=env.action_space(agent0),
        device=device,
        num_neighbors=env_cfg.num_neighbors,
        num_agents=NUM_AGENTS,
    )
    GgswarmGNNPolicy.init_edge_cache(memory_size=1, num_envs=env.num_envs * NUM_AGENTS)

    sp_stats = load_actor_weights(args.checkpoint, actor, device)
    actor.to(device)  # force move all params to device after state_dict load
    actor.train(False)
    state_preprocessor = init_state_preprocessor(sp_stats, actor.observation_space, device)

    traj_dir = _resolve_traj_dir()
    per_seed: dict[int, dict[str, float]] = {}

    for seed in seeds:
        positions, goals = _run_seed_rollout(
            env,
            actor,
            state_preprocessor,
            seed,
            args.play_length,
            device,
        )
        if traj_dir is not None:
            _write_trajectory_csv(traj_dir, seed, positions, goals)

        if args.forest:
            print(f"  forest smoke seed={seed}: {args.play_length} steps clean, no shape errors")
            continue
        m = per_seed_metrics(positions, goals)
        print(
            f"  seed={seed}: slot_err={m['mean_slot_error_m']:.4f}, "
            f"final_dist={m['final_distance_to_goal_m']:.4f}, "
            f"coll/step={m['collision_pairs_per_step']:.4f}"
        )
        per_seed[seed] = m

    if args.forest:
        print("\n=== Forest smoke (G1a-3) PASS — no shape errors ===")
        env.close()
        simulation_app.close()
        return 0

    fail = _aggregate_and_report(
        per_seed,
        ref,
        args.sigma_tol,
        Path("logs/sweeps/phase1a_replay_gate.txt"),
    )
    env.close()
    simulation_app.close()
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
