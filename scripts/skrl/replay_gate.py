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
parser.add_argument("--seeds", type=str, default="7,13,21,42,99",
                    help="Comma-separated seeds to evaluate")
parser.add_argument("--ref_dir", type=str, default="logs/ref/v1.0.0-capstone",
                    help="Reference rollouts directory")
parser.add_argument("--sigma_tol", type=float, default=2.0, help="Pass tolerance")
parser.add_argument("--forest", action="store_true",
                    help="G1a-3 forest-mode smoke: enable forest_enabled cfg, "
                    "skip metric comparison; pass = no shape errors over play_length steps")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import ggswarm.tasks  # noqa: F401, E402
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from ggswarm.gnn_policy import GgswarmGNNPolicy  # noqa: E402

NUM_AGENTS = 8
COLLISION_RADIUS_M = 0.10


def build_env(num_envs: int, play_length: int, forest: bool = False):
    env_cfg = parse_env_cfg("ggswarm-marl-v0", num_envs=num_envs, use_fabric=True)
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = play_length * env_cfg.decimation * env_cfg.sim.dt + 1.0
    env_cfg.formation_centroid = (0.0, 0.0, 1.0)
    env_cfg.dropout_enabled = False
    env_cfg.forest_enabled = forest
    env = gym.make("ggswarm-marl-v0", cfg=env_cfg)
    return SkrlVecEnvWrapper(env, ml_framework="torch"), env_cfg


def load_actor_weights(checkpoint_path: str, actor: GgswarmGNNPolicy, device) -> dict | None:
    """Loads actor weights from either a capstone single-agent PPO checkpoint
    (top-level `policy`/`value`/`state_preprocessor` keys) or a MAPPO MARL
    checkpoint (per-agent dicts keyed by `drone_0..drone_{A-1}`, each with the
    same nested structure). Returns the state_preprocessor stats dict for
    manual obs normalization, or None if absent.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict):
        policy_sd = ckpt
        sp = None
    elif "policy" in ckpt:
        # Single-agent PPO format (capstone)
        policy_sd = ckpt["policy"]
        sp = ckpt.get("state_preprocessor")
        print("[INFO] Loading single-agent PPO checkpoint (capstone format)")
    elif "drone_0" in ckpt:
        # MAPPO format — all drones share weights, pull from drone_0
        policy_sd = ckpt["drone_0"]["policy"]
        sp = ckpt["drone_0"].get("state_preprocessor")
        print("[INFO] Loading MAPPO checkpoint (per-agent dict format, "
              "using drone_0's shared weights)")
    else:
        raise RuntimeError(
            f"Unrecognized checkpoint structure. Top keys: {list(ckpt.keys())}"
        )
    missing, unexpected = actor.load_state_dict(policy_sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys in actor: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys in checkpoint: {unexpected}")
    return sp


# Backwards-compat alias for older callers / docs.
load_capstone_actor = load_actor_weights


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


def main() -> int:
    seeds = [int(s) for s in args.seeds.split(",")]
    ref_meta = json.loads(Path(args.ref_dir, "rollouts_metadata.json").read_text())
    ref = ref_meta["metrics"]

    env, env_cfg = build_env(args.num_envs, args.play_length, forest=args.forest)
    device = env.device
    if args.forest:
        print(f"[INFO] Forest-mode smoke (G1a-3): forest_enabled=True, "
              f"play_length={args.play_length}, no metric comparison")

    agent0 = env.possible_agents[0]
    actor = GgswarmGNNPolicy(
        observation_space=env.observation_space(agent0),
        action_space=env.action_space(agent0),
        device=device,
        num_neighbors=env_cfg.num_neighbors,
        num_agents=NUM_AGENTS,
    )
    GgswarmGNNPolicy.init_edge_cache(memory_size=1, num_envs=env.num_envs * NUM_AGENTS)

    state_preproc_stats = load_actor_weights(args.checkpoint, actor, device)
    actor.to(device)  # force move all params to device after state_dict load
    actor.train(False)  # inference mode (equivalent to .eval())

    if state_preproc_stats is not None:
        running_mean = state_preproc_stats["running_mean"].to(device=device, dtype=torch.float32)
        running_var = state_preproc_stats["running_variance"].to(device=device, dtype=torch.float32)
        eps = 1e-8
        print(f"[INFO] Using capstone state_preprocessor (mean_avg={running_mean.mean():.4f},"
              f" var_avg={running_var.mean():.4f}, count={state_preproc_stats['current_count']})")
    else:
        running_mean = None
        running_var = None
        print("[WARN] No state_preprocessor in checkpoint; running unnormalized")

    per_seed: dict[int, dict[str, float]] = {}

    # Forest smoke: only run one seed, skip metric comparison.
    if args.forest:
        seeds = seeds[:1]

    for seed in seeds:
        env.unwrapped.seed(seed)
        obs_dict, _ = env.reset()

        positions = torch.zeros(args.play_length, NUM_AGENTS, 3, device=device)
        goals = torch.zeros(args.play_length, NUM_AGENTS, 3, device=device)

        with torch.no_grad():
            for t in range(args.play_length):
                actions = {}
                for agent_id in env.possible_agents:
                    o = obs_dict[agent_id]
                    if running_mean is not None:
                        o = (o - running_mean) / torch.sqrt(running_var + eps)
                    mean_action, _, _ = actor.compute({"states": o}, role="policy")
                    actions[agent_id] = mean_action.clamp(-1.0, 1.0)
                obs_dict, _, _, _, _ = env.step(actions)

                positions[t] = env.unwrapped._robot.data.root_pos_w[:NUM_AGENTS] - \
                    env.unwrapped._terrain.env_origins[0:1].repeat(NUM_AGENTS, 1)
                goals[t] = env.unwrapped._desired_pos_w[:NUM_AGENTS] - \
                    env.unwrapped._terrain.env_origins[0:1].repeat(NUM_AGENTS, 1)

        if args.forest:
            print(f"  forest smoke seed={seed}: {args.play_length} steps clean, no shape errors")
            continue
        m = per_seed_metrics(positions, goals)
        print(f"  seed={seed}: slot_err={m['mean_slot_error_m']:.4f}, "
              f"final_dist={m['final_distance_to_goal_m']:.4f}, "
              f"coll/step={m['collision_pairs_per_step']:.4f}")
        per_seed[seed] = m

    if args.forest:
        print("\n=== Forest smoke (G1a-3) PASS — no shape errors ===")
        env.close()
        simulation_app.close()
        return 0

    keys = ["mean_slot_error_m", "collision_pairs_per_step", "final_distance_to_goal_m"]
    aggregate = {
        k: {
            "mean": statistics.mean([per_seed[s][k] for s in seeds]),
            "std": statistics.stdev([per_seed[s][k] for s in seeds]) if len(seeds) > 1 else 0.0,
        } for k in keys
    }

    print("\n=== Replay gate result ===")
    print(f"{'metric':30s} | {'capstone (mean +/- std)':25s} | {'1a env (mean +/- std)':25s} | "
          f"{'sigma-dist':10s} | pass")
    fail = False
    for k in keys:
        cap_mean, cap_std = ref[k]["mean"], ref[k]["std"]
        new_mean, new_std = aggregate[k]["mean"], aggregate[k]["std"]
        sigma_dist = abs(new_mean - cap_mean) / max(cap_std, 1e-9)
        passed = sigma_dist <= args.sigma_tol
        if not passed:
            fail = True
        print(f"{k:30s} | {cap_mean:8.4f} +/- {cap_std:7.4f}  | "
              f"{new_mean:8.4f} +/- {new_std:7.4f}  | {sigma_dist:10.2f} | "
              f"{'PASS' if passed else 'FAIL'}")

    out_path = Path("logs/sweeps/phase1a_replay_gate.txt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        f.write("metric,capstone_mean,capstone_std,marl_mean,marl_std,sigma_distance,pass\n")
        for k in keys:
            cap_mean, cap_std = ref[k]["mean"], ref[k]["std"]
            new_mean, new_std = aggregate[k]["mean"], aggregate[k]["std"]
            sigma_dist = abs(new_mean - cap_mean) / max(cap_std, 1e-9)
            passed = sigma_dist <= args.sigma_tol
            f.write(f"{k},{cap_mean:.6f},{cap_std:.6f},{new_mean:.6f},"
                    f"{new_std:.6f},{sigma_dist:.6f},{passed}\n")
    print(f"\nSaved: {out_path}")

    env.close()
    simulation_app.close()
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
