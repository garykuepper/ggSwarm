"""One-off helper for Phase 1a Task 0: compute reference rollout metrics
across the 5 capstone seeds and emit logs/ref/v1.0.0-capstone/rollouts_metadata.json.

Reads CSVs already produced by `play.py --trajectories`, copies them into
logs/ref/v1.0.0-capstone/, and computes mean_slot_error_m,
collision_pairs_per_step, final_distance_to_goal_m across the 5 seeds.

`episode_reward` is omitted because play.py does not surface it.
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import statistics
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
# Capstone training tree lives under _capstone_archive/ post 2026-05-01.
SRC_DIR = REPO / "logs/skrl/ggswarm/_capstone_archive/p4/2026-04-06_21-09-24_ppo_torch/trajectories"
DST_DIR = REPO / "logs/ref/v1.0.0-capstone"
# Promoted canonical checkpoint location (independent of the archive path).
CKPT = REPO / "logs/ref/v1.0.0-capstone/best_agent.pt"
SEEDS = [7, 13, 21, 42, 99]
NUM_AGENTS = 8
PLAY_LENGTH = 500
COLLISION_RADIUS_M = 0.10  # capstone collision_radius (post-revert-4)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_csv(path: Path) -> list[dict[str, float]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    return [{k: float(v) for k, v in r.items()} for r in rows]


def per_seed_metrics(rows: list[dict[str, float]]) -> dict[str, float]:
    slot_errs: list[float] = []
    coll_per_step: list[int] = []
    for r in rows:
        # per-drone slot error (||pos - goal||)
        for d in range(NUM_AGENTS):
            dx = r[f"d{d}_x"] - r[f"d{d}_gx"]
            dy = r[f"d{d}_y"] - r[f"d{d}_gy"]
            dz = r[f"d{d}_z"] - r[f"d{d}_gz"]
            slot_errs.append((dx * dx + dy * dy + dz * dz) ** 0.5)
        # pairwise collisions
        positions = [(r[f"d{d}_x"], r[f"d{d}_y"], r[f"d{d}_z"]) for d in range(NUM_AGENTS)]
        cnt = 0
        for i in range(NUM_AGENTS):
            for j in range(i + 1, NUM_AGENTS):
                ax, ay, az = positions[i]
                bx, by, bz = positions[j]
                d2 = (ax - bx) ** 2 + (ay - by) ** 2 + (az - bz) ** 2
                if d2 < COLLISION_RADIUS_M ** 2:
                    cnt += 1
        coll_per_step.append(cnt)
    last = rows[-1]
    final_dists = []
    for d in range(NUM_AGENTS):
        dx = last[f"d{d}_x"] - last[f"d{d}_gx"]
        dy = last[f"d{d}_y"] - last[f"d{d}_gy"]
        dz = last[f"d{d}_z"] - last[f"d{d}_gz"]
        final_dists.append((dx * dx + dy * dy + dz * dz) ** 0.5)
    return {
        "mean_slot_error_m": sum(slot_errs) / len(slot_errs),
        "collision_pairs_per_step": sum(coll_per_step) / len(coll_per_step),
        "final_distance_to_goal_m": sum(final_dists) / len(final_dists),
    }


def main() -> None:
    DST_DIR.mkdir(parents=True, exist_ok=True)
    per_seed: dict[int, dict[str, float]] = {}
    for seed in SEEDS:
        src = SRC_DIR / f"capstone-ref-seed{seed}-trajectory_data.csv"
        dst = DST_DIR / src.name
        shutil.copy2(src, dst)
        per_seed[seed] = per_seed_metrics(parse_csv(dst))

    metric_keys = ["mean_slot_error_m", "collision_pairs_per_step", "final_distance_to_goal_m"]
    metrics_summary: dict[str, dict[str, float]] = {}
    for k in metric_keys:
        vals = [per_seed[s][k] for s in SEEDS]
        metrics_summary[k] = {
            "mean": statistics.mean(vals),
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
            "per_seed": {str(s): per_seed[s][k] for s in SEEDS},
        }

    metadata = {
        "checkpoint_path": str(CKPT.relative_to(REPO)).replace("\\", "/"),
        "checkpoint_sha256": sha256(CKPT),
        "capstone_commit": "c33d339",
        "capstone_tag": "v1.0.0-capstone",
        "task": "ggswarm-v0",
        "num_agents": NUM_AGENTS,
        "play_length": PLAY_LENGTH,
        "seeds": SEEDS,
        "collision_radius_m": COLLISION_RADIUS_M,
        "metrics": metrics_summary,
        "notes": (
            "episode_reward intentionally omitted: play.py does not surface "
            "per-rollout reward. Replay gate uses the three CSV-derivable "
            "metrics (mean_slot_error_m, collision_pairs_per_step, "
            "final_distance_to_goal_m) at 2-sigma tolerance."
        ),
    }
    out = DST_DIR / "rollouts_metadata.json"
    out.write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"Wrote {out}")
    for k, v in metrics_summary.items():
        print(f"  {k}: mean={v['mean']:.4f}  std={v['std']:.4f}")


if __name__ == "__main__":
    main()
