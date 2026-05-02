"""Phase 1a throughput sweep for the shared-scene MARL env.

Runs train.py at progressively larger num_envs for a small max_iterations
under MAPPO, parses wall-clock training time, computes steps/sec, and
reports a knee selection. The cost profile of shared-scene + MAPPO on
the local 3070 is genuinely unknown; the sweep is the discovery (G1a-2).

Stops at first failure (likely OOM or PhysX scene-size limits) — a
non-zero return code or absent "Training time:" line counts as failure.

Output: logs/sweeps/phase1a_throughput.txt (CSV).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

# Sweep configuration. Each row uses 5 iters * 24 rollouts = 120 sim steps.
ENV_COUNTS = [16, 32, 64, 128, 256, 512]
MAX_ITERS = 5
TASK = "ggswarm-marl-v0"
ALGORITHM = "MAPPO"
REPO = Path(__file__).resolve().parents[2]
PYTHON = str(REPO / "env_isaaclab" / "Scripts" / "python.exe")
OUT = REPO / "logs/sweeps/phase1a_throughput.txt"


_TRAINING_TIME_RE = re.compile(r"Training time:\s*([\d.]+)\s*seconds")


def _run(num_envs: int, log_path: Path) -> tuple[float | None, str]:
    """Returns (training_time_seconds | None, status). steps/sec is computed
    from rollouts*max_iters and num_envs * num_agents."""
    cmd = [
        PYTHON, "-u", "scripts/skrl/train.py", "--headless",
        "--task", TASK,
        "--num_envs", str(num_envs),
        "--max_iterations", str(MAX_ITERS),
        "--algorithm", ALGORITHM,
        "--log_subdir", "p1a-sweep",
    ]
    print(f"\n=== num_envs={num_envs}: {' '.join(cmd)}")
    with log_path.open("wb") as logf:
        proc = subprocess.run(
            cmd, cwd=str(REPO), stdout=logf, stderr=subprocess.STDOUT, timeout=900
        )
    text = log_path.read_text(errors="replace")
    if proc.returncode != 0:
        return None, "exit-error"
    m = _TRAINING_TIME_RE.search(text)
    if m is None:
        return None, "no-time-marker"
    return float(m.group(1)), "ok"


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results: list[tuple[int, float | None, float | None, str]] = []  # (n, time_s, steps_per_sec, status)

    for n in ENV_COUNTS:
        log_path = REPO / f"logs/sweeps/phase1a_throughput_n{n}.log"
        train_time, status = _run(n, log_path)
        if status != "ok":
            print(f"  FAIL ({status}); see {log_path}")
            results.append((n, None, None, status))
            break
        # Total sim steps = max_iterations * rollouts (24 from skrl_mappo_cfg.yaml).
        total_steps = MAX_ITERS * 24 * n  # per-env * envs (drone count factor folded into MAPPO update cost)
        sps = total_steps / train_time
        results.append((n, train_time, sps, "ok"))
        print(f"  num_envs={n}: {train_time:.2f}s, {sps:.1f} env-steps/s")

    # Print summary
    print("\n=== Sweep summary ===")
    print(f"{'num_envs':>9s} | {'train_time_s':>13s} | {'env_steps/s':>11s} | status")
    for n, t, s, status in results:
        ts = f"{t:.2f}" if t is not None else "-"
        ss = f"{s:.1f}" if s is not None else "-"
        print(f"{n:>9d} | {ts:>13s} | {ss:>11s} | {status}")

    ok = [(n, t, s) for n, t, s, status in results if status == "ok"]
    if not ok:
        print("\nNo successful runs. Stop condition #1.")
        OUT.write_text("num_envs,train_time_s,env_steps_per_sec,status\n")
        return 1

    # Knee selection: largest num_envs that completed (cost is roughly linear,
    # but the wall-clock includes Isaac Sim cold start which dominates at low
    # num_envs — so highest successful num_envs that fits in memory wins).
    knee_n, knee_t, knee_s = max(ok, key=lambda x: x[0])
    print(f"\nChosen knee: num_envs={knee_n} ({knee_s:.1f} env-steps/s, {knee_t:.2f}s wall)")

    OUT.write_text("num_envs,train_time_s,env_steps_per_sec,status\n")
    with OUT.open("a") as f:
        for n, t, s, status in results:
            ts = f"{t:.4f}" if t is not None else ""
            ss = f"{s:.4f}" if s is not None else ""
            f.write(f"{n},{ts},{ss},{status}\n")
        f.write(f"chosen_knee,{knee_n},{knee_s:.4f},ok\n")
    print(f"Saved: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
