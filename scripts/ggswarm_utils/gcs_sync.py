"""GCS sync helpers for pulling training run artifacts locally.

Uses ``gcloud storage cp`` (NOT rsync -- rsync is broken on Windows per the
shell-syntax rule).  Syncs checkpoints/, params/, and TFEvents files from
``gs://gg-swarm-training-logs/logs/skrl/ggswarm_marl/<run_name>/``.

Public API:
    DEFAULT_GCS_BUCKET    -- default bucket URI constant
    has_local_data(run_dir)            -> bool
    sync_from_gcs(run_dir, gcs_bucket) -> None
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from ggswarm_utils.gcloud_exec import gcloud_not_found_message, resolve_gcloud_executable

DEFAULT_GCS_BUCKET: str = "gs://gg-swarm-training-logs"
_GCS_PREFIX: str = "logs/skrl/ggswarm_marl"


def has_local_data(run_dir: Path) -> bool:
    """Return True if run_dir already contains at least one checkpoint.

    Args:
        run_dir: Local path to the training run directory.
    """
    ckpts_dir = run_dir / "checkpoints"
    if not ckpts_dir.exists():
        return False
    return any(ckpts_dir.glob("*.pt"))


def sync_from_gcs(run_dir: Path, gcs_bucket: str = DEFAULT_GCS_BUCKET) -> None:
    """Sync run artifacts from GCS to the local run_dir.

    Syncs three artifact groups:
      - checkpoints/*.pt
      - params/*
      - events.out.* (TFEvents at top level)

    Uses ``gcloud storage cp`` not rsync (Windows compatibility).

    Args:
        run_dir:    Local destination directory.  Created if it doesn't exist.
        gcs_bucket: GCS bucket URI (default: gs://gg-swarm-training-logs).

    Raises:
        RuntimeError: If ``gcloud`` cannot be found (PATH / ``GGSWARM_GCLOUD``).
    """
    gcloud = resolve_gcloud_executable()
    if gcloud is None:
        raise RuntimeError(gcloud_not_found_message())

    run_name = run_dir.name
    remote_base = f"{gcs_bucket}/{_GCS_PREFIX}/{run_name}"

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)
    (run_dir / "params").mkdir(exist_ok=True)

    print(f"[SYNC] Source: {remote_base}/")
    print(f"[SYNC] Dest  : {run_dir}/")

    for subdir in ("checkpoints", "params"):
        remote_sub = f"{remote_base}/{subdir}/*"
        local_sub = str(run_dir / subdir) + "/"
        print(f"[SYNC]   {subdir}/  ...", end=" ", flush=True)
        result = subprocess.run(
            [gcloud, "storage", "cp", remote_sub, local_sub],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            print("ok")
        else:
            # Non-zero may just mean no files matched; check if anything exists.
            existing = list((run_dir / subdir).glob("*"))
            if not existing:
                print(f"WARN -- {result.stderr.strip()[:120]}")
            else:
                print("ok (already present)")

    # TFEvents files live at the top level of the run directory
    remote_events = f"{remote_base}/events.out.*"
    print("[SYNC]   events.out.*  ...", end=" ", flush=True)
    result = subprocess.run(
        [gcloud, "storage", "cp", remote_events, str(run_dir) + "/"],
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        print("ok")
    else:
        print(f"WARN -- {result.stderr.strip()[:120]}")

    print("[SYNC] Done.")
