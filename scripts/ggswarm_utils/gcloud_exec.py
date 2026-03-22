"""Resolve the ``gcloud`` executable for subprocess use (Windows-safe).

Isaac Lab / IDE terminals often omit the Cloud SDK from ``PATH``. On Windows the
launcher is ``gcloud.cmd`` under ``%LOCALAPPDATA%\\Google\\Cloud SDK\\...``.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


def gcloud_not_found_message() -> str:
    """Human-readable hint when ``gcloud`` cannot be located."""
    return (
        "gcloud was not found.\n"
        "  • Install Google Cloud SDK, then add its "
        r"'google-cloud-sdk\bin' folder to PATH, or"
        "\n  • Set GGSWARM_GCLOUD to the full path of gcloud "
        r"(Windows: ...\Google\Cloud SDK\google-cloud-sdk\bin\gcloud.cmd)."
    )


def resolve_gcloud_executable() -> str | None:
    """Return a path to ``gcloud`` for use with ``subprocess`` (no shell)."""
    override = (os.environ.get("GGSWARM_GCLOUD") or os.environ.get("GCLOUD_PATH") or "").strip()
    if override:
        p = Path(override)
        if p.is_file():
            return str(p)
        if sys.platform == "win32" and p.suffix.lower() != ".cmd":
            p_cmd = Path(str(p) + ".cmd")
            if p_cmd.is_file():
                return str(p_cmd)

    for name in ("gcloud", "gcloud.cmd"):
        found = shutil.which(name)
        if found:
            return found

    if sys.platform == "win32":
        localappdata = os.environ.get("LOCALAPPDATA", "")
        program_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        candidates = [
            Path(localappdata)
            / "Google"
            / "Cloud SDK"
            / "google-cloud-sdk"
            / "bin"
            / "gcloud.cmd",
            Path(program_files)
            / "Google"
            / "Cloud SDK"
            / "google-cloud-sdk"
            / "bin"
            / "gcloud.cmd",
            Path(program_files_x86)
            / "Google"
            / "Cloud SDK"
            / "google-cloud-sdk"
            / "bin"
            / "gcloud.cmd",
        ]
        for c in candidates:
            if c.is_file():
                return str(c)

    return None
