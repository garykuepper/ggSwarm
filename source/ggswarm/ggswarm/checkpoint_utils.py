# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared checkpoint-loading helpers for capstone single-agent PPO and MAPPO MARL formats.

Both training (`scripts/skrl/train.py` warm-start) and evaluation
(`scripts/skrl/replay_gate.py`) need to:

1. Load actor weights from either a capstone single-agent PPO checkpoint
   (top-level ``policy`` / ``state_preprocessor`` keys) or a MAPPO MARL
   checkpoint (per-agent dicts keyed ``drone_0..drone_{A-1}`` — all agents
   share weights, so we pull from ``drone_0``).
2. Build a ``RunningStandardScaler`` from the saved running-stats dict.

This module centralizes both so the format-detection branch lives in one
place and any future eval harness can reuse it directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from skrl.resources.preprocessors.torch import RunningStandardScaler
from torch import nn


def load_actor_weights(
    checkpoint_path: str | Path,
    actor: nn.Module,
    device: torch.device | str,
) -> dict[str, torch.Tensor] | None:
    """Load actor weights from a capstone or MAPPO checkpoint into ``actor``.

    Detects the checkpoint format from its top-level keys:

    - ``"policy"`` present → capstone single-agent PPO. Loads ``ckpt["policy"]``.
    - ``"drone_0"`` present → MAPPO MARL. Loads ``ckpt["drone_0"]["policy"]``
      (all agents share weights).
    - Otherwise treats the dict itself as a state_dict.

    Uses ``strict=False`` so a capstone state_dict (which contains both
    ``policy_head`` and ``value_head``) cleanly loads into a policy-only
    actor without raising on the unused value-head keys.

    Args:
        checkpoint_path: Path to the ``.pt`` checkpoint file.
        actor: Module to receive the weights (e.g. ``GgswarmGNNPolicy``).
        device: Torch device for ``map_location``.

    Returns:
        The ``state_preprocessor`` running-stats dict if present in the
        checkpoint, else ``None``. Pass it to :func:`init_state_preprocessor`.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if not isinstance(ckpt, dict):
        policy_sd = ckpt
        sp_stats = None
    elif "policy" in ckpt:
        policy_sd = ckpt["policy"]
        sp_stats = ckpt.get("state_preprocessor")
        print("[INFO] Loading single-agent PPO checkpoint (capstone format)")
    elif "drone_0" in ckpt:
        policy_sd = ckpt["drone_0"]["policy"]
        sp_stats = ckpt["drone_0"].get("state_preprocessor")
        print("[INFO] Loading MAPPO checkpoint (per-agent dict, using drone_0 shared weights)")
    else:
        raise RuntimeError(f"Unrecognized checkpoint structure. Top keys: {list(ckpt.keys())}")

    missing, unexpected = actor.load_state_dict(policy_sd, strict=False)
    if missing:
        print(f"[WARN] Missing keys in actor: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected keys in checkpoint: {unexpected}")
    return sp_stats


def init_state_preprocessor(
    stats_dict: dict[str, torch.Tensor] | None,
    observation_space: Any,
    device: torch.device | str,
) -> RunningStandardScaler | None:
    """Build a SKRL ``RunningStandardScaler`` from saved running-stats.

    Using SKRL's module (rather than manual ``(x - mean) / sqrt(var + eps)``)
    matches the exact preprocessor the agent was trained with — same epsilon
    placement and the +/-clip_threshold clamp. Skipping the clamp drives
    extreme actions on out-of-distribution obs at episode start.

    Args:
        stats_dict: ``state_preprocessor`` running-stats from a checkpoint
            (see :func:`load_actor_weights`). If ``None``, returns ``None``.
        observation_space: Observation space passed to the scaler ``size``.
        device: Torch device.

    Returns:
        A scaler in eval mode, or ``None`` if no stats were provided.
    """
    if stats_dict is None:
        print("[WARN] No state_preprocessor in checkpoint; running unnormalized")
        return None

    scaler = RunningStandardScaler(size=observation_space, device=device)
    scaler.load_state_dict(stats_dict)
    scaler.to(device)
    scaler.train(False)

    rm_avg = scaler.running_mean.mean().item()
    rv_avg = scaler.running_variance.mean().item()
    count = int(scaler.current_count.item())
    print(f"[INFO] Loaded state_preprocessor (mean_avg={rm_avg:.4f}, var_avg={rv_avg:.4f}, count={count})")
    return scaler
