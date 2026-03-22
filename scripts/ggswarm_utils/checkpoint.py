"""Checkpoint loading and GNN policy injection utilities.

Centralises the repeated pattern of:
  - Loading a policy state dict into the right module inside an skrl agent
  - Monkey-patching Runner._component to resolve GGSwarmGNNPolicy by name
  - Discovering the best or latest checkpoint in a run directory

These helpers are importable before AppLauncher is constructed; they only
import torch (not isaaclab) at module level.
"""

from __future__ import annotations

from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------


def find_best_checkpoint(run_dir: Path) -> Path:
    """Return the best or latest checkpoint in a run directory.

    Prefers best_agent.pt; falls back to the highest-numbered agent_*.pt.

    Args:
        run_dir: Path to the training run directory containing checkpoints/.

    Returns:
        Absolute path to the checkpoint file.

    Raises:
        FileNotFoundError: If no .pt files are found.
    """
    best = run_dir / "checkpoints" / "best_agent.pt"
    if best.exists():
        return best.resolve()

    ckpts = sorted((run_dir / "checkpoints").glob("agent_*.pt"))
    if not ckpts:
        raise FileNotFoundError(f"No checkpoints found in {run_dir / 'checkpoints'}")
    return ckpts[-1].resolve()


# ---------------------------------------------------------------------------
# Policy loading
# ---------------------------------------------------------------------------


def load_policy_from_checkpoint(agent: object, checkpoint_path: str | Path) -> None:
    """Load a checkpoint's policy weights into an skrl agent.

    skrl's Runner/agent objects don't always expose the policy as
    ``agent.policy``.  We try common locations (agent.policy, agent.models,
    all torch.nn.Module attributes) and load into the first candidate that
    accepts the provided keys.

    Args:
        agent: The resolved skrl agent object (from Runner).
        checkpoint_path: Path to the .pt checkpoint file.

    Raises:
        KeyError: If no policy key is found in the checkpoint, or if loading
                  fails for all candidate modules.
    """
    checkpoint = torch.load(str(checkpoint_path), map_location=agent.device)

    def _load_state_dict(state_dict: dict, source: str) -> bool:
        candidates: list[torch.nn.Module] = []

        def _maybe_add(x: object) -> None:
            if isinstance(x, torch.nn.Module):
                candidates.append(x)

        if hasattr(agent, "policy"):
            _maybe_add(getattr(agent, "policy"))

        if hasattr(agent, "models"):
            models = getattr(agent, "models")
            if isinstance(models, dict):
                policy_like = [m for k, m in models.items() if "policy" in str(k).lower()]
                for m in policy_like if policy_like else list(models.values()):
                    _maybe_add(m)
            else:
                if hasattr(models, "policy"):
                    _maybe_add(getattr(models, "policy"))

        for name in dir(agent):
            try:
                v = getattr(agent, name)
            except Exception:
                continue
            if isinstance(v, torch.nn.Module):
                _maybe_add(v)
            elif isinstance(v, dict):
                for vv in v.values():
                    _maybe_add(vv)

        for m in candidates:
            try:
                m.load_state_dict(state_dict)
                print(f"[CKPT] Policy loaded from {source}")
                return True
            except Exception:
                continue
        return False

    if "policy" in checkpoint:
        if not _load_state_dict(checkpoint["policy"], "checkpoint['policy']"):
            raise KeyError("Could not load policy from checkpoint['policy']")
    elif "policy_0" in checkpoint:
        if not _load_state_dict(checkpoint["policy_0"], "checkpoint['policy_0']"):
            raise KeyError("Could not load policy from checkpoint['policy_0']")
    else:
        loaded = False
        if isinstance(checkpoint, dict):
            for k, v in checkpoint.items():
                if isinstance(v, dict) and "policy" in v:
                    loaded = _load_state_dict(v["policy"], f"checkpoint['{k}']['policy']")
                    if loaded:
                        break
        if not loaded:
            raise KeyError(
                f"Could not find policy in checkpoint. Available keys: {list(checkpoint.keys())}"
            )


# ---------------------------------------------------------------------------
# GNN policy injection
# ---------------------------------------------------------------------------


def inject_gnn_policy(runner_cls: type) -> None:
    """Monkey-patch Runner._component to resolve GGSwarmGNNPolicy by name.

    Must be called after importing Runner but before constructing it.
    Idempotent — safe to call multiple times on the same class.

    Args:
        runner_cls: The skrl Runner class (``skrl.utils.runner.torch.Runner``).
    """
    if getattr(runner_cls, "_ggswarm_gnn_patched", False):
        return

    original_component = runner_cls._component

    def _custom_component(self, name: str):
        if name.lower() == "ggswarmgnnpolicy":
            from ggSwarm.tasks.direct.ggswarm_marl.agents.skrl_gnn_policy import (  # noqa: PLC0415
                GGSwarmGNNPolicy,
            )
            return GGSwarmGNNPolicy
        return original_component(self, name)

    runner_cls._component = _custom_component
    runner_cls._ggswarm_gnn_patched = True


def resolve_agent(runner: object) -> object:
    """Extract the first agent from a Runner (handles both .agent and .agents).

    Args:
        runner: An skrl Runner instance.

    Returns:
        The resolved agent object.

    Raises:
        RuntimeError: If no agent can be found.
    """
    agent = getattr(runner, "agent", None)
    if agent is None:
        agents_attr = getattr(runner, "agents", None)
        if isinstance(agents_attr, (list, tuple)) and agents_attr:
            agent = agents_attr[0]
    if agent is None:
        raise RuntimeError("Could not resolve skrl agent from Runner")
    return agent
