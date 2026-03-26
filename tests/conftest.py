"""Shared pytest fixtures for ggSwarm unit tests.

All fixtures are pure-torch (no Isaac Sim) and run on CPU by default.
Tests are parametrised for CUDA automatically when a GPU is available.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

# ---------------------------------------------------------------------------
# Path injection
# ---------------------------------------------------------------------------
# The top-level ggSwarm package __init__.py imports isaaclab_tasks (a
# simulation dependency that is unavailable during pure-torch unit tests).
# To avoid triggering that import, we inject the ggswarm_marl directory
# directly into sys.path so test files can do:
#   from cbf_safety import ...
# instead of going through the package hierarchy.

_MARL_DIR = Path(__file__).parent.parent / "source" / "ggSwarm" / "ggSwarm" / "tasks" / "direct" / "ggswarm_marl"
if str(_MARL_DIR) not in sys.path:
    sys.path.insert(0, str(_MARL_DIR))

# ---------------------------------------------------------------------------
# Omni / Isaac Sim stubs
# ---------------------------------------------------------------------------
# skrl_gnn_policy.py lives inside the ggSwarm package (uses relative imports),
# so it must be loaded through the package hierarchy rather than as a flat
# module.  The ggSwarm top-level __init__.py and ui_extension_example.py pull
# in `omni.*` and `isaaclab_tasks` which are not available outside Isaac Sim.
# We stub the minimum set of modules before any package import touches them.

_OMNI_STUBS = [
    "isaaclab_tasks",
    "isaaclab_tasks.utils",
    "omni",
    "omni.ext",
    "omni.usd",
    "omni.kit",
]
for _mod_name in _OMNI_STUBS:
    if _mod_name not in sys.modules:
        from unittest.mock import MagicMock
        sys.modules[_mod_name] = MagicMock()

# Also make source/ggSwarm importable as a package root so relative imports
# inside agents/ resolve correctly.
_SOURCE_DIR = Path(__file__).parent.parent / "source" / "ggSwarm"
if str(_SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(_SOURCE_DIR))


# ---------------------------------------------------------------------------
# Device fixture
# ---------------------------------------------------------------------------


def _available_devices() -> list[str]:
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    return devices


@pytest.fixture(params=_available_devices())
def device(request: pytest.FixtureRequest) -> str:
    """Parametrised device: always includes CPU; adds CUDA when available."""
    return request.param


# ---------------------------------------------------------------------------
# Standard problem dimensions
# ---------------------------------------------------------------------------

NUM_ENVS = 4
NUM_AGENTS = 3
ACTION_DIM = 4

# Flight envelope used for position tensors
POS_Z_MIN = 0.3    # metres above ground
POS_Z_MAX = 2.5
POS_XY_RANGE = 2.0  # ± metres in XY


# ---------------------------------------------------------------------------
# Tensor factory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def make_pos_w(device):
    """Factory: return random [num_envs, num_agents, 3] position tensors.

    Positions are sampled within the normal flight envelope so that
    tests start in a physically reasonable state by default.
    """

    torch.manual_seed(0)

    def _make(
        num_envs: int = NUM_ENVS,
        num_agents: int = NUM_AGENTS,
        *,
        spread: float = POS_XY_RANGE,
        z_min: float = POS_Z_MIN,
        z_max: float = POS_Z_MAX,
    ) -> torch.Tensor:
        # shape: [num_envs, num_agents, 3]
        pos = torch.zeros(num_envs, num_agents, 3, device=device)
        pos[..., :2] = torch.rand(num_envs, num_agents, 2, device=device) * 2 * spread - spread
        pos[..., 2] = torch.rand(num_envs, num_agents, device=device) * (z_max - z_min) + z_min
        return pos

    return _make


@pytest.fixture
def make_actions(device):
    """Factory: return random [num_envs, num_agents, action_dim] action tensors clamped to [-1, 1]."""

    torch.manual_seed(1)

    def _make(
        num_envs: int = NUM_ENVS,
        num_agents: int = NUM_AGENTS,
        action_dim: int = ACTION_DIM,
    ) -> torch.Tensor:
        return (torch.rand(num_envs, num_agents, action_dim, device=device) * 2 - 1).clamp(-1, 1)

    return _make


@pytest.fixture
def make_vel(device):
    """Factory: return small random [num_envs, num_agents, 3] velocity tensors (m/s)."""

    torch.manual_seed(2)

    def _make(
        num_envs: int = NUM_ENVS,
        num_agents: int = NUM_AGENTS,
        max_speed: float = 0.5,
    ) -> torch.Tensor:
        return (torch.rand(num_envs, num_agents, 3, device=device) * 2 - 1) * max_speed

    return _make


# ---------------------------------------------------------------------------
# Standard scenario fixtures (pre-built tensors for quick use)
# ---------------------------------------------------------------------------


@pytest.fixture
def std_pos_w(make_pos_w) -> torch.Tensor:
    """Standard random position tensor [NUM_ENVS, NUM_AGENTS, 3]."""
    return make_pos_w()


@pytest.fixture
def std_actions(make_actions) -> torch.Tensor:
    """Standard random action tensor [NUM_ENVS, NUM_AGENTS, ACTION_DIM]."""
    return make_actions()


@pytest.fixture
def std_vel(make_vel) -> torch.Tensor:
    """Standard random velocity tensor [NUM_ENVS, NUM_AGENTS, 3]."""
    return make_vel()


# ---------------------------------------------------------------------------
# Geometry helpers (reused across test files)
# ---------------------------------------------------------------------------


def agents_far_apart(num_envs: int, num_agents: int, spacing: float, device: str) -> torch.Tensor:
    """Place agents on a line, separated by `spacing` metres in X.

    Guaranteed: min pairwise distance == spacing.

    Returns:
        pos_w: shape [num_envs, num_agents, 3]
    """
    pos = torch.zeros(num_envs, num_agents, 3, device=device)
    for i in range(num_agents):
        pos[:, i, 0] = i * spacing
        pos[:, i, 2] = 1.0  # hover height
    return pos


def agents_at_same_pos(num_envs: int, num_agents: int, device: str) -> torch.Tensor:
    """All agents co-located at the origin (worst-case for CBF)."""
    return torch.zeros(num_envs, num_agents, 3, device=device)


def circular_formation(
    num_envs: int,
    num_agents: int,
    target_dist: float,
    device: str,
    height: float = 1.0,
) -> torch.Tensor:
    """Arrange agents in a perfect circle with chord-length == target_dist."""
    radius = target_dist / (2.0 * math.sin(math.pi / num_agents))
    angles = torch.linspace(0, 2 * math.pi, num_agents + 1, device=device)[:-1]
    pos = torch.zeros(num_envs, num_agents, 3, device=device)
    pos[:, :, 0] = radius * torch.cos(angles)
    pos[:, :, 1] = radius * torch.sin(angles)
    pos[:, :, 2] = height
    return pos
