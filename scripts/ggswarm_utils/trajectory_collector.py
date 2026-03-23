"""Standalone trajectory data collector implementing the PhaseCollector protocol.

Accumulates per-step position and (optionally) attitude data during eval,
without requiring inheritance from Phase2Collector.  Attach via
``CompositeCollector`` to piggyback on the primary eval loop.

Public API:
    TrajectoryDataCollector(track_envs=1, euler_fn=None)
"""

from __future__ import annotations

from typing import Callable

import torch


class TrajectoryDataCollector:
    """Collect per-step ``root_pos_w`` and optional roll/pitch trajectories.

    Implements the :class:`PhaseCollector` protocol
    (``on_step``, ``on_episode_end``, ``summarize``).

    Trajectory data is accessed directly via the public attributes
    ``pos_traj``, ``roll_traj``, and ``pitch_traj`` — each a dict mapping
    ``episode_num`` to a list of CPU tensors.

    Args:
        track_envs: Number of environments to record (first *N* envs).
        euler_fn:   Optional ``euler_xyz_from_quat(quat) -> (roll, pitch, yaw)``
                    callable.  When *None*, attitude data is not collected.
    """

    def __init__(
        self,
        track_envs: int = 1,
        euler_fn: Callable | None = None,
    ) -> None:
        self._track = track_envs
        self._euler_fn = euler_fn

        # episode_num -> list of [track_envs, num_agents, 3] cpu tensors
        self.pos_traj: dict[int, list[torch.Tensor]] = {}
        # episode_num -> list of [track_envs, num_agents] cpu tensors (degrees)
        self.roll_traj: dict[int, list[torch.Tensor]] = {}
        self.pitch_traj: dict[int, list[torch.Tensor]] = {}

    # ------------------------------------------------------------------
    # PhaseCollector protocol
    # ------------------------------------------------------------------

    def on_step(
        self,
        *,
        base_env: object,
        obs: object,
        step: int,
        episode_step: int,
        episode_num: int,
    ) -> None:
        """Record position (and optionally attitude) for tracked envs."""
        n_env = base_env.num_envs
        n_ag = base_env.cfg.num_agents
        t = min(self._track, n_env)

        pos_w = base_env.robot.data.root_pos_w.view(n_env, n_ag, 3)

        if episode_num not in self.pos_traj:
            self.pos_traj[episode_num] = []
            self.roll_traj[episode_num] = []
            self.pitch_traj[episode_num] = []

        self.pos_traj[episode_num].append(pos_w[:t].detach().cpu().clone())

        if self._euler_fn is not None:
            quat_w = base_env.robot.data.root_quat_w.view(n_env, n_ag, 4)
            roll_rad, pitch_rad, _ = self._euler_fn(
                quat_w[:t].reshape(-1, 4)
            )
            self.roll_traj[episode_num].append(
                torch.rad2deg(roll_rad.reshape(t, n_ag)).detach().cpu().clone()
            )
            self.pitch_traj[episode_num].append(
                torch.rad2deg(pitch_rad.reshape(t, n_ag)).detach().cpu().clone()
            )

    def on_episode_end(self, episode_num: int) -> None:
        """No-op — trajectory data is already stored per-step."""

    def summarize(self) -> dict[str, float]:
        """Return empty dict — trajectory data is accessed via attributes."""
        return {}
