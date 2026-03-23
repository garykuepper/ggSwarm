"""Composite collector that delegates to multiple PhaseCollector instances.

Allows trajectory data collection to piggyback on the same eval loop as the
primary metric collector without running a second simulation.

Public API:
    CompositeCollector(primary, *extras)
"""

from __future__ import annotations


class CompositeCollector:
    """Fan-out wrapper: delegates ``on_step`` / ``on_episode_end`` to every
    wrapped collector, and returns the *primary* collector's ``summarize()``
    result.

    The primary collector is always called **first** in ``on_step`` and
    ``on_episode_end`` so that its internal state (e.g.
    ``Phase2Collector._episode_step_count``) is updated before any secondary
    collector reads ``base_env`` attributes.

    Args:
        primary: The main metric collector whose ``summarize()`` will be used.
        *extras: Additional collectors (e.g. ``TrajectoryDataCollector``).
    """

    def __init__(self, primary: object, *extras: object) -> None:
        self._primary = primary
        self._extras = list(extras)
        self._all = [primary, *extras]

    def on_step(
        self,
        *,
        base_env: object,
        obs: object,
        step: int,
        episode_step: int,
        episode_num: int,
    ) -> None:
        """Delegate to all collectors, primary first."""
        for c in self._all:
            c.on_step(
                base_env=base_env,
                obs=obs,
                step=step,
                episode_step=episode_step,
                episode_num=episode_num,
            )

    def on_episode_end(self, episode_num: int) -> None:
        """Delegate to all collectors, primary first."""
        for c in self._all:
            c.on_episode_end(episode_num)

    def summarize(self) -> dict[str, float]:
        """Return the primary collector's metrics only."""
        return self._primary.summarize()
