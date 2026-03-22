"""L3 Consensus: SwarmRaft -- decentralised formation consensus for drone swarms.

Pure-torch, no Isaac imports -- safe for unit testing in isolation.

SwarmRaft is a simplified, simulation-friendly adaptation of the Raft consensus
algorithm.  In a real decentralised deployment each agent would maintain its own
state machine and communicate via radio.  In Isaac Lab simulation all agents share
the same GPU tensor store, so we implement the *equivalent* distributed logic
deterministically:

1. **Heartbeat tracking** -- each alive agent resets a per-env counter each step.
   Agents whose counter exceeds `heartbeat_timeout` are declared lost.
2. **Deterministic leader election** -- the lowest-index alive agent per environment
   is the leader.  No election rounds required; avoids split-brain by construction.
3. **Formation redistribution** -- the leader recomputes circular formation slots
   for the remaining alive agents using the chord-length formula identical to
   `_reset_idx`.  New targets are written into the caller's `_desired_pos_w`.
4. **Cooldown guard** -- a per-environment cooldown counter prevents thrashing when
   multiple agents are lost in quick succession.

Caller contract:
- Call `SwarmRaft.tick()` every `raft_tick_interval` physics steps.
- Pass the current `agent_alive` mask derived from the done signals.
- Receive updated `desired_pos_w` and consensus observation features.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SwarmRaftParams:
    """SwarmRaft configuration parameters."""

    heartbeat_timeout: int
    replan_cooldown: int
    target_formation_dist: float
    # Height to use for redistributed formation slots.
    formation_height: float = 1.0


class SwarmRaft:
    """Stateful SwarmRaft consensus manager.

    One instance per environment batch. All state tensors live on `device`.

    Args:
        num_envs: Number of parallel environments.
        num_agents: Number of agents per environment.
        params: SwarmRaft configuration.
        env_origins: World-frame environment origins. Shape [num_envs, 3].
        device: Torch device.
    """

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        params: SwarmRaftParams,
        env_origins: torch.Tensor,
        device: torch.device | str,
    ) -> None:
        self.num_envs = num_envs
        self.num_agents = num_agents
        self.params = params
        self.device = device

        # shape: [num_envs, 3]
        self.env_origins = env_origins.to(device)

        # Heartbeat counters -- incremented each step; reset to 0 when agent is alive.
        # shape: [num_envs, num_agents]
        self.heartbeat = torch.zeros(num_envs, num_agents, dtype=torch.long, device=device)

        # Persistent alive mask -- updated by tick().
        # shape: [num_envs, num_agents]
        self.agent_alive = torch.ones(num_envs, num_agents, dtype=torch.bool, device=device)

        # Cooldown counter per environment (counts down from replan_cooldown to 0).
        # shape: [num_envs]
        self.replan_cooldown = torch.zeros(num_envs, dtype=torch.long, device=device)

        # Leader index per environment (-1 = no leader / all dead).
        # shape: [num_envs]
        self.leader_idx = torch.zeros(num_envs, dtype=torch.long, device=device)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tick(
        self,
        agent_alive_step: torch.Tensor,
        desired_pos_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run one SwarmRaft consensus tick.

        Args:
            agent_alive_step: Bool mask of agents alive this step.
                Shape [num_envs, num_agents].
            desired_pos_w: Current formation goal positions (modified in-place
                when redistribution triggers).
                Shape [num_envs, num_agents, 3].

        Returns:
            desired_pos_w: Updated (or unchanged) formation goals.
                Shape [num_envs, num_agents, 3].
            is_leader: Float indicator -- 1.0 for the leader agent, else 0.0.
                Shape [num_envs, num_agents].
            num_alive_frac: Fraction of original agents still alive (0–1).
                Shape [num_envs, num_agents] -- same value broadcast per env.
        """

        # shape: [num_envs, num_agents]
        if agent_alive_step.shape != (self.num_envs, self.num_agents):
            raise ValueError(
                "agent_alive_step must be [num_envs, num_agents]; "
                f"got {tuple(agent_alive_step.shape)}"
            )

        # 1. Update heartbeats
        self._update_heartbeats(agent_alive_step)

        # 2. Elect leaders
        self._elect_leaders()

        # 3. Redistribution if any environment lost agents and cooldown expired
        desired_pos_w = self._maybe_replan(desired_pos_w)

        # 4. Tick down cooldowns
        self.replan_cooldown = (self.replan_cooldown - 1).clamp(min=0)

        # 5. Build consensus observation features
        is_leader, num_alive_frac = self._build_obs_features()

        return desired_pos_w, is_leader, num_alive_frac

    def reset_envs(self, env_ids: torch.Tensor, desired_pos_w: torch.Tensor) -> None:
        """Reset SwarmRaft state for the given environment indices.

        Called from `_reset_idx` so the consensus state is consistent with the
        freshly spawned formation.

        Args:
            env_ids: Indices of environments to reset. Shape [num_reset].
            desired_pos_w: New formation goals after reset.
                Shape [num_envs, num_agents, 3] (full buffer, indexed by env_ids).
        """

        self.heartbeat[env_ids] = 0
        self.agent_alive[env_ids] = True
        self.replan_cooldown[env_ids] = 0
        self.leader_idx[env_ids] = 0

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_heartbeats(self, agent_alive_step: torch.Tensor) -> None:
        """Increment heartbeat counters for dead agents; reset for alive ones."""

        # shape: [num_envs, num_agents]
        self.heartbeat = torch.where(
            agent_alive_step,
            torch.zeros_like(self.heartbeat),
            self.heartbeat + 1,
        )
        # Declare agent permanently lost when heartbeat exceeds timeout
        timed_out = self.heartbeat >= self.params.heartbeat_timeout
        self.agent_alive = self.agent_alive & ~timed_out

    def _elect_leaders(self) -> None:
        """Elect the lowest-index alive agent as leader per environment."""

        num_envs, num_agents = self.num_envs, self.num_agents
        # Mask dead agents with a large sentinel value so argmin picks alive agents
        sentinel = num_agents + 1
        indices = torch.arange(num_agents, device=self.device).unsqueeze(0).expand(num_envs, -1)
        masked = torch.where(self.agent_alive, indices, torch.full_like(indices, sentinel))
        # shape: [num_envs]
        self.leader_idx = masked.min(dim=-1).values

    def _maybe_replan(self, desired_pos_w: torch.Tensor) -> torch.Tensor:
        """Recompute formation goals for environments where agents were lost.

        Only triggers when:
        - At least one agent is newly declared lost (alive count decreased), AND
        - The environment's replan cooldown has expired (== 0).

        Args:
            desired_pos_w: Shape [num_envs, num_agents, 3].

        Returns:
            Updated desired_pos_w.
        """

        # Count alive agents per environment
        # shape: [num_envs]
        num_alive = self.agent_alive.sum(dim=-1)

        # Environments where cooldown expired and there is at least one alive agent
        needs_replan = (self.replan_cooldown == 0) & (num_alive > 0) & (num_alive < self.num_agents)

        if not needs_replan.any():
            return desired_pos_w

        desired_pos_w = desired_pos_w.clone()
        env_ids = needs_replan.nonzero(as_tuple=False).squeeze(-1)

        for env_id in env_ids:
            env_id = env_id.item()
            alive_mask = self.agent_alive[env_id]  # shape: [num_agents]
            alive_indices = alive_mask.nonzero(as_tuple=False).squeeze(-1)
            n_alive = alive_indices.shape[0]
            if n_alive == 0:
                continue

            # Recompute circular formation for alive agents only
            new_goals = self._compute_circular_formation(
                n_alive=n_alive,
                env_origin=self.env_origins[env_id],
                height=self.params.formation_height,
            )  # shape: [n_alive, 3]

            # Assign new goals to alive agent slots; dead agents keep old goals
            desired_pos_w[env_id, alive_indices] = new_goals

        # Arm cooldown for replanned environments
        self.replan_cooldown[needs_replan] = self.params.replan_cooldown

        return desired_pos_w

    def _compute_circular_formation(
        self,
        n_alive: int,
        env_origin: torch.Tensor,
        height: float,
    ) -> torch.Tensor:
        """Compute circular formation goals for n_alive agents.

        Uses the same chord-length formula as `_reset_idx`:
            r = d / (2 * sin(pi / N))

        Args:
            n_alive: Number of agents to place.
            env_origin: World origin of this environment. Shape [3].
            height: Z coordinate for the formation.

        Returns:
            goals: Shape [n_alive, 3].
        """

        d = self.params.target_formation_dist
        if n_alive == 1:
            # Single agent goes to origin
            goal = env_origin.clone()
            goal[2] = height
            return goal.unsqueeze(0)

        radius = d / (2.0 * math.sin(math.pi / n_alive))
        angles = torch.linspace(
            0.0,
            2.0 * math.pi,
            n_alive + 1,
            device=self.device,
            dtype=torch.float32,
        )[:-1]
        offsets = torch.stack(
            [
                radius * torch.cos(angles),
                radius * torch.sin(angles),
                torch.full((n_alive,), height, device=self.device, dtype=torch.float32),
            ],
            dim=-1,
        )
        # shape: [n_alive, 3]
        goals = env_origin.unsqueeze(0) + offsets
        # Use provided height rather than env_origin z
        goals[:, 2] = height
        return goals

    def _build_obs_features(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Build per-agent consensus observation features.

        Returns:
            is_leader: 1.0 for the leader agent, else 0.0.
                Shape [num_envs, num_agents].
            num_alive_frac: Fraction of original agents still alive.
                Shape [num_envs, num_agents].
        """

        num_envs, num_agents = self.num_envs, self.num_agents

        # shape: [num_envs, num_agents]
        is_leader = torch.zeros(num_envs, num_agents, device=self.device)
        # Set 1.0 for the elected leader (if valid index)
        valid_leader = self.leader_idx < num_agents
        env_ids_valid = valid_leader.nonzero(as_tuple=False).squeeze(-1)
        if env_ids_valid.numel() > 0:
            is_leader[env_ids_valid, self.leader_idx[env_ids_valid]] = 1.0

        # shape: [num_envs]
        num_alive_frac = self.agent_alive.float().mean(dim=-1)
        # Broadcast to [num_envs, num_agents]
        num_alive_frac = num_alive_frac.unsqueeze(-1).expand(num_envs, num_agents)

        return is_leader, num_alive_frac
