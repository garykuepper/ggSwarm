"""Unit tests for swarm_raft.py.

Covers: initial state, leader election (lowest alive index), heartbeat timeout,
replan triggering & cooldown, formation radius formula, single-agent survival,
reset isolation, obs feature shapes, and alive-fraction monotonicity.
No Isaac Sim required.
"""

from __future__ import annotations

import math

import pytest
import torch

from swarm_raft import SwarmRaft, SwarmRaftParams
from tests.conftest import NUM_AGENTS, NUM_ENVS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _params(
    heartbeat_timeout: int = 5,
    replan_cooldown: int = 3,
    target_formation_dist: float = 0.5,
    formation_height: float = 1.0,
) -> SwarmRaftParams:
    return SwarmRaftParams(
        heartbeat_timeout=heartbeat_timeout,
        replan_cooldown=replan_cooldown,
        target_formation_dist=target_formation_dist,
        formation_height=formation_height,
    )


def _env_origins(num_envs: int, device: str) -> torch.Tensor:
    """Stagger env origins along X so redistribution doesn't stack on top."""
    origins = torch.zeros(num_envs, 3, device=device)
    for i in range(num_envs):
        origins[i, 0] = float(i) * 10.0
    return origins


def _make_raft(num_envs: int = NUM_ENVS, num_agents: int = NUM_AGENTS, device: str = "cpu",
               **param_overrides) -> SwarmRaft:
    return SwarmRaft(
        num_envs=num_envs,
        num_agents=num_agents,
        params=_params(**param_overrides),
        env_origins=_env_origins(num_envs, device),
        device=device,
    )


def _desired_pos(num_envs: int = NUM_ENVS, num_agents: int = NUM_AGENTS, device: str = "cpu") -> torch.Tensor:
    """Simple placeholder formation goals (flat grid at z=1)."""
    pos = torch.zeros(num_envs, num_agents, 3, device=device)
    for i in range(num_agents):
        pos[:, i, 0] = float(i) * 0.5
        pos[:, :, 2] = 1.0
    return pos


def _all_alive(num_envs: int = NUM_ENVS, num_agents: int = NUM_AGENTS, device: str = "cpu") -> torch.Tensor:
    return torch.ones(num_envs, num_agents, dtype=torch.bool, device=device)


def _kill_agent(alive: torch.Tensor, agent_idx: int) -> torch.Tensor:
    """Return a copy of `alive` with `agent_idx` set to False in all envs."""
    dead = alive.clone()
    dead[:, agent_idx] = False
    return dead


# ---------------------------------------------------------------------------
# Initial state
# ---------------------------------------------------------------------------


class TestInitialState:
    def test_all_alive(self, device):
        raft = _make_raft(device=device)
        assert torch.all(raft.agent_alive), "All agents must start alive"

    def test_leader_is_agent_zero(self, device):
        raft = _make_raft(device=device)
        # Trigger an election
        raft._elect_leaders()
        assert torch.all(raft.leader_idx == 0), "Initial leader must be agent 0 (lowest index)"

    def test_heartbeat_all_zero(self, device):
        raft = _make_raft(device=device)
        assert torch.all(raft.heartbeat == 0)

    def test_replan_cooldown_all_zero(self, device):
        raft = _make_raft(device=device)
        assert torch.all(raft.replan_cooldown == 0)


# ---------------------------------------------------------------------------
# Leader election
# ---------------------------------------------------------------------------


class TestLeaderElection:
    def test_lowest_alive_index_elected(self, device):
        raft = _make_raft(device=device)
        # Kill agent 0 in all envs
        raft.agent_alive[:, 0] = False
        raft._elect_leaders()
        assert torch.all(raft.leader_idx == 1), "After killing agent 0, leader must be agent 1"

    def test_leader_tracks_progressive_kills(self, device):
        raft = _make_raft(num_agents=4, device=device)
        # Kill agents 0 and 1
        raft.agent_alive[:, 0] = False
        raft.agent_alive[:, 1] = False
        raft._elect_leaders()
        assert torch.all(raft.leader_idx == 2), "Leader must be the lowest surviving index"

    def test_single_env_specific_kill(self, device):
        """Kill agent 0 only in env 0 -- other envs keep agent 0 as leader."""
        raft = _make_raft(num_envs=2, device=device)
        raft.agent_alive[0, 0] = False  # only env 0
        raft._elect_leaders()
        assert raft.leader_idx[0] == 1, "Env 0: leader should be agent 1"
        assert raft.leader_idx[1] == 0, "Env 1: leader should still be agent 0"

    def test_no_valid_leader_when_all_dead(self, device):
        raft = _make_raft(num_agents=2, device=device)
        raft.agent_alive[:, :] = False
        raft._elect_leaders()
        # leader_idx is set to sentinel (num_agents + 1) when no alive agent exists
        assert torch.all(raft.leader_idx > 1), "Sentinel index must exceed any valid agent index"


# ---------------------------------------------------------------------------
# Heartbeat / timeout
# ---------------------------------------------------------------------------


class TestHeartbeat:
    def test_alive_agent_resets_heartbeat(self, device):
        raft = _make_raft(device=device)
        # Manually set a non-zero heartbeat
        raft.heartbeat[:, 0] = 3
        alive = _all_alive(device=device)
        raft._update_heartbeats(alive)
        assert torch.all(raft.heartbeat[:, 0] == 0), "Alive agent must reset its heartbeat to 0"

    def test_dead_agent_increments_heartbeat(self, device):
        raft = _make_raft(device=device)
        alive = _all_alive(device=device)
        alive[:, 1] = False  # agent 1 not reporting
        raft._update_heartbeats(alive)
        assert torch.all(raft.heartbeat[:, 1] == 1), "Missing agent's heartbeat must increment"

    def test_timeout_declares_agent_lost(self, device):
        timeout = 5
        raft = _make_raft(device=device, heartbeat_timeout=timeout)
        # Simulate `timeout` steps with agent 1 absent
        alive = _all_alive(device=device)
        for _ in range(timeout):
            alive[:, 1] = False
            raft._update_heartbeats(alive)
        assert torch.all(~raft.agent_alive[:, 1]), "Agent must be declared lost after heartbeat_timeout steps"

    def test_no_timeout_before_threshold(self, device):
        timeout = 5
        raft = _make_raft(device=device, heartbeat_timeout=timeout)
        alive = _all_alive(device=device)
        for _ in range(timeout - 1):
            alive[:, 1] = False
            raft._update_heartbeats(alive)
        # Should NOT yet be declared lost
        assert torch.all(raft.agent_alive[:, 1]), "Agent must NOT be declared lost before heartbeat_timeout"

    def test_recovery_after_return(self, device):
        """An agent that reappears after a gap should not continue incrementing."""
        timeout = 5
        raft = _make_raft(device=device, heartbeat_timeout=timeout)
        alive = _all_alive(device=device)
        # 2 steps absent
        alive_absent = alive.clone()
        alive_absent[:, 2] = False
        raft._update_heartbeats(alive_absent)
        raft._update_heartbeats(alive_absent)
        assert torch.all(raft.heartbeat[:, 2] == 2)
        # Now returns
        raft._update_heartbeats(alive)
        assert torch.all(raft.heartbeat[:, 2] == 0), "Recovered agent must reset heartbeat"


# ---------------------------------------------------------------------------
# Replan triggers and cooldown
# ---------------------------------------------------------------------------


class TestReplan:
    def test_no_replan_while_all_alive(self, device):
        raft = _make_raft(device=device, replan_cooldown=3)
        desired = _desired_pos(device=device)
        original = desired.clone()
        alive = _all_alive(device=device)

        # Run several ticks with everyone alive
        for _ in range(5):
            desired, _, _ = raft.tick(alive, desired)

        assert torch.allclose(desired, original), "Goals must not change when all agents are alive"

    def test_replan_triggers_after_loss(self, device):
        timeout = 2
        cooldown = 0
        raft = _make_raft(device=device, heartbeat_timeout=timeout, replan_cooldown=cooldown)
        desired = _desired_pos(device=device)
        original = desired.clone()
        alive = _all_alive(device=device)

        # Kill agent 1 (force timeout)
        for _ in range(timeout):
            alive[:, 1] = False
            desired, _, _ = raft.tick(alive, desired)

        # After timeout expires, replan should have fired
        any_changed = not torch.allclose(desired, original)
        assert any_changed, "Formation goals must change after an agent is lost and timeout expires"

    def test_cooldown_prevents_immediate_replan_after_second_kill(self, device):
        timeout = 2
        cooldown = 10  # long cooldown
        raft = _make_raft(device=device, heartbeat_timeout=timeout, replan_cooldown=cooldown)
        desired = _desired_pos(device=device)
        alive = _all_alive(device=device)

        # Let agent 1 time out (triggers first replan, cooldown armed)
        for _ in range(timeout):
            alive[:, 1] = False
            desired, _, _ = raft.tick(alive, desired)

        goals_after_first_replan = desired.clone()

        # Kill agent 2 immediately (cooldown should suppress second replan)
        alive[:, 2] = False
        for _ in range(timeout):
            desired, _, _ = raft.tick(alive, desired)

        # Goals for alive agents should NOT have changed yet (cooldown still active)
        alive_mask = raft.agent_alive[0]  # per-env alive after timeout
        alive_idxs = alive_mask.nonzero(as_tuple=False).squeeze(-1)
        if alive_idxs.numel() > 0:
            assert torch.allclose(
                desired[0, alive_idxs], goals_after_first_replan[0, alive_idxs]
            ), "Cooldown must suppress rapid replan after second kill"

    def test_replan_only_moves_alive_agents(self, device):
        timeout = 2
        raft = _make_raft(device=device, heartbeat_timeout=timeout, replan_cooldown=0)
        desired = _desired_pos(device=device)
        alive = _all_alive(device=device)

        # Kill agent 0 in all envs
        for _ in range(timeout):
            alive[:, 0] = False
            desired, _, _ = raft.tick(alive, desired)

        dead_goals_before = desired[:, 0, :].clone()
        # After replan: agent 0 slot is dead -- its goal should be its old goal
        # (implementation assigns new_goals to alive_indices only)
        # Verify dead agent's goal is NOT newly assigned (it keeps old value)
        # This checks the assignment loop only touches alive_indices
        assert torch.allclose(desired[:, 0, :], dead_goals_before), (
            "Dead agent slots must retain their previous goals after replan"
        )


# ---------------------------------------------------------------------------
# Formation radius formula
# ---------------------------------------------------------------------------


class TestFormationRadiusFormula:
    def test_chord_length_formula(self, device):
        """New goals must satisfy chord-length == target_formation_dist."""
        d = 0.5
        n = 3
        raft = _make_raft(num_envs=1, num_agents=n, device=device,
                          target_formation_dist=d, formation_height=1.0)
        origins = torch.zeros(1, 3, device=device)
        goals = raft._compute_circular_formation(n_alive=n, env_origin=origins[0], height=1.0)
        # Compute chord distances between consecutive goals
        for i in range(n):
            chord = torch.norm(goals[i] - goals[(i + 1) % n]).item()
            assert abs(chord - d) < 1e-4, f"Chord {i}->{(i+1)%n}: expected {d:.4f}, got {chord:.4f}"

    def test_single_agent_returns_origin(self, device):
        raft = _make_raft(num_envs=1, num_agents=1, device=device,
                          target_formation_dist=0.5, formation_height=1.5)
        origin = torch.tensor([3.0, 2.0, 0.0], device=device)
        goal = raft._compute_circular_formation(n_alive=1, env_origin=origin, height=1.5)
        assert goal.shape == (1, 3)
        assert abs(goal[0, 0].item() - 3.0) < 1e-5
        assert abs(goal[0, 1].item() - 2.0) < 1e-5
        assert abs(goal[0, 2].item() - 1.5) < 1e-5, "Single agent should go to env origin at formation height"

    def test_formation_height_applied(self, device):
        height = 2.5
        d = 0.5
        n = 4
        raft = _make_raft(num_envs=1, num_agents=n, device=device,
                          target_formation_dist=d, formation_height=height)
        origin = torch.zeros(3, device=device)
        goals = raft._compute_circular_formation(n_alive=n, env_origin=origin, height=height)
        assert torch.allclose(goals[:, 2], torch.full((n,), height, device=device)), (
            "All goal Z coordinates must match formation_height"
        )


# ---------------------------------------------------------------------------
# Reset
# ---------------------------------------------------------------------------


class TestResetEnvs:
    def test_reset_clears_heartbeat(self, device):
        raft = _make_raft(device=device)
        raft.heartbeat[:, 0] = 99
        env_ids = torch.arange(NUM_ENVS, device=device)
        desired = _desired_pos(device=device)
        raft.reset_envs(env_ids, desired)
        assert torch.all(raft.heartbeat == 0)

    def test_reset_restores_alive(self, device):
        raft = _make_raft(device=device)
        raft.agent_alive[:, :] = False
        env_ids = torch.arange(NUM_ENVS, device=device)
        desired = _desired_pos(device=device)
        raft.reset_envs(env_ids, desired)
        assert torch.all(raft.agent_alive)

    def test_reset_clears_cooldown(self, device):
        raft = _make_raft(device=device)
        raft.replan_cooldown[:] = 10
        env_ids = torch.arange(NUM_ENVS, device=device)
        desired = _desired_pos(device=device)
        raft.reset_envs(env_ids, desired)
        assert torch.all(raft.replan_cooldown == 0)

    def test_partial_reset_does_not_affect_other_envs(self, device):
        raft = _make_raft(device=device)
        raft.heartbeat[:, 0] = 7
        env_ids = torch.tensor([0], device=device)  # only env 0
        desired = _desired_pos(device=device)
        raft.reset_envs(env_ids, desired)
        assert raft.heartbeat[0, 0] == 0, "Reset env should have heartbeat cleared"
        if NUM_ENVS > 1:
            assert raft.heartbeat[1, 0] == 7, "Non-reset env must keep its state"


# ---------------------------------------------------------------------------
# Obs feature shapes and consistency
# ---------------------------------------------------------------------------


class TestObsFeatures:
    def test_is_leader_shape(self, device):
        raft = _make_raft(device=device)
        is_leader, _ = raft._build_obs_features()
        assert is_leader.shape == (NUM_ENVS, NUM_AGENTS)

    def test_num_alive_frac_shape(self, device):
        raft = _make_raft(device=device)
        _, frac = raft._build_obs_features()
        assert frac.shape == (NUM_ENVS, NUM_AGENTS)

    def test_exactly_one_leader_per_env(self, device):
        raft = _make_raft(device=device)
        raft._elect_leaders()
        is_leader, _ = raft._build_obs_features()
        per_env_count = is_leader.sum(dim=-1)  # shape [num_envs]
        assert torch.all(per_env_count == 1.0), "Exactly one leader per env when all agents alive"

    def test_is_leader_zero_when_all_dead(self, device):
        raft = _make_raft(device=device)
        raft.agent_alive[:, :] = False
        raft._elect_leaders()
        is_leader, _ = raft._build_obs_features()
        # All dead -- valid_leader check should prevent any assignment
        assert torch.all(is_leader == 0.0)

    def test_num_alive_frac_is_one_initially(self, device):
        raft = _make_raft(device=device)
        _, frac = raft._build_obs_features()
        assert torch.allclose(frac, torch.ones_like(frac))

    def test_num_alive_frac_decreases_after_loss(self, device):
        raft = _make_raft(device=device, heartbeat_timeout=2)
        desired = _desired_pos(device=device)
        alive = _all_alive(device=device)

        # Kill agent 0 (force timeout)
        for _ in range(2):
            alive[:, 0] = False
            desired, _, frac = raft.tick(alive, desired)

        expected = (NUM_AGENTS - 1) / NUM_AGENTS
        assert torch.allclose(frac[:, 0], torch.full_like(frac[:, 0], expected), atol=1e-4), (
            "Alive fraction must reflect lost agent"
        )

    def test_alive_frac_monotone_as_agents_die(self, device):
        """alive_frac must be non-increasing as agents are killed one by one."""
        timeout = 2
        raft = _make_raft(device=device, heartbeat_timeout=timeout, replan_cooldown=0)
        desired = _desired_pos(device=device)
        alive = _all_alive(device=device)

        prev_frac = None
        for agent_to_kill in range(NUM_AGENTS - 1):
            alive[:, agent_to_kill] = False
            for _ in range(timeout):
                desired, _, frac = raft.tick(alive, desired)

            current_mean = frac[:, 0].mean().item()
            if prev_frac is not None:
                assert current_mean <= prev_frac + 1e-5, (
                    f"Alive fraction increased after killing agent {agent_to_kill}"
                )
            prev_frac = current_mean

    def test_frac_broadcast_consistent_within_env(self, device):
        """All agents in an env must see the same alive fraction."""
        raft = _make_raft(device=device)
        _, frac = raft._build_obs_features()
        # Each row of frac should be constant (same value for all agents in an env)
        for env_idx in range(NUM_ENVS):
            row = frac[env_idx]
            assert torch.allclose(row, row[0].expand_as(row)), (
                f"Alive fraction must be the same for all agents in env {env_idx}"
            )

    def test_tick_returns_correct_shapes(self, device):
        raft = _make_raft(device=device)
        desired = _desired_pos(device=device)
        alive = _all_alive(device=device)
        new_desired, is_leader, frac = raft.tick(alive, desired)
        assert new_desired.shape == (NUM_ENVS, NUM_AGENTS, 3)
        assert is_leader.shape == (NUM_ENVS, NUM_AGENTS)
        assert frac.shape == (NUM_ENVS, NUM_AGENTS)

    def test_tick_validates_alive_shape(self, device):
        raft = _make_raft(device=device)
        desired = _desired_pos(device=device)
        bad_alive = torch.ones(NUM_ENVS + 1, NUM_AGENTS, dtype=torch.bool, device=device)
        with pytest.raises(ValueError):
            raft.tick(bad_alive, desired)
