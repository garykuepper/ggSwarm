"""Shared Isaac Lab simulation helpers for all eval/play/bench scripts.

Eliminates cross-script duplication of:
  - GNN policy configuration (configure_gnn_policy)
  - Agent count override (override_agent_count)
  - Action extraction from agent outputs (extract_actions)
  - Eval cfg overrides (configure_eval_cfg)
  - Phase-to-task mapping (PHASE_REGISTRY / resolve_phase)

IMPORTANT: This module does NOT import Isaac Lab or torch at module level.
All sim-dependent types are accepted as arguments so the module remains
importable and testable without a GPU or Isaac Lab installation.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Phase registry (single source of truth for task IDs)
# ---------------------------------------------------------------------------


@dataclass
class PhaseConfig:
    """Immutable descriptor for a training/eval phase."""

    task: str
    gnn_default: bool
    collector_cls: str  # dotted import path relative to ggswarm_utils package
    default_episodes: int = 5
    description: str = ""


# Maps CLI ``--phase`` values to their full configuration.
# Phase 2C is a placeholder pointing at the base Phase 2 task until its own
# curriculum config and task ID are defined (TODO Phase 3+).
PHASE_REGISTRY: dict[str, PhaseConfig] = {
    "hover": PhaseConfig(
        task="GGS-Hover-v0",
        gnn_default=False,
        collector_cls="ggswarm_utils.phases.hover.HoverCollector",
        default_episodes=10,
        description="Hover baseline (single-agent, no formation)",
    ),
    "2": PhaseConfig(
        task="Template-GGSwarm-Marl-Direct-v0",
        gnn_default=True,
        collector_cls="ggswarm_utils.phases.phase2.Phase2Collector",
        description="Phase 2 formation (standard task)",
    ),
    "2a": PhaseConfig(
        task="Template-GGSwarm-Marl-HoverStability-v0",
        gnn_default=True,
        collector_cls="ggswarm_utils.phases.phase2.Phase2Collector",
        description="Phase 2A hover-stability (formation rewards disabled)",
    ),
    "2b": PhaseConfig(
        task="Template-GGSwarm-Marl-Formation-v0",
        gnn_default=True,
        collector_cls="ggswarm_utils.phases.phase2.Phase2Collector",
        description="Phase 2B formation (resume from Phase 2A checkpoint)",
    ),
    "2c": PhaseConfig(
        task="Template-GGSwarm-Marl-Direct-v0",
        gnn_default=True,
        collector_cls="ggswarm_utils.phases.phase2.Phase2Collector",
        description="Phase 2C perturbation (TODO: placeholder, uses Phase 2 task)",
    ),
    "3": PhaseConfig(
        task="Template-GGSwarm-Marl-Phase3-v0",
        gnn_default=True,
        collector_cls="ggswarm_utils.phases.phase3.Phase3Collector",
        default_episodes=10,
        description="Phase 3 muscle refinement (CBF, MINCO, SwarmRaft)",
    ),
    "4": PhaseConfig(
        task="Template-GGSwarm-Marl-Phase4-v0",
        gnn_default=True,
        # intentional: phase4 reuses Phase3Collector metrics (Rule 16)
        collector_cls="ggswarm_utils.phases.phase3.Phase3Collector",
        default_episodes=10,
        description="Phase 4 stress testing (agent loss scenarios)",
    ),
}


def resolve_phase(phase: str) -> PhaseConfig:
    """Return the PhaseConfig for a given phase key.

    Args:
        phase: One of the keys in PHASE_REGISTRY (e.g. '2a', 'hover').

    Returns:
        The matching PhaseConfig.

    Raises:
        ValueError: If the phase key is not in PHASE_REGISTRY.
    """
    if phase not in PHASE_REGISTRY:
        known = ", ".join(sorted(PHASE_REGISTRY))
        raise ValueError(
            f"Unknown phase '{phase}'. Known phases: {known}"
        )
    return PHASE_REGISTRY[phase]


def phase_from_task(task: str) -> str | None:
    """Reverse-look up the phase key for a given Gym task ID.

    Returns the first matching phase key or None if not found.
    Useful when the caller only has the task ID (e.g. post_train_assess.py).
    """
    for key, cfg in PHASE_REGISTRY.items():
        if cfg.task == task:
            return key
    return None


# ---------------------------------------------------------------------------
# GNN policy configuration
# ---------------------------------------------------------------------------


def configure_gnn_policy(cfg: dict, runner_cls: type) -> None:
    """Configure the skrl runner cfg to use GGSwarmGNNPolicy and patch Runner.

    Replaces the 4-line block that was copy-pasted into 6 scripts:
        cfg["models"]["policy"]["class"] = "GGSwarmGNNPolicy"
        cfg["models"]["policy"].pop("network", None)
        inject_gnn_policy(Runner)

    Args:
        cfg:        The skrl experiment config dict (modified in-place).
        runner_cls: The skrl Runner class to monkey-patch.
    """
    policy = cfg["models"]["policy"]
    policy["class"] = "GGSwarmGNNPolicy"
    policy.pop("network", None)
    # Match train.py: GNN-only kwargs must exist before skrl builds GGSwarmGNNPolicy.
    policy.setdefault("hidden_channels", 128)
    policy.setdefault("num_heads", 2)
    from ggswarm_utils.checkpoint import inject_gnn_policy  # noqa: PLC0415
    inject_gnn_policy(runner_cls)


# ---------------------------------------------------------------------------
# Agent count override
# ---------------------------------------------------------------------------


def override_agent_count(
    env_cfg: object,
    num_agents: int,
    obs_dim: int = 12,
) -> None:
    """Override num_agents and rebuild the dependent config fields.

    Replaces the 4-line block that was copy-pasted into 5 scripts:
        env_cfg.num_agents = N
        env_cfg.possible_agents = [f"drone_{i}" for i in range(N)]
        env_cfg.action_spaces = {a: 4 for a in env_cfg.possible_agents}
        env_cfg.observation_spaces = {a: obs_dim for a in env_cfg.possible_agents}

    Args:
        env_cfg:    The IsaacLab env config object (modified in-place).
        num_agents: New number of agents.
        obs_dim:    Observation dimension per agent (default 12; use 14 for
                    Phase 3 tasks with raft_enabled=True).
    """
    env_cfg.num_agents = num_agents  # type: ignore[attr-defined]
    env_cfg.possible_agents = [f"drone_{i}" for i in range(num_agents)]  # type: ignore[attr-defined]
    env_cfg.action_spaces = {a: 4 for a in env_cfg.possible_agents}  # type: ignore[attr-defined]
    env_cfg.observation_spaces = {a: obs_dim for a in env_cfg.possible_agents}  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Action extraction
# ---------------------------------------------------------------------------


def extract_actions(agent: object, obs: object, base_env: object) -> object:
    """Run agent.act() and return **mean** Gaussian actions when skrl exposes them.

    ``GaussianMixin.act`` returns sampled actions in the first tuple slot but
    attaches ``mean_actions`` to the third bundle; we prefer those so eval/assess
    use deterministic policy means (training still samples).

    Handles both multi-agent (dict keyed by agent ID) and single-agent (tensor).

    Replaces the 5-line if/else block that was copy-pasted into 7 scripts.

    Args:
        agent:    The skrl agent (MAPPO object or single-agent wrapper).
        obs:      Current observation from the environment.
        base_env: The unwrapped gymnasium environment.

    Returns:
        A dict ``{agent_id: action_tensor}`` for multi-agent envs, or a plain
        action tensor for single-agent envs.
    """
    outputs = agent.act(obs, timestep=0, timesteps=0)  # type: ignore[attr-defined]
    if hasattr(base_env, "possible_agents"):
        return {
            a: outputs[-1][a].get("mean_actions", outputs[0][a])
            for a in base_env.possible_agents  # type: ignore[attr-defined]
        }
    return outputs[-1].get("mean_actions", outputs[0])


# ---------------------------------------------------------------------------
# Eval config helper
# ---------------------------------------------------------------------------


def configure_eval_cfg(
    cfg: dict,
    log_root: str,
    run_name: str,
) -> None:
    """Disable checkpoint writing and fix experiment directories for eval.

    Replaces the repeated cfg override block found in 6 scripts:
        cfg["trainer"]["checkpoint_interval"] = 0
        cfg["agent"]["experiment"]["directory"] = log_root
        cfg["agent"]["experiment"]["experiment_name"] = run_name

    Args:
        cfg:      The skrl experiment config dict (modified in-place).
        log_root: Absolute path to the log root directory.
        run_name: Human-readable run name (typically the timestamp directory).
    """
    cfg["trainer"]["checkpoint_interval"] = 0
    cfg["agent"]["experiment"]["directory"] = log_root
    cfg["agent"]["experiment"]["experiment_name"] = run_name
