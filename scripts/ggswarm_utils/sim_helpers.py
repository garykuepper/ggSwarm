"""Shared Isaac Lab simulation helpers for all eval/play/bench scripts.

Eliminates cross-script duplication of:
  - GNN policy configuration (configure_gnn_policy)
  - GNN batched act patch (patch_mappo_gnn_batched_act)
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
        task="Template-GGSwarm-Marl-Perturbation-v0",
        gnn_default=True,
        collector_cls="ggswarm_utils.phases.phase2.Phase2Collector",
        description="Phase 2C perturbation (random pushes on formation hover)",
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
# GNN adj_matrix pipeline patch
# ---------------------------------------------------------------------------


def patch_mappo_gnn_batched_act(agent: object, env: object) -> None:
    """Patch MAPPO for centralized GNN forward: batch all agents, call GNN once.

    SKRL's MAPPO calls ``policy.act()`` per agent with obs ``[num_envs, obs_dim]``,
    but the GNN needs ALL agents simultaneously as ``[num_envs * num_agents, obs_dim]``
    to build the graph from ``adj_matrix``.

    This patch:

    1. Syncs policy weights across all agent instances (parameter sharing).
    2. Overrides ``agent.act()`` to batch all agents' preprocessed obs, call ONE
       policy's ``compute()`` with the full graph, then split + sample per-agent.
    3. Stores ``adj_matrix`` in SKRL memory during ``record_transition`` so it's
       available during training (GNN gradient fix).
    4. Overrides ``agent._update()`` to inject stored adj_matrix into each policy's
       ``act()`` during gradient computation, then sync weights afterward.

    Preprocessor safety: ``RunningStandardScaler`` lives on
    ``agent._state_preprocessor[uid]``, NOT inside ``policy.state_dict()``.
    Weight sync only touches nn.Module parameters — preprocessor stats are untouched.

    Args:
        agent: The skrl MAPPO multi-agent object.
        env:   The skrl-wrapped environment (``IsaacLabMultiAgentWrapper``).
    """
    import torch  # noqa: PLC0415
    from torch.distributions import Normal  # noqa: PLC0415

    uids = list(agent.possible_agents)  # type: ignore[attr-defined]
    num_agents = len(uids)
    # Use first agent's policy as the canonical GNN for centralized forward.
    lead_policy = agent.policies[uids[0]]  # type: ignore[attr-defined]

    # --- 1. Initial weight sync: lead → all others ---
    src_state = lead_policy.state_dict()
    for uid in uids[1:]:
        agent.policies[uid].load_state_dict(src_state)  # type: ignore[attr-defined]

    # --- 1b. GNN gradient fix: store adj_matrix in SKRL memory ---
    # Create adj_matrix tensor in each agent's memory so it's available during _update.
    # Flattened to 1D for SKRL memory compat (reshape back during _update).
    adj_flat_size = num_agents * num_agents
    for uid in uids:
        memory = agent.memories[uid]  # type: ignore[attr-defined]
        memory.create_tensor(
            name="adj_matrix",
            size=adj_flat_size,
            dtype=torch.float32,
        )
    # NOTE: Do NOT add "adj_matrix" to _tensors_names — MAPPO's _update unpacks
    # sample_all() by position (expects exactly 7 tensors). Instead, we fetch
    # adj_matrix directly from memory via get_tensor_by_name() in the _update patch.

    # Patch record_transition to store adj_matrix from infos into memory.
    original_record = agent.record_transition  # type: ignore[attr-defined]

    def patched_record(
        states: object, actions: object, rewards: object,
        next_states: object, terminated: object, truncated: object,
        infos: dict, timestep: int, timesteps: int,
    ) -> None:
        original_record(states, actions, rewards, next_states,
                        terminated, truncated, infos, timestep, timesteps)
        adj = infos.get("adj_matrix") if isinstance(infos, dict) else None
        if adj is not None:
            # shape: [num_envs, num_agents, num_agents] → [num_envs, num_agents*num_agents]
            flat_adj = adj.reshape(adj.shape[0], -1)
            for uid in uids:
                agent.memories[uid].add_samples(adj_matrix=flat_adj)  # type: ignore[attr-defined]

    agent.record_transition = patched_record  # type: ignore[attr-defined]

    # --- 2. Patched act(): centralized GNN forward ---
    original_act = agent.act  # type: ignore[attr-defined]

    def patched_act(
        states: dict[str, torch.Tensor],
        timestep: int,
        timesteps: int,
    ) -> tuple[dict, dict, dict]:
        info = getattr(env, "_info", {})
        adj = info.get("adj_matrix") if isinstance(info, dict) else None
        if adj is None:
            return original_act(states, timestep=timestep, timesteps=timesteps)

        # Preprocess each agent's obs with its own RunningStandardScaler.
        # shape per uid: [num_envs, obs_dim]
        preprocessed = [
            agent._state_preprocessor[uid](states[uid])  # type: ignore[attr-defined]
            for uid in uids
        ]
        # shape: [num_envs, num_agents, obs_dim]
        stacked = torch.stack(preprocessed, dim=1)
        # shape: [num_envs * num_agents, obs_dim]  (row-major matches edge_index)
        batched_obs = stacked.reshape(-1, stacked.shape[-1])

        # Centralized GNN forward (one call for all agents).
        with torch.autocast(
            device_type=lead_policy.device.type,  # type: ignore[union-attr]
            enabled=getattr(agent, "_mixed_precision", False),
        ):
            mean_all, log_std, _ = lead_policy.compute(
                {"states": batched_obs, "extras": {"adj_matrix": adj}}, role="policy"
            )

        # shape: [num_envs, num_agents, action_dim]
        mean_3d = mean_all.reshape(-1, num_agents, mean_all.shape[-1])

        # Clamp log_std (shared across all agents — single nn.Parameter).
        if lead_policy._g_clip_log_std:
            log_std = torch.clamp(
                log_std, lead_policy._g_log_std_min, lead_policy._g_log_std_max
            )

        actions_dict: dict[str, torch.Tensor] = {}
        log_prob_dict: dict[str, torch.Tensor] = {}
        outputs_dict: dict[str, dict[str, torch.Tensor]] = {}

        for i, uid in enumerate(uids):
            # shape: [num_envs, action_dim]
            agent_mean = mean_3d[:, i, :]
            policy = agent.policies[uid]  # type: ignore[attr-defined]

            dist = Normal(agent_mean, log_std.exp())
            # Set on each policy so get_entropy() works during _update.
            policy._g_log_std = log_std
            policy._g_num_samples = agent_mean.shape[0]
            policy._g_distribution = dist

            sampled = dist.rsample()
            if policy._g_clip_actions:
                sampled = torch.clamp(
                    sampled,
                    min=policy._g_clip_actions_min,
                    max=policy._g_clip_actions_max,
                )

            log_prob = dist.log_prob(sampled)
            if policy._g_reduction is not None:
                log_prob = policy._g_reduction(log_prob, dim=-1)
            if log_prob.dim() != sampled.dim():
                log_prob = log_prob.unsqueeze(-1)

            actions_dict[uid] = sampled
            log_prob_dict[uid] = log_prob
            outputs_dict[uid] = {"mean_actions": agent_mean}

        # Store for record_transition (MAPPO reads agent._current_log_prob).
        agent._current_log_prob = log_prob_dict  # type: ignore[attr-defined]

        return actions_dict, log_prob_dict, outputs_dict

    agent.act = patched_act  # type: ignore[attr-defined]

    # --- 3. Patched _update(): centralized GNN forward during training ---
    # Strategy: wrap lead policy's act() to do centralized forward with graph.
    # Only the lead agent (uid[0]) processes the PPO update — other agents are
    # skipped. Weights are synced afterward. This avoids the "backward through
    # graph twice" problem (shared computation graph across agents).
    #
    # The lead agent's mini-batch uses ALL agents' states (batched) with
    # adj_matrix for the GNN forward, so attention weights get real graph
    # gradients. PPO loss, value loss, and entropy use the lead agent's own
    # memory (agent 0's experiences), which is representative under parameter
    # sharing.
    if hasattr(agent, "_update"):
        original_update = agent._update  # type: ignore[attr-defined]

        def patched_update(timestep: int, timesteps: int) -> object:
            # Wrap lead policy's act() for centralized GNN forward.
            lead_uid = uids[0]
            lead_saved_act = agent.policies[lead_uid].act  # type: ignore[attr-defined]

            def _centralized_act(inputs: dict, role: str = "") -> object:
                taken_actions = inputs.get("taken_actions")
                if taken_actions is None:
                    return lead_saved_act(inputs, role=role)

                sampled_states = inputs["states"]
                batch_size = sampled_states.shape[0]

                # Collect all agents' states for centralized GNN forward.
                all_states = [sampled_states]  # lead agent's preprocessed states
                for other_idx in range(1, num_agents):
                    other_uid = uids[other_idx]
                    other_mem = agent.memories[other_uid]  # type: ignore[attr-defined]
                    other_all = other_mem.get_tensor_by_name("states", keepdim=False)
                    other_states = other_all[-batch_size:]
                    other_states = agent._state_preprocessor[other_uid](  # type: ignore[attr-defined]
                        other_states, train=False
                    )
                    all_states.append(other_states)

                # Get adj_matrix from memory
                try:
                    flat_adj = agent.memories[lead_uid].get_tensor_by_name(  # type: ignore[attr-defined]
                        "adj_matrix", keepdim=False
                    )
                    adj_batch = flat_adj[-batch_size:]
                    adj_3d = adj_batch.reshape(-1, num_agents, num_agents)
                except (KeyError, RuntimeError):
                    adj_3d = None

                # Stack: [batch, num_agents, obs_dim] → [batch * num_agents, obs_dim]
                stacked = torch.stack(all_states, dim=1)
                batched_obs = stacked.reshape(-1, stacked.shape[-1])

                # Centralized GNN forward (WITH gradients — single backward)
                extras = {"adj_matrix": adj_3d} if adj_3d is not None else {}
                mean_all, log_std, _ = lead_policy.compute(
                    {"states": batched_obs, "extras": extras}, role="policy"
                )

                # Extract lead agent's mean (index 0)
                # shape: [batch, num_agents, action_dim]
                mean_3d = mean_all.reshape(-1, num_agents, mean_all.shape[-1])
                agent_mean = mean_3d[:, 0, :]

                # GaussianMixin.act() logic for training
                if lead_policy._g_clip_log_std:
                    log_std = torch.clamp(log_std, lead_policy._g_log_std_min, lead_policy._g_log_std_max)

                from torch.distributions import Normal  # noqa: PLC0415
                dist = Normal(agent_mean, log_std.exp())
                lead_policy._g_log_std = log_std
                lead_policy._g_num_samples = agent_mean.shape[0]
                lead_policy._g_distribution = dist

                log_prob = dist.log_prob(taken_actions)
                if lead_policy._g_reduction is not None:
                    log_prob = lead_policy._g_reduction(log_prob, dim=-1)
                if log_prob.dim() != taken_actions.dim():
                    log_prob = log_prob.unsqueeze(-1)

                return taken_actions, log_prob, {"mean_actions": agent_mean}

            # Only run _update for the lead agent — skip others.
            # Temporarily replace possible_agents to contain only lead uid.
            original_agents = agent.possible_agents  # type: ignore[attr-defined]
            agent.possible_agents = [lead_uid]  # type: ignore[attr-defined]
            agent.policies[lead_uid].act = _centralized_act  # type: ignore[attr-defined]

            try:
                result = original_update(timestep, timesteps)
            finally:
                agent.possible_agents = original_agents  # type: ignore[attr-defined]
                agent.policies[lead_uid].act = lead_saved_act  # type: ignore[attr-defined]

            # Sync policy weights: lead → all others.
            src = lead_policy.state_dict()
            for uid in uids[1:]:
                agent.policies[uid].load_state_dict(src)  # type: ignore[attr-defined]
            return result

        agent._update = patched_update  # type: ignore[attr-defined]


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


def extract_actions(
    agent: object,
    obs: object,
    base_env: object,
    eval_noise_std: float = 0.0,
) -> object:
    """Run agent.act() and return **mean** Gaussian actions when skrl exposes them.

    ``GaussianMixin.act`` returns sampled actions in the first tuple slot but
    attaches ``mean_actions`` to the third bundle; we prefer those so eval/assess
    use deterministic policy means (training still samples).

    When ``eval_noise_std > 0``, small Gaussian noise is added to the mean actions
    to replicate the dithering effect that helps the PD controller escape saturation
    during stochastic training (see PD10 train-eval gap analysis in changelog).

    Handles both multi-agent (dict keyed by agent ID) and single-agent (tensor).

    Args:
        agent:          The skrl agent (MAPPO object or single-agent wrapper).
        obs:            Current observation from the environment.
        base_env:       The unwrapped gymnasium environment.
        eval_noise_std: Standard deviation of Gaussian noise added to mean actions
                        during eval. 0.0 = fully deterministic (default).

    Returns:
        A dict ``{agent_id: action_tensor}`` for multi-agent envs, or a plain
        action tensor for single-agent envs.
    """
    import torch  # noqa: PLC0415

    outputs = agent.act(obs, timestep=0, timesteps=0)  # type: ignore[attr-defined]
    if hasattr(base_env, "possible_agents"):
        actions = {
            a: outputs[-1][a].get("mean_actions", outputs[0][a])
            for a in base_env.possible_agents  # type: ignore[attr-defined]
        }
        if eval_noise_std > 0:
            for a in actions:
                actions[a] = (actions[a] + eval_noise_std * torch.randn_like(actions[a])).clamp(-1.0, 1.0)
        return actions
    act = outputs[-1].get("mean_actions", outputs[0])
    if eval_noise_std > 0:
        act = (act + eval_noise_std * torch.randn_like(act)).clamp(-1.0, 1.0)
    return act


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
