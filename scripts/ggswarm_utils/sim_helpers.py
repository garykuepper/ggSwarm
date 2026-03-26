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
    # Create adj_matrix tensor in lead agent's memory only.
    # Flattened to 1D for SKRL memory compat (reshape back during _update).
    adj_flat_size = num_agents * num_agents
    lead_mem = agent.memories[uids[0]]  # type: ignore[attr-defined]
    lead_mem.create_tensor(
        name="adj_matrix",
        size=adj_flat_size,
        dtype=torch.float32,
    )
    # SKRL initializes new float tensors with NaN (base.py line 182-184).
    # Zero-fill so unfilled entries default to "no edges" instead of NaN.
    lead_mem.tensors["adj_matrix"].zero_()

    # Patch record_transition to store adj_matrix at the correct memory index.
    # IMPORTANT: do NOT call add_samples() separately — that double-increments
    # memory_index (base.py line 239). Instead, write directly to the tensor
    # at the index original_record just used.
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
            # Write at the index original_record just used (memory_index was
            # already incremented, so go back 1).
            mem = agent.memories[uids[0]]  # type: ignore[attr-defined]
            prev_idx = (mem.memory_index - 1) % mem.memory_size
            mem.tensors["adj_matrix"][prev_idx].copy_(flat_adj)

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

    # --- 3. Custom centralized PPO _update ---
    # Replaces MAPPO's per-agent _update loop with a centralized training step
    # where ALL agents' losses flow through ONE GNN forward pass.
    #
    # Why: MAPPO processes agents independently during _update. The lead-only
    # approach (p2c-6b) used all agents' states but only agent 0's loss, creating
    # biased gradients that destabilized training. This custom loop computes
    # per-agent PPO losses from all agents and sums them before backward().
    if hasattr(agent, "_update"):
        import itertools  # noqa: PLC0415
        import torch.nn as nn  # noqa: PLC0415
        import torch.nn.functional as F  # noqa: PLC0415
        from torch.distributions import Normal  # noqa: PLC0415

        original_update = agent._update  # type: ignore[attr-defined]
        lead_uid = uids[0]
        lead_value = agent.values[lead_uid]  # type: ignore[attr-defined]

        def _compute_gae(
            rewards: torch.Tensor, dones: torch.Tensor,
            values: torch.Tensor, last_values: torch.Tensor,
            discount: float, lam: float,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            advantage = 0
            advantages = torch.zeros_like(rewards)
            not_dones = dones.logical_not()
            for i in reversed(range(rewards.shape[0])):
                nv = values[i + 1] if i < rewards.shape[0] - 1 else last_values
                advantage = rewards[i] - values[i] + discount * not_dones[i] * (nv + lam * advantage)
                advantages[i] = advantage
            returns = advantages + values
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            return returns, advantages

        def patched_update(timestep: int, timesteps: int) -> object:
            # --- Phase 1: GAE computation (per-agent, same as MAPPO) ---
            for uid in uids:
                value = agent.values[uid]  # type: ignore[attr-defined]
                memory = agent.memories[uid]  # type: ignore[attr-defined]
                with torch.no_grad(), torch.autocast(device_type=lead_policy.device.type, enabled=getattr(agent, "_mixed_precision", False)):
                    value.train(False)
                    last_vals, _, _ = value.act(
                        {"states": agent._shared_state_preprocessor[uid](agent._current_shared_next_states.float())},  # type: ignore[attr-defined]
                        role="value",
                    )
                    value.train(True)
                last_vals = agent._value_preprocessor[uid](last_vals, inverse=True)  # type: ignore[attr-defined]

                vals = memory.get_tensor_by_name("values")
                rets, advs = _compute_gae(
                    rewards=memory.get_tensor_by_name("rewards"),
                    dones=memory.get_tensor_by_name("terminated") | memory.get_tensor_by_name("truncated"),
                    values=vals, last_values=last_vals,
                    discount=agent._discount_factor[uid],  # type: ignore[attr-defined]
                    lam=agent._lambda[uid],  # type: ignore[attr-defined]
                )
                memory.set_tensor_by_name("values", agent._value_preprocessor[uid](vals, train=True))  # type: ignore[attr-defined]
                memory.set_tensor_by_name("returns", agent._value_preprocessor[uid](rets, train=True))  # type: ignore[attr-defined]
                memory.set_tensor_by_name("advantages", advs)

            # --- Phase 2: Custom centralized mini-batch training ---
            lead_mem = agent.memories[lead_uid]  # type: ignore[attr-defined]
            total_samples = lead_mem.memory_size * lead_mem.num_envs
            mini_batches = agent._mini_batches[lead_uid]  # type: ignore[attr-defined]
            batch_size = total_samples // mini_batches
            learning_epochs = agent._learning_epochs[lead_uid]  # type: ignore[attr-defined]
            ratio_clip = agent._ratio_clip[lead_uid]  # type: ignore[attr-defined]
            entropy_scale = agent._entropy_loss_scale[lead_uid]  # type: ignore[attr-defined]
            value_loss_scale = agent._value_loss_scale[lead_uid]  # type: ignore[attr-defined]
            grad_clip = agent._grad_norm_clip[lead_uid]  # type: ignore[attr-defined]
            optimizer = agent.optimizers[lead_uid]  # type: ignore[attr-defined]
            scaler = agent.scaler  # type: ignore[attr-defined]
            clip_predicted = agent._clip_predicted_values  # type: ignore[attr-defined]
            value_clip = agent._value_clip[lead_uid]  # type: ignore[attr-defined]

            cumulative_policy_loss = 0.0
            cumulative_entropy_loss = 0.0
            cumulative_value_loss = 0.0

            for epoch in range(learning_epochs):
                kl_divergences: list[torch.Tensor] = []

                for mb in range(mini_batches):
                    start = mb * batch_size
                    end = start + batch_size

                    with torch.autocast(device_type=lead_policy.device.type, enabled=getattr(agent, "_mixed_precision", False)):
                        # 1. Gather all agents' data from aligned memory slices.
                        all_states = []
                        all_actions = []
                        all_old_log_probs = []
                        all_advantages = []
                        for i, uid in enumerate(uids):
                            mem = agent.memories[uid]  # type: ignore[attr-defined]
                            st = mem.tensors_view["states"][start:end]
                            st = agent._state_preprocessor[uid](st, train=(epoch == 0 and i == 0))  # type: ignore[attr-defined]
                            all_states.append(st)
                            all_actions.append(mem.tensors_view["actions"][start:end])
                            all_old_log_probs.append(mem.tensors_view["log_prob"][start:end])
                            all_advantages.append(mem.tensors_view["advantages"][start:end])

                        # Value data from lead agent.
                        shared_st = lead_mem.tensors_view["shared_states"][start:end]
                        shared_st = agent._shared_state_preprocessor[lead_uid](shared_st, train=(epoch == 0))  # type: ignore[attr-defined]
                        sampled_values = lead_mem.tensors_view["values"][start:end]
                        sampled_returns = lead_mem.tensors_view["returns"][start:end]

                        # 2. Centralized GNN forward.
                        stacked = torch.stack(all_states, dim=1)  # [batch, num_agents, obs_dim]
                        batched_obs = stacked.reshape(-1, stacked.shape[-1])
                        adj_flat = lead_mem.tensors_view["adj_matrix"][start:end]
                        adj_3d = adj_flat.reshape(-1, num_agents, num_agents)

                        mean_all, log_std, _ = lead_policy.compute(
                            {"states": batched_obs, "extras": {"adj_matrix": adj_3d}},
                            role="policy",
                        )
                        mean_3d = mean_all.reshape(-1, num_agents, mean_all.shape[-1])

                        if lead_policy._g_clip_log_std:
                            log_std = torch.clamp(log_std, lead_policy._g_log_std_min, lead_policy._g_log_std_max)
                        std = log_std.exp()

                        # 3. Per-agent PPO loss (ALL agents contribute).
                        total_policy_loss = torch.tensor(0.0, device=lead_policy.device)
                        total_entropy = torch.tensor(0.0, device=lead_policy.device)

                        for i, uid in enumerate(uids):
                            agent_mean = mean_3d[:, i, :]
                            dist = Normal(agent_mean, std)
                            new_lp = dist.log_prob(all_actions[i])
                            if lead_policy._g_reduction is not None:
                                new_lp = lead_policy._g_reduction(new_lp, dim=-1)
                            if new_lp.dim() != all_actions[i].dim():
                                new_lp = new_lp.unsqueeze(-1)

                            ratio = torch.exp(new_lp - all_old_log_probs[i])
                            surr = all_advantages[i] * ratio
                            surr_clip = all_advantages[i] * torch.clip(ratio, 1.0 - ratio_clip, 1.0 + ratio_clip)
                            total_policy_loss += -torch.min(surr, surr_clip).mean()

                            if entropy_scale:
                                total_entropy += -entropy_scale * dist.entropy().mean()

                            # KL for first agent (representative, used for LR scheduling)
                            if i == 0:
                                with torch.no_grad():
                                    kl_ratio = new_lp - all_old_log_probs[i]
                                    kl_divergences.append(((torch.exp(kl_ratio) - 1) - kl_ratio).mean())

                        total_policy_loss = total_policy_loss / num_agents

                        # Set distribution on lead policy for logging (stddev tracking).
                        lead_policy._g_distribution = Normal(mean_3d[:, 0, :], std)

                        # 4. Value loss (lead agent's value network + shared states).
                        predicted_values, _, _ = lead_value.act({"states": shared_st}, role="value")
                        if clip_predicted:
                            predicted_values = sampled_values + torch.clip(
                                predicted_values - sampled_values, min=-value_clip, max=value_clip
                            )
                        value_loss = value_loss_scale * F.mse_loss(sampled_returns, predicted_values)

                    # 5. Single backward + optimizer step.
                    optimizer.zero_grad()
                    total_loss = total_policy_loss + total_entropy + value_loss
                    scaler.scale(total_loss).backward()

                    if grad_clip > 0:
                        scaler.unscale_(optimizer)
                        if lead_policy is lead_value:
                            nn.utils.clip_grad_norm_(lead_policy.parameters(), grad_clip)
                        else:
                            nn.utils.clip_grad_norm_(
                                itertools.chain(lead_policy.parameters(), lead_value.parameters()), grad_clip
                            )

                    scaler.step(optimizer)
                    scaler.update()

                    cumulative_policy_loss += total_policy_loss.item()
                    cumulative_value_loss += value_loss.item()
                    if entropy_scale:
                        cumulative_entropy_loss += total_entropy.item()

                # Learning rate scheduling (KL-adaptive for lead agent).
                sched = agent.schedulers.get(lead_uid)  # type: ignore[attr-defined]
                if sched is not None:
                    from skrl.utils.scheduler import KLAdaptiveLR  # noqa: PLC0415
                    if isinstance(sched, KLAdaptiveLR):
                        kl = torch.tensor(kl_divergences, device=lead_policy.device).mean()
                        sched.step(kl.item())
                    else:
                        sched.step()

            # Record data for TensorBoard (same keys MAPPO uses).
            denom = learning_epochs * mini_batches
            agent.track_data(f"Loss / Policy loss ({lead_uid})", cumulative_policy_loss / denom)  # type: ignore[attr-defined]
            agent.track_data(f"Loss / Value loss ({lead_uid})", cumulative_value_loss / denom)  # type: ignore[attr-defined]
            if entropy_scale:
                agent.track_data(f"Loss / Entropy loss ({lead_uid})", cumulative_entropy_loss / denom)  # type: ignore[attr-defined]
            agent.track_data(  # type: ignore[attr-defined]
                f"Policy / Standard deviation ({lead_uid})",
                lead_policy.distribution(role="policy").stddev.mean().item(),
            )
            if sched is not None:
                agent.track_data(f"Learning / Learning rate ({lead_uid})", sched.get_last_lr()[0])  # type: ignore[attr-defined]

            # NaN guard: fail fast if training diverged.
            for p in lead_policy.parameters():
                if torch.isnan(p).any():
                    raise RuntimeError(
                        "NaN detected in GNN policy parameters after _update. "
                        "Training diverged — check reward scales or graph alignment."
                    )

            # Sync policy weights: lead → all others.
            src = lead_policy.state_dict()
            for uid in uids[1:]:
                agent.policies[uid].load_state_dict(src)  # type: ignore[attr-defined]

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
