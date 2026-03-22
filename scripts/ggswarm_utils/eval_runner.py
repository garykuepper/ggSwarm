"""Shared Isaac Lab evaluation boilerplate for all phase evaluations.

Provides the PhaseCollector protocol and run_eval() function that handle:
  - @hydra_task_config decoration
  - skrl Runner construction and agent resolution
  - GNN policy injection (optional)
  - Agent count override (optional)
  - Checkpoint loading
  - The simulation step loop
  - Returning the final metrics dict

Usage (the caller must have already booted AppLauncher):

    from ggswarm_utils.eval_runner import run_eval
    from ggswarm_utils.phases.phase2 import Phase2Collector

    metrics = run_eval(
        task="Template-GGSwarm-Marl-HoverStability-v0",
        simulation_app=simulation_app,
        collector=Phase2Collector(),
        checkpoint="logs/.../checkpoints/best_agent.pt",
        num_episodes=5,
        gnn=True,
        seed=1,
        device=None,
        num_envs=None,
        num_agents=None,
    )

IMPORTANT: This module imports Isaac Lab lazily (inside run_eval) so it can be
imported at the top of entry-point scripts before AppLauncher is created.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# PhaseCollector protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class PhaseCollector(Protocol):
    """Interface that every phase-specific metric collector must implement.

    The eval loop in run_eval() calls:
      - on_step()  once per simulation step (inside torch.inference_mode())
      - on_episode_end()  when an episode boundary is detected
      - summarize()  once at the end to get the final metrics dict
    """

    def on_step(
        self,
        *,
        base_env: object,
        obs: object,
        step: int,
        episode_step: int,
        episode_num: int,
    ) -> None:
        """Accumulate per-step metrics from the live environment.

        Args:
            base_env:     The unwrapped gymnasium environment.
            obs:          Current observation (after env.step).
            step:         Global step index (0-based).
            episode_step: Step index within the current episode.
            episode_num:  Episode index (0-based).
        """
        ...

    def on_episode_end(self, episode_num: int) -> None:
        """Called when an episode ends (step count reached max_episode_length).

        Args:
            episode_num: Index of the episode that just ended (0-based).
        """
        ...

    def summarize(self) -> dict[str, float]:
        """Return a flat dict of metric_name -> mean_value over all steps."""
        ...


# ---------------------------------------------------------------------------
# run_eval
# ---------------------------------------------------------------------------


def run_eval(
    task: str,
    simulation_app: object,
    collector: PhaseCollector,
    checkpoint: str | Path | None,
    num_episodes: int,
    gnn: bool,
    seed: int,
    device: str | None,
    num_envs: int | None,
    num_agents: int | None,
    run_dir: str | Path | None = None,
    extra_cfg_fn: Callable[[dict], None] | None = None,
) -> dict[str, float]:
    """Run a complete evaluation loop and return summarized metrics.

    Handles all Isaac Lab / skrl boilerplate so that each eval script or
    post_train_assess.py only needs to supply a PhaseCollector.

    Args:
        task:           Gym task ID.
        simulation_app: Running Isaac Lab simulation app (from AppLauncher).
        collector:      Phase-specific collector implementing PhaseCollector.
        checkpoint:     Path to .pt checkpoint.  If None, auto-discovery via
                        the task's log directory is attempted.
        num_episodes:   Number of complete episodes to run.
        gnn:            Whether to inject GGSwarmGNNPolicy.
        seed:           RNG seed for the environment.
        device:         Override sim device (e.g. "cuda:0").  None = use cfg.
        num_envs:       Override number of parallel environments.
        num_agents:     Override number of agents per env.
        run_dir:        Optional path to the run directory for log naming.
        extra_cfg_fn:   Optional callable(cfg) applied before Runner is built.

    Returns:
        Dict of metric_name -> mean value produced by collector.summarize().

    Raises:
        RuntimeError: If Isaac Lab is not available or the simulation exits early.
        FileNotFoundError: If checkpoint is None and no checkpoint can be found.
    """
    # -----------------------------------------------------------------------
    # Deferred Isaac Lab / skrl imports (require AppLauncher to be running)
    # -----------------------------------------------------------------------
    import gymnasium as gym  # noqa: PLC0415
    import torch  # noqa: PLC0415
    from packaging import version  # noqa: PLC0415
    import skrl  # noqa: PLC0415
    from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg, DirectRLEnvCfg  # noqa: PLC0415
    from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: PLC0415
    from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: PLC0415
    from isaaclab_tasks.utils import get_checkpoint_path  # noqa: PLC0415
    from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: PLC0415
    from skrl.utils.runner.torch import Runner  # noqa: PLC0415

    if version.parse(skrl.__version__) < version.parse("1.4.3"):
        raise RuntimeError(
            f"Unsupported skrl version: {skrl.__version__}. Install skrl>=1.4.3"
        )

    from ggswarm_utils.checkpoint import load_policy_from_checkpoint, resolve_agent  # noqa: PLC0415
    from ggswarm_utils.sim_helpers import configure_eval_cfg, configure_gnn_policy  # noqa: PLC0415
    from ggswarm_utils.sim_helpers import extract_actions, override_agent_count  # noqa: PLC0415

    # Mutable holder — @hydra_task_config swallows return values
    _result_holder: list[dict[str, float] | None] = [None]

    @hydra_task_config(task, "skrl_mappo_cfg_entry_point")
    def _run(
        env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg,
        cfg: dict,
    ) -> None:
        # ----------------------------------------------------------------
        # Env / cfg overrides
        # ----------------------------------------------------------------
        if device is not None:
            env_cfg.sim.device = device
        if num_envs is not None:
            env_cfg.scene.num_envs = num_envs
        if num_agents is not None:
            obs_dim = 14 if getattr(env_cfg, "raft_enabled", False) else 12
            override_agent_count(env_cfg, num_agents, obs_dim=obs_dim)

        cfg["seed"] = seed
        env_cfg.seed = seed

        if gnn:
            configure_gnn_policy(cfg, Runner)

        # Derive log root and run name from run_dir or cfg defaults
        log_root = os.path.abspath(
            os.path.join("logs", "skrl", cfg["agent"]["experiment"]["directory"])
        )
        _run_name = Path(run_dir).name if run_dir else cfg["agent"]["experiment"].get(
            "experiment_name", "eval"
        )
        configure_eval_cfg(cfg, log_root, _run_name)

        if extra_cfg_fn is not None:
            extra_cfg_fn(cfg)

        # ----------------------------------------------------------------
        # Resolve checkpoint path
        # ----------------------------------------------------------------
        if checkpoint is not None:
            resume_path = os.path.abspath(str(checkpoint))
        else:
            resume_path = get_checkpoint_path(
                log_root,
                run_dir=".*_mappo_torch",
                other_dirs=["checkpoints"],
            )

        # ----------------------------------------------------------------
        # Build environment
        # ----------------------------------------------------------------
        env = gym.make(task, cfg=env_cfg)
        base_env = env.unwrapped
        env = SkrlVecEnvWrapper(env, ml_framework="torch")

        # ----------------------------------------------------------------
        # Build runner + load checkpoint
        # ----------------------------------------------------------------
        runner = Runner(env, cfg)
        agent = resolve_agent(runner)
        load_policy_from_checkpoint(agent, resume_path)
        agent.set_running_mode("eval")

        # ----------------------------------------------------------------
        # Sim loop
        # ----------------------------------------------------------------
        max_ep_len = getattr(base_env, "max_episode_length", None)
        if max_ep_len is None:
            raise RuntimeError("Could not determine max_episode_length from env")

        total_steps = num_episodes * int(max_ep_len)
        obs, _ = env.reset()
        step = 0
        episode_step = 0
        episode_num = 0

        while simulation_app.is_running() and step < total_steps:  # type: ignore[union-attr]
            with torch.inference_mode():
                actions = extract_actions(agent, obs, base_env)
                obs, _, _, _, _ = env.step(actions)

                collector.on_step(
                    base_env=base_env,
                    obs=obs,
                    step=step,
                    episode_step=episode_step,
                    episode_num=episode_num,
                )

            step += 1
            episode_step += 1

            if episode_step >= max_ep_len:
                collector.on_episode_end(episode_num)
                episode_num += 1
                episode_step = 0

        env.close()
        _result_holder[0] = collector.summarize()

    _run()

    if _result_holder[0] is None:
        raise RuntimeError(
            "run_eval produced no metrics — simulation may have exited early."
        )
    return _result_holder[0]
