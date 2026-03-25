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
        seed=42,
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


def _wrap_env_for_eval_recording(
    env: object,
    *,
    resume_path: str,
    run_name: str,
    log_dir: str,
    video_length: int,
    video_codec: str | None,
    video_bitrate: str | None,
    video_preset: str | None,
    video_ffmpeg_params: str | None,
) -> object:
    """Wrap a gym env with ``EncodingRecordVideo`` (same pattern as ``play.py``).

    Writes MP4s under ``<log_dir>/videos/eval/``.

    Args:
        env:            Gymnasium env from ``gym.make(..., render_mode=\"rgb_array\")``.
        resume_path:    Resolved checkpoint path (for filename prefix).
        run_name:       Training run / experiment name segment.
        log_dir:        Run directory (parent of ``checkpoints/``).
        video_length:   Number of steps to record from episode start.
        video_codec:    Preferred ffmpeg codec (default ``hevc_nvenc`` with CPU fallback).
        video_bitrate:  Optional ffmpeg bitrate (e.g. ``8M``).
        video_preset:   Optional ffmpeg preset.
        video_ffmpeg_params: Optional extra ffmpeg args as one string (shell-split).

    Returns:
        Env wrapped with ``EncodingRecordVideo``.
    """
    import shlex

    from ggswarm_utils.encoding_record_video import EncodingRecordVideo

    checkpoint_stem = os.path.splitext(os.path.basename(resume_path))[0]
    raw_video_prefix = f"{run_name}__{checkpoint_stem}"
    safe_video_prefix = "".join(
        ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in raw_video_prefix
    )
    preferred_codec = video_codec or "hevc_nvenc"
    ffmpeg_params = (
        shlex.split(video_ffmpeg_params) if video_ffmpeg_params else None
    )
    if ffmpeg_params is None and preferred_codec == "hevc_nvenc":
        ffmpeg_params = [
            "-rc",
            "vbr",
            "-cq",
            "19",
            "-b:v",
            "0",
            "-spatial_aq",
            "1",
            "-aq-strength",
            "8",
        ]
    video_folder = os.path.join(log_dir, "videos", "eval")
    video_kwargs = {
        "video_folder": video_folder,
        "step_trigger": lambda step: step == 0,
        "video_length": video_length,
        "name_prefix": safe_video_prefix,
        "disable_logger": True,
    }
    print("[EVAL] Recording video (headless rgb_array). First steps may take 30–60s.")
    print(
        "[EVAL] Video encoding config: "
        f"preferred_codec={preferred_codec} "
        f"bitrate={video_bitrate} preset={video_preset} "
        f"ffmpeg_params={ffmpeg_params}"
    )
    return EncodingRecordVideo(
        env,
        **video_kwargs,
        preferred_codec=preferred_codec,
        bitrate=video_bitrate,
        preset=video_preset,
        ffmpeg_params=ffmpeg_params,
    )


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
    *,
    video: bool = False,
    video_length: int = 200,
    video_codec: str | None = None,
    video_bitrate: str | None = None,
    video_preset: str | None = None,
    video_ffmpeg_params: str | None = None,
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
        video:          If True, record rgb_array clip under run_dir/videos/eval.
        video_length:   Steps to record from global step 0 (default 200).
        video_codec:    Preferred ffmpeg codec (default hevc_nvenc with fallback).
        video_bitrate:  Optional ffmpeg bitrate (e.g. ``8M``).
        video_preset:   Optional ffmpeg preset.
        video_ffmpeg_params: Extra ffmpeg args as one string (shell-split).

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
    from isaaclab.envs import DirectMARLEnvCfg, DirectRLEnvCfg  # noqa: PLC0415
    from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: PLC0415
    from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: PLC0415
    from isaaclab_tasks.utils import get_checkpoint_path  # noqa: PLC0415
    from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: PLC0415
    from skrl.utils.runner.torch import Runner  # noqa: PLC0415

    if version.parse(skrl.__version__) < version.parse("1.4.3"):
        raise RuntimeError(
            f"Unsupported skrl version: {skrl.__version__}. Install skrl>=1.4.3"
        )

    from ggswarm_utils.checkpoint import (  # noqa: PLC0415
        load_policy_from_checkpoint,
        resolve_agent,
        validate_eval_checkpoint_path,
    )
    from ggswarm_utils.sim_helpers import configure_eval_cfg, configure_gnn_policy  # noqa: PLC0415
    from ggswarm_utils.sim_helpers import extract_actions, override_agent_count  # noqa: PLC0415
    from ggswarm_utils.sim_helpers import patch_mappo_gnn_batched_act  # noqa: PLC0415

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

        validate_eval_checkpoint_path(resume_path)

        log_dir = os.path.dirname(os.path.dirname(resume_path))
        env_cfg.log_dir = log_dir

        if video:
            if num_envs is None and env_cfg.scene.num_envs > 1:
                env_cfg.scene.num_envs = 1
                print(
                    "[EVAL] Video mode: forcing num_envs=1 for stable recording "
                    "(override with --num_envs)."
                )

        # ----------------------------------------------------------------
        # Build environment
        # ----------------------------------------------------------------
        env = gym.make(
            task,
            cfg=env_cfg,
            render_mode="rgb_array" if video else None,
        )
        if video:
            env = _wrap_env_for_eval_recording(
                env,
                resume_path=resume_path,
                run_name=_run_name,
                log_dir=log_dir,
                video_length=video_length,
                video_codec=video_codec,
                video_bitrate=video_bitrate,
                video_preset=video_preset,
                video_ffmpeg_params=video_ffmpeg_params,
            )
        base_env = env.unwrapped
        env = SkrlVecEnvWrapper(env, ml_framework="torch")

        # ----------------------------------------------------------------
        # Build runner + load checkpoint
        # ----------------------------------------------------------------
        runner = Runner(env, cfg)
        agent = resolve_agent(runner)
        # Use SKRL's built-in load() which restores policy weights, value
        # network, state preprocessors (RunningStandardScaler), and optimizer.
        # Our old load_policy_from_checkpoint() only loaded policy weights,
        # leaving the preprocessor at fresh mean=0/var=1 — this was the root
        # cause of the train-eval gap across PD1-PD20.
        agent.load(str(resume_path))
        agent.set_running_mode("eval")

        # Centralized GNN forward: batch all agents, call GNN once with adj_matrix.
        if gnn:
            patch_mappo_gnn_batched_act(agent, env)

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
