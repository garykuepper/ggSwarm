"""Pytest test suite for the ggswarm_utils package and core env modules.

Covers: eval_stats, scorecard, checkpoint, gcs_sync, cli, sim_helpers,
        attitude_controller, and contract_logic (stable hover rewards).
No Isaac Lab or Isaac Sim required — all pure Python / torch.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pytest
import torch

# Ensure ggswarm_utils is importable from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

# ---------------------------------------------------------------------------
# eval_stats
# ---------------------------------------------------------------------------

from ggswarm_utils.eval_stats import (  # noqa: E402
    EvalStats,
    compute_altitude_metrics,
    compute_orientation_metrics,
    pairwise_mean_abs_spacing_error,
)


def _make_eval_kwargs(**overrides) -> dict:
    """Return a complete keyword-arg dict for EvalStats.update() with sane defaults.

    Per-step metrics only; use EvalStats.record_episode_survival() for survival_steps.
    """
    defaults = {
        "formation_error": torch.tensor(1.0),
        "separation_event_rate": torch.tensor(0.1),
        "mean_speed": torch.tensor(0.5),
        "mean_altitude_error": torch.tensor(0.2),
        "ground_hit_rate": torch.tensor(0.0),
        "airborne_ratio": torch.tensor(0.95),
        "mean_roll_deg": torch.tensor(5.0),
        "mean_pitch_deg": torch.tensor(3.0),
        "orientation_violation_rate": torch.tensor(0.02),
        "altitude_std": torch.tensor(0.05),
    }
    defaults.update(overrides)
    return defaults


class TestEvalStats:
    def test_update_increments_counts(self) -> None:
        s = EvalStats()
        s.update(**_make_eval_kwargs())
        assert s.formation_error_count == 1
        assert s.airborne_ratio_count == 1

    def test_update_accumulates_sums(self) -> None:
        s = EvalStats()
        s.update(**_make_eval_kwargs(formation_error=torch.tensor(2.0)))
        s.update(**_make_eval_kwargs(formation_error=torch.tensor(4.0)))
        assert abs(s.formation_error_sum - 6.0) < 1e-6

    def test_summarize_returns_correct_means(self) -> None:
        s = EvalStats()
        s.update(**_make_eval_kwargs(airborne_ratio=torch.tensor(0.8)))
        s.update(**_make_eval_kwargs(airborne_ratio=torch.tensor(1.0)))
        summary = s.summarize()
        assert abs(summary["airborne_ratio"] - 0.9) < 1e-6

    def test_summarize_handles_zero_count_safely(self) -> None:
        s = EvalStats()
        # No calls to update — counts are all 0.
        summary = s.summarize()
        for v in summary.values():
            assert v == 0.0, f"Expected 0.0 for empty stats, got {v}"

    def test_summarize_survival_steps_key_present(self) -> None:
        s = EvalStats()
        s.update(**_make_eval_kwargs())
        s.record_episode_survival(750.0)
        assert s.summarize()["survival_steps"] == pytest.approx(750.0)

    def test_record_episode_survival_mean_over_episodes(self) -> None:
        s = EvalStats()
        s.update(**_make_eval_kwargs())
        s.record_episode_survival(100.0)
        s.update(**_make_eval_kwargs())
        s.record_episode_survival(300.0)
        assert s.summarize()["survival_steps"] == pytest.approx(200.0)


class TestPhase2CollectorSurvival:
    """Episode survival = first step with batch ground hit, else full horizon."""

    def _make_env(self, z_sequence: list[tuple[float, float]]) -> object:
        """1 env, 2 agents; z_sequence gives (z0,z1) per on_step call."""
        from types import SimpleNamespace

        num_envs, na = 1, 2
        pos_flat = torch.zeros(num_envs * na * 3)
        lin_flat = torch.zeros_like(pos_flat)
        grav_flat = torch.zeros_like(pos_flat)
        grav_flat[2::3] = -1.0

        class RobotData:
            pass

        rd = RobotData()
        rd.root_pos_w = pos_flat
        rd.root_lin_vel_b = lin_flat
        rd.projected_gravity_b = grav_flat

        class Robot:
            data = rd

        env = SimpleNamespace(
            num_envs=num_envs,
            max_episode_length=len(z_sequence),
            cfg=SimpleNamespace(
                num_agents=na,
                target_formation_dist=0.2,
                min_separation_dist=0.1,
                min_height=0.1,
            ),
            _desired_pos_w=torch.ones(num_envs, na, 3),
            robot=Robot(),
            _z_sequence=z_sequence,
            _call_idx=[0],
            _pos_flat=pos_flat,
            _num_envs=num_envs,
            _na=na,
        )

        def _fill_pos(e: object) -> None:
            i = e._call_idx[0]
            z0, z1 = e._z_sequence[i]
            pw = e._pos_flat.view(e._num_envs, e._na, 3)
            pw[:, :, 2] = torch.tensor([z0, z1], dtype=pw.dtype)

        env._fill_pos = lambda: _fill_pos(env)
        return env

    def test_survival_full_horizon_when_no_ground_hit(self) -> None:
        from ggswarm_utils.phases.phase2 import Phase2Collector

        seq = [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)]
        env = self._make_env(seq)
        col = Phase2Collector()
        for t in range(len(seq)):
            env._call_idx[0] = t
            env._fill_pos()  # type: ignore[operator]
            col.on_step(
                base_env=env, obs=None, step=t, episode_step=t, episode_num=0
            )
        col.on_episode_end(0)
        assert col.summarize()["survival_steps"] == pytest.approx(3.0)

    def test_survival_first_ground_step(self) -> None:
        from ggswarm_utils.phases.phase2 import Phase2Collector

        seq = [(1.0, 1.0), (0.05, 1.0), (1.0, 1.0)]
        env = self._make_env(seq)
        col = Phase2Collector()
        for t in range(len(seq)):
            env._call_idx[0] = t
            env._fill_pos()  # type: ignore[operator]
            col.on_step(
                base_env=env, obs=None, step=t, episode_step=t, episode_num=0
            )
        col.on_episode_end(0)
        assert col.summarize()["survival_steps"] == pytest.approx(2.0)


class TestPairwiseMeanAbsSpacingError:
    def test_returns_zero_when_spacing_equals_target(self) -> None:
        # 1 env, 2 agents exactly 2 m apart
        pos = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])  # [1, 2, 3]
        err = pairwise_mean_abs_spacing_error(pos, target_dist=2.0)
        assert float(err.item()) < 1e-5

    def test_nonzero_error_when_spacing_differs(self) -> None:
        pos = torch.tensor([[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]]])  # actual dist=3, target=2
        err = pairwise_mean_abs_spacing_error(pos, target_dist=2.0)
        assert abs(float(err.item()) - 1.0) < 1e-5

    def test_multiple_envs_averaged(self) -> None:
        # env0: dist=2 (err=0), env1: dist=4 (err=2 against target=2)
        pos = torch.tensor([
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
            [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]],
        ])
        err = pairwise_mean_abs_spacing_error(pos, target_dist=2.0)
        assert abs(float(err.item()) - 1.0) < 1e-5


class TestComputeOrientationMetrics:
    def test_zero_roll_pitch_for_level_gravity(self) -> None:
        # Gravity pointing straight down in body frame => grav_b = [0, 0, -1]
        proj = torch.tensor([[[0.0, 0.0, -1.0]]])  # [1, 1, 3]
        roll, pitch, viol = compute_orientation_metrics(proj, orientation_threshold_deg=45.0)
        assert float(roll.item()) < 1e-5
        assert float(pitch.item()) < 1e-5

    def test_violation_when_tilted_past_threshold(self) -> None:
        # Tilt so roll > 45°: grav mostly in y-direction
        proj = torch.tensor([[[0.0, 1.0, 0.0]]])  # 90° roll
        _, _, viol = compute_orientation_metrics(proj, orientation_threshold_deg=45.0)
        assert float(viol.item()) == pytest.approx(1.0)

    def test_no_violation_when_below_threshold(self) -> None:
        # Small tilt — should be well under 45°
        proj = torch.tensor([[[0.1, 0.0, -0.995]]])
        _, _, viol = compute_orientation_metrics(proj, orientation_threshold_deg=45.0)
        assert float(viol.item()) == pytest.approx(0.0)


class TestComputeAltitudeMetrics:
    def test_returns_correct_std_and_mean(self) -> None:
        # 1 env, 2 agents at z=1 and z=3 → mean=2.0, std=sqrt(2) (Bessel-corrected, n=2)
        pos = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 0.0, 3.0]]])  # [1, 2, 3]
        std, mean = compute_altitude_metrics(pos, desired_pos_w=pos)
        assert abs(float(mean.item()) - 2.0) < 1e-5
        # torch.std uses Bessel correction by default: std([1, 3]) = sqrt(2) ≈ 1.4142
        import math
        assert abs(float(std.item()) - math.sqrt(2)) < 1e-4

    def test_uniform_altitude_has_zero_std(self) -> None:
        pos = torch.tensor([[[0.0, 0.0, 2.0], [1.0, 0.0, 2.0]]])  # both at z=2
        std, mean = compute_altitude_metrics(pos, desired_pos_w=pos)
        assert float(std.item()) == pytest.approx(0.0)
        assert abs(float(mean.item()) - 2.0) < 1e-5


# ---------------------------------------------------------------------------
# scorecard
# ---------------------------------------------------------------------------

from ggswarm_utils.scorecard import assess_verdict, print_scorecard  # noqa: E402


class TestAssessVerdict:
    def test_pass_for_good_airborne_ratio(self) -> None:
        verdict, _ = assess_verdict("airborne_ratio", 0.95)
        assert verdict == "PASS"

    def test_warn_for_borderline_airborne_ratio(self) -> None:
        verdict, _ = assess_verdict("airborne_ratio", 0.87)
        assert verdict == "WARN"

    def test_fail_for_bad_airborne_ratio(self) -> None:
        verdict, _ = assess_verdict("airborne_ratio", 0.3)
        assert verdict == "FAIL"

    def test_pass_for_low_ground_hit_rate(self) -> None:
        verdict, _ = assess_verdict("ground_hit_rate", 0.01)
        assert verdict == "PASS"

    def test_fail_for_high_ground_hit_rate(self) -> None:
        verdict, _ = assess_verdict("ground_hit_rate", 0.8)
        assert verdict == "FAIL"

    def test_pass_for_low_mean_roll(self) -> None:
        verdict, _ = assess_verdict("mean_roll_deg", 5.0)
        assert verdict == "PASS"

    def test_fail_for_high_mean_roll(self) -> None:
        verdict, _ = assess_verdict("mean_roll_deg", 70.0)
        assert verdict == "FAIL"

    def test_unknown_for_unrecognised_metric(self) -> None:
        verdict, hint = assess_verdict("not_a_real_metric", 1.0)
        assert verdict == "UNKNOWN"
        assert hint == "n/a"


class TestPrintScorecard:
    def test_returns_pass_when_all_metrics_pass(self, capsys: pytest.CaptureFixture) -> None:
        metrics = {
            "survival_steps": 600.0,
            "airborne_ratio": 0.95,
            "ground_hit_rate": 0.01,
            "mean_roll_deg": 5.0,
            "orientation_violation_rate": 0.02,
            "mean_formation_error_m": 0.3,
        }
        overall = print_scorecard("test_run", metrics)
        assert overall == "PASS"

    def test_returns_fail_when_any_metric_fails(self, capsys: pytest.CaptureFixture) -> None:
        metrics = {
            "survival_steps": 5.0,   # FAIL (< 10)
            "airborne_ratio": 0.95,
            "ground_hit_rate": 0.01,
            "mean_roll_deg": 5.0,
            "orientation_violation_rate": 0.02,
            "mean_formation_error_m": 0.3,
        }
        overall = print_scorecard("test_run", metrics)
        assert overall == "FAIL"

    def test_returns_warn_when_borderline(self, capsys: pytest.CaptureFixture) -> None:
        metrics = {
            "survival_steps": 600.0,
            "airborne_ratio": 0.87,   # WARN
            "ground_hit_rate": 0.01,
            "mean_roll_deg": 5.0,
            "orientation_violation_rate": 0.02,
            "mean_formation_error_m": 0.3,
        }
        overall = print_scorecard("test_run", metrics)
        assert overall == "WARN"


# ---------------------------------------------------------------------------
# gcs_sync
# ---------------------------------------------------------------------------

from ggswarm_utils.gcs_sync import has_local_data  # noqa: E402


class TestHasLocalData:
    def test_returns_false_for_missing_directory(self, tmp_path: Path) -> None:
        assert has_local_data(tmp_path / "does_not_exist") is False

    def test_returns_false_for_empty_checkpoints_dir(self, tmp_path: Path) -> None:
        (tmp_path / "checkpoints").mkdir()
        assert has_local_data(tmp_path) is False

    def test_returns_true_when_pt_file_present(self, tmp_path: Path) -> None:
        ckpts = tmp_path / "checkpoints"
        ckpts.mkdir()
        (ckpts / "best_agent.pt").write_bytes(b"fake")
        assert has_local_data(tmp_path) is True

    def test_returns_false_when_only_non_pt_files_present(self, tmp_path: Path) -> None:
        ckpts = tmp_path / "checkpoints"
        ckpts.mkdir()
        (ckpts / "readme.txt").write_text("not a checkpoint")
        assert has_local_data(tmp_path) is False


# ---------------------------------------------------------------------------
# checkpoint
# ---------------------------------------------------------------------------

from ggswarm_utils.checkpoint import (  # noqa: E402
    find_best_checkpoint,
    validate_eval_checkpoint_path,
)


class TestFindBestCheckpoint:
    def test_returns_best_agent_when_present(self, tmp_path: Path) -> None:
        ckpts = tmp_path / "checkpoints"
        ckpts.mkdir()
        best = ckpts / "best_agent.pt"
        best.write_bytes(b"fake")
        (ckpts / "agent_000001000.pt").write_bytes(b"fake")
        result = find_best_checkpoint(tmp_path)
        assert result.name == "best_agent.pt"

    def test_falls_back_to_highest_numbered(self, tmp_path: Path) -> None:
        ckpts = tmp_path / "checkpoints"
        ckpts.mkdir()
        (ckpts / "agent_000001000.pt").write_bytes(b"fake")
        (ckpts / "agent_000002000.pt").write_bytes(b"fake")
        result = find_best_checkpoint(tmp_path)
        assert result.name == "agent_000002000.pt"

    def test_raises_when_no_checkpoints(self, tmp_path: Path) -> None:
        (tmp_path / "checkpoints").mkdir()
        with pytest.raises(FileNotFoundError):
            find_best_checkpoint(tmp_path)


class TestValidateEvalCheckpointPath:
    def test_rejects_angle_bracket_placeholder(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="placeholder"):
            validate_eval_checkpoint_path(
                "logs/skrl/ggswarm_marl/<run>/checkpoints/best_agent.pt"
            )

    def test_raises_when_file_missing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        rel = Path("logs/skrl/ggswarm_marl/real_run/checkpoints/best_agent.pt")
        rel.parent.mkdir(parents=True)
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            validate_eval_checkpoint_path(rel)

    def test_accepts_existing_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        ckpt = tmp_path / "logs" / "run" / "checkpoints" / "best_agent.pt"
        ckpt.parent.mkdir(parents=True)
        ckpt.write_bytes(b"x")
        validate_eval_checkpoint_path(ckpt)


# ---------------------------------------------------------------------------
# cli
# ---------------------------------------------------------------------------

from ggswarm_utils.cli import (  # noqa: E402
    acquire_single_instance_lock,
    build_assess_cmd,
    build_eval_cmd,
    build_play_cmd,
    build_train_cmd,
    find_latest_checkpoint,
    release_lock,
)


def _make_namespace(**kwargs) -> argparse.Namespace:
    """Return a minimal Namespace with defaults appropriate for build_* tests."""
    defaults = dict(
        headless=False,
        device=None,
        num_envs=None,
        num_agents=None,
        max_iterations=None,
        action_telemetry_steps=None,
        no_progress=False,
        progress_interval_s=None,
        eta_window_s=None,
        checkpoint=None,
        gnn=False,
        no_gnn=False,
        max_steps=None,
        video=False,
        video_length=None,
        rendering_mode=None,
        video_codec=None,
        video_bitrate=None,
        video_preset=None,
        video_ffmpeg_params=None,
        hover_debug=False,
        num_episodes=5,
        run_dir="logs/skrl/ggswarm_marl/run1",
    )
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


class TestBuildTrainCmd:
    def test_includes_headless_when_set(self) -> None:
        args = _make_namespace(headless=True)
        cmd = build_train_cmd("MyTask", args, gnn_default=False)
        assert "--headless" in cmd

    def test_excludes_headless_when_not_set(self) -> None:
        args = _make_namespace(headless=False)
        cmd = build_train_cmd("MyTask", args, gnn_default=False)
        assert "--headless" not in cmd

    def test_includes_gnn_when_gnn_default_and_no_no_gnn(self) -> None:
        args = _make_namespace(no_gnn=False)
        cmd = build_train_cmd("MyTask", args, gnn_default=True)
        assert "--gnn" in cmd

    def test_excludes_gnn_when_no_gnn_flag(self) -> None:
        args = _make_namespace(no_gnn=True)
        cmd = build_train_cmd("MyTask", args, gnn_default=True)
        assert "--gnn" not in cmd

    def test_excludes_gnn_when_gnn_default_false(self) -> None:
        args = _make_namespace(no_gnn=False)
        cmd = build_train_cmd("MyTask", args, gnn_default=False)
        assert "--gnn" not in cmd

    def test_includes_max_iterations_when_set(self) -> None:
        args = _make_namespace(max_iterations=80000)
        cmd = build_train_cmd("MyTask", args, gnn_default=False)
        assert "--max_iterations" in cmd
        assert "80000" in cmd

    def test_task_present(self) -> None:
        args = _make_namespace()
        cmd = build_train_cmd("GGS-Hover-v0", args, gnn_default=False)
        assert "GGS-Hover-v0" in cmd

    def test_checkpoint_forwarded_when_set(self) -> None:
        args = _make_namespace(checkpoint="logs/run1/checkpoints/best_agent.pt")
        cmd = build_train_cmd("T", args, gnn_default=False)
        assert "--checkpoint" in cmd
        assert "logs/run1/checkpoints/best_agent.pt" in cmd

    def test_action_telemetry_steps_forwarded_when_set(self) -> None:
        args = _make_namespace(action_telemetry_steps=200)
        cmd = build_train_cmd("T", args, gnn_default=False)
        assert "--action_telemetry_steps" in cmd
        assert "200" in cmd

    def test_explicit_gnn_when_family_defaults_mlp(self) -> None:
        """--gnn on CLI enables train.py --gnn when gnn_default is False."""
        args = _make_namespace(gnn=True, no_gnn=False)
        cmd = build_train_cmd("T", args, gnn_default=False)
        assert "--gnn" in cmd


class TestBuildPlayCmd:
    def test_includes_video_flag(self) -> None:
        args = _make_namespace(video=True)
        cmd = build_play_cmd("Task", args, gnn_default=False)
        assert "--video" in cmd

    def test_excludes_video_flag_when_not_set(self) -> None:
        args = _make_namespace(video=False)
        cmd = build_play_cmd("Task", args, gnn_default=False)
        assert "--video" not in cmd

    def test_auto_discover_checkpoint_from_log_root(self, tmp_path: Path) -> None:
        ckpts = tmp_path / "run1" / "checkpoints"
        ckpts.mkdir(parents=True)
        (ckpts / "best_agent.pt").write_bytes(b"fake")
        args = _make_namespace(checkpoint=None)
        cmd = build_play_cmd("Task", args, gnn_default=False, log_root=tmp_path)
        assert "--checkpoint" in cmd
        assert "best_agent.pt" in " ".join(cmd)

    def test_no_checkpoint_when_log_root_empty(self, tmp_path: Path) -> None:
        args = _make_namespace(checkpoint=None)
        cmd = build_play_cmd("Task", args, gnn_default=False, log_root=tmp_path)
        assert "--checkpoint" not in cmd


class TestBuildEvalCmd:
    """Tests for build_eval_cmd with new phase-based signature."""

    def test_includes_num_episodes(self) -> None:
        args = _make_namespace(num_episodes=10)
        cmd = build_eval_cmd("2a", args)
        assert "--num_episodes" in cmd
        assert "10" in cmd

    def test_script_path_in_cmd(self) -> None:
        args = _make_namespace()
        cmd = build_eval_cmd("2b", args)
        assert "scripts/eval.py" in cmd

    def test_headless_always_set(self) -> None:
        args = _make_namespace(headless=False)
        cmd = build_eval_cmd("hover", args)
        assert "--headless" in cmd

    def test_extra_flags_appended(self) -> None:
        args = _make_namespace()
        cmd = build_eval_cmd("3", args, extra_flags=["--kill_agent_step", "50"])
        assert "--kill_agent_step" in cmd
        assert "50" in cmd

    def test_video_forwards_encoding_flags(self) -> None:
        args = _make_namespace(
            video=True,
            video_length=300,
            video_codec="libx265",
            rendering_mode="balanced",
        )
        cmd = build_eval_cmd("2a", args)
        assert "--video" in cmd
        assert "--video_length" in cmd
        assert "300" in cmd
        assert "--video_codec" in cmd
        assert "libx265" in cmd
        assert "--rendering_mode" in cmd
        assert "balanced" in cmd


class TestFindLatestCheckpointCli:
    def test_raises_when_log_root_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_latest_checkpoint(tmp_path / "nonexistent")

    def test_raises_when_no_checkpoints(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            find_latest_checkpoint(tmp_path)

    def test_returns_best_agent_when_present(self, tmp_path: Path) -> None:
        run = tmp_path / "run1"
        ckpts = run / "checkpoints"
        ckpts.mkdir(parents=True)
        (ckpts / "best_agent.pt").write_bytes(b"fake")
        result = find_latest_checkpoint(tmp_path, prefer_best=True)
        assert "best_agent.pt" in result

    def test_falls_back_to_numbered_when_no_best(self, tmp_path: Path) -> None:
        run = tmp_path / "run1"
        ckpts = run / "checkpoints"
        ckpts.mkdir(parents=True)
        (ckpts / "agent_000001000.pt").write_bytes(b"fake")
        result = find_latest_checkpoint(tmp_path, prefer_best=True)
        assert "agent_000001000.pt" in result


class TestLockHelpers:
    def test_acquire_creates_lock_file(self, tmp_path: Path) -> None:
        lock = tmp_path / "train.lock"
        acquire_single_instance_lock(lock)
        assert lock.exists()
        release_lock(lock)

    def test_release_removes_lock_file(self, tmp_path: Path) -> None:
        lock = tmp_path / "train.lock"
        acquire_single_instance_lock(lock)
        release_lock(lock)
        assert not lock.exists()

    def test_release_is_idempotent(self, tmp_path: Path) -> None:
        lock = tmp_path / "train.lock"
        release_lock(lock)  # should not raise

    def test_double_acquire_raises_system_exit(self, tmp_path: Path) -> None:
        lock = tmp_path / "train.lock"
        acquire_single_instance_lock(lock, ttl_s=9999)
        with pytest.raises(SystemExit):
            acquire_single_instance_lock(lock, ttl_s=9999)
        release_lock(lock)

    def test_stale_lock_is_overwritten(self, tmp_path: Path) -> None:
        lock = tmp_path / "train.lock"
        lock.write_text("pid=9999 time_s=0\n")
        # Set mtime to ancient past so TTL=1 s considers it stale
        import os
        os.utime(str(lock), (0, 0))
        acquire_single_instance_lock(lock, ttl_s=1)
        assert lock.exists()
        release_lock(lock)


# ---------------------------------------------------------------------------
# sim_helpers tests
# ---------------------------------------------------------------------------

from ggswarm_utils.sim_helpers import (  # noqa: E402
    PHASE_REGISTRY,
    PhaseConfig,
    configure_eval_cfg,
    extract_actions,
    override_agent_count,
    phase_from_task,
    resolve_phase,
)


class TestPhaseRegistry:
    """Tests for PHASE_REGISTRY coverage and resolve_phase."""

    ALL_PHASES = ["hover", "2", "2a", "2b", "2c", "3", "4"]

    def test_all_expected_phases_present(self) -> None:
        for p in self.ALL_PHASES:
            assert p in PHASE_REGISTRY, f"Phase '{p}' missing from PHASE_REGISTRY"

    def test_each_entry_is_phase_config(self) -> None:
        for key, val in PHASE_REGISTRY.items():
            assert isinstance(val, PhaseConfig), f"Registry['{key}'] is not a PhaseConfig"

    def test_all_tasks_non_empty(self) -> None:
        for key, cfg in PHASE_REGISTRY.items():
            assert cfg.task, f"Phase '{key}' has empty task string"

    def test_all_collector_paths_non_empty(self) -> None:
        for key, cfg in PHASE_REGISTRY.items():
            assert cfg.collector_cls, f"Phase '{key}' has empty collector_cls"

    def test_resolve_phase_returns_correct_config(self) -> None:
        cfg = resolve_phase("2a")
        assert cfg.task == "Template-GGSwarm-Marl-HoverStability-v0"
        assert cfg.gnn_default is True

    def test_resolve_phase_hover(self) -> None:
        cfg = resolve_phase("hover")
        assert cfg.task == "GGS-Hover-v0"
        assert cfg.gnn_default is False

    def test_resolve_phase_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown phase"):
            resolve_phase("999")

    def test_phase_from_task_known(self) -> None:
        key = phase_from_task("GGS-Hover-v0")
        assert key == "hover"

    def test_phase_from_task_unknown_returns_none(self) -> None:
        key = phase_from_task("NonExistent-Task-v0")
        assert key is None


class TestOverrideAgentCount:
    """Tests for sim_helpers.override_agent_count."""

    class _FakeCfg:
        num_agents: int = 3
        possible_agents: list = []
        action_spaces: dict = {}
        observation_spaces: dict = {}

    def test_sets_num_agents(self) -> None:
        cfg = self._FakeCfg()
        override_agent_count(cfg, 5)
        assert cfg.num_agents == 5

    def test_builds_possible_agents(self) -> None:
        cfg = self._FakeCfg()
        override_agent_count(cfg, 4)
        assert cfg.possible_agents == ["drone_0", "drone_1", "drone_2", "drone_3"]

    def test_builds_action_spaces(self) -> None:
        cfg = self._FakeCfg()
        override_agent_count(cfg, 3)
        assert cfg.action_spaces == {"drone_0": 4, "drone_1": 4, "drone_2": 4}

    def test_builds_observation_spaces_default(self) -> None:
        cfg = self._FakeCfg()
        override_agent_count(cfg, 2)
        assert cfg.observation_spaces == {"drone_0": 12, "drone_1": 12}

    def test_builds_observation_spaces_custom_dim(self) -> None:
        cfg = self._FakeCfg()
        override_agent_count(cfg, 2, obs_dim=14)
        assert cfg.observation_spaces == {"drone_0": 14, "drone_1": 14}


class TestExtractActions:
    """Tests for sim_helpers.extract_actions (mocked agent/env)."""

    class _FakeAgent:
        def act(self, obs, timestep, timesteps):
            # Returns (out_0, _ignored, last) tuple matching skrl agent.act()
            last = {"mean_actions": torch.tensor([1.0, 2.0, 3.0, 4.0])}
            return (last, None, last)

    class _FakeMAAgent:
        def __init__(self, agent_ids):
            self._ids = agent_ids

        def act(self, obs, timestep, timesteps):
            last = {a: {"mean_actions": torch.ones(4)} for a in self._ids}
            return (last, None, last)

    class _SingleAgentEnv:
        pass  # no possible_agents attribute

    class _MultiAgentEnv:
        possible_agents = ["drone_0", "drone_1"]

    def test_single_agent_returns_tensor(self) -> None:
        agent = self._FakeAgent()
        env = self._SingleAgentEnv()
        result = extract_actions(agent, None, env)
        assert isinstance(result, torch.Tensor)

    def test_multi_agent_returns_dict(self) -> None:
        ids = ["drone_0", "drone_1"]
        agent = self._FakeMAAgent(ids)
        env = self._MultiAgentEnv()
        result = extract_actions(agent, None, env)
        assert isinstance(result, dict)
        assert set(result.keys()) == set(ids)

    def test_multi_agent_values_are_tensors(self) -> None:
        ids = ["drone_0", "drone_1"]
        agent = self._FakeMAAgent(ids)
        env = self._MultiAgentEnv()
        result = extract_actions(agent, None, env)
        for v in result.values():
            assert isinstance(v, torch.Tensor)


class TestConfigureEvalCfg:
    """Tests for sim_helpers.configure_eval_cfg."""

    def _make_cfg(self) -> dict:
        return {
            "trainer": {"checkpoint_interval": 100},
            "agent": {
                "experiment": {
                    "directory": "old_dir",
                    "experiment_name": "old_name",
                }
            },
        }

    def test_zeroes_checkpoint_interval(self) -> None:
        cfg = self._make_cfg()
        configure_eval_cfg(cfg, "/logs/root", "run_001")
        assert cfg["trainer"]["checkpoint_interval"] == 0

    def test_sets_directory(self) -> None:
        cfg = self._make_cfg()
        configure_eval_cfg(cfg, "/logs/root", "run_001")
        assert cfg["agent"]["experiment"]["directory"] == "/logs/root"

    def test_sets_experiment_name(self) -> None:
        cfg = self._make_cfg()
        configure_eval_cfg(cfg, "/logs/root", "run_001")
        assert cfg["agent"]["experiment"]["experiment_name"] == "run_001"


# ---------------------------------------------------------------------------
# attitude_controller tests
# ---------------------------------------------------------------------------
# conftest.py injects source/ggSwarm/ggSwarm/tasks/direct/ggswarm_marl into
# sys.path, so these flat imports work without the full package hierarchy.

from attitude_controller import (  # noqa: E402
    AttitudeControllerParams,
    compute_attitude_control,
)


def _make_att_params(**overrides) -> AttitudeControllerParams:
    defaults = dict(
        kp_att=0.045,
        kd_att=0.005,
        kp_yaw=0.01,
        max_tilt_angle=0.52,
        max_yaw_rate=3.14159,
        max_moment=0.03,
        thrust_to_weight=2.0,
    )
    defaults.update(overrides)
    return AttitudeControllerParams(**defaults)


def _make_level_state(n: int = 4):
    """Level drone state: gravity pointing straight down in body frame."""
    proj_grav_b = torch.zeros(n, 3)
    proj_grav_b[:, 2] = -1.0  # level: [0, 0, -1]
    ang_vel_b = torch.zeros(n, 3)
    return proj_grav_b, ang_vel_b


class TestAttitudeController:
    def _run(self, actions, proj_grav_b, ang_vel_b, robot_weight=0.274, **param_overrides):
        params = _make_att_params(**param_overrides)
        n = actions.shape[0]
        thrust_buf = torch.zeros(n, 1, 3)
        moment_buf = torch.zeros(n, 1, 3)
        compute_attitude_control(
            actions=actions,
            proj_grav_b=proj_grav_b,
            ang_vel_b=ang_vel_b,
            robot_weight=robot_weight,
            params=params,
            thrust_buf=thrust_buf,
            moment_buf=moment_buf,
            moment_pre_clamp_buf=None,
        )
        return thrust_buf, moment_buf

    def test_neutral_action_produces_hover_thrust(self) -> None:
        """action[0]=0 -> thrust_val=0.5 -> total_thrust = thrust_to_weight * weight * 0.5."""
        actions = torch.zeros(1, 4)
        proj_grav_b, ang_vel_b = _make_level_state(1)
        robot_weight = 0.274  # 28g Crazyflie * 9.81
        thrust_buf, _ = self._run(actions, proj_grav_b, ang_vel_b, robot_weight=robot_weight)
        expected = 2.0 * robot_weight * 0.5
        assert abs(float(thrust_buf[0, 0, 2]) - expected) < 1e-5

    def test_max_thrust_at_action_plus_one(self) -> None:
        """action[0]=1 -> thrust_val=1.0 -> full thrust_to_weight."""
        actions = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        proj_grav_b, ang_vel_b = _make_level_state(1)
        robot_weight = 0.274
        thrust_buf, _ = self._run(actions, proj_grav_b, ang_vel_b, robot_weight=robot_weight)
        expected = 2.0 * robot_weight * 1.0
        assert abs(float(thrust_buf[0, 0, 2]) - expected) < 1e-5

    def test_zero_moment_when_level_and_zero_attitude_command(self) -> None:
        """Level drone + zero attitude command -> moments should be near zero."""
        actions = torch.zeros(1, 4)
        proj_grav_b, ang_vel_b = _make_level_state(1)
        _, moment_buf = self._run(actions, proj_grav_b, ang_vel_b)
        assert moment_buf.abs().max().item() < 1e-6

    def test_positive_roll_command_produces_positive_roll_moment(self) -> None:
        """Positive desired roll with level drone -> positive roll moment."""
        actions = torch.tensor([[0.0, 1.0, 0.0, 0.0]])  # max positive roll command
        proj_grav_b, ang_vel_b = _make_level_state(1)
        _, moment_buf = self._run(actions, proj_grav_b, ang_vel_b)
        assert float(moment_buf[0, 0, 0]) > 0.0

    def test_moment_clamp_limits_output(self) -> None:
        """Very large attitude error should be clamped to max_moment."""
        # Create 90-degree roll error: drone tilted fully sideways
        actions = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        proj_grav_b = torch.tensor([[0.0, -1.0, 0.0]])  # 90° roll
        ang_vel_b = torch.zeros(1, 3)
        max_moment = 0.02
        _, moment_buf = self._run(
            actions, proj_grav_b, ang_vel_b, max_moment=max_moment
        )
        assert moment_buf.abs().max().item() <= max_moment + 1e-6

    def test_moment_pre_clamp_buffer_records_unclamped_torque(self) -> None:
        """Optional buffer receives moments before max_moment saturation."""
        actions = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
        proj_grav_b = torch.tensor([[0.0, -1.0, 0.0]])
        ang_vel_b = torch.zeros(1, 3)
        params = _make_att_params(max_moment=0.02)
        thrust_buf = torch.zeros(1, 1, 3)
        moment_buf = torch.zeros(1, 1, 3)
        pre_buf = torch.zeros(1, 1, 3)
        compute_attitude_control(
            actions=actions,
            proj_grav_b=proj_grav_b,
            ang_vel_b=ang_vel_b,
            robot_weight=0.274,
            params=params,
            thrust_buf=thrust_buf,
            moment_buf=moment_buf,
            moment_pre_clamp_buf=pre_buf,
        )
        assert float(pre_buf[0, 0, 0].abs()) > float(moment_buf[0, 0, 0].abs()) + 1e-8
        assert float(moment_buf[0, 0, 0].abs()) <= 0.02 + 1e-6

    def test_moment_saturation_fraction_roll_grid(self) -> None:
        """Across roll angles, many samples need torque clamping (PD authority audit)."""
        n = 64
        phis = torch.linspace(-math.pi / 2 + 0.05, math.pi / 2 - 0.05, n)
        grav_y = torch.sin(phis)
        grav_z = -torch.cos(phis)
        proj = torch.stack(
            [torch.zeros(n), grav_y, grav_z],
            dim=1,
        )
        actions = torch.zeros(n, 4)
        actions[:, 1] = 1.0  # max desired roll magnifies moment demand
        ang_vel_b = torch.zeros(n, 3)
        params = _make_att_params()
        thrust_buf = torch.zeros(n, 1, 3)
        moment_buf = torch.zeros(n, 1, 3)
        pre_buf = torch.zeros(n, 1, 3)
        compute_attitude_control(
            actions=actions,
            proj_grav_b=proj,
            ang_vel_b=ang_vel_b,
            robot_weight=0.274,
            params=params,
            thrust_buf=thrust_buf,
            moment_buf=moment_buf,
            moment_pre_clamp_buf=pre_buf,
        )
        mm = params.max_moment
        saturated = (pre_buf[:, 0, :].abs() >= mm - 1e-6).any(dim=-1).float().mean().item()
        assert saturated > 0.2

    def test_thrust_z_axis_only(self) -> None:
        """Thrust should only be in Z axis of body frame (X and Y must be zero)."""
        actions = torch.zeros(4, 4)
        proj_grav_b, ang_vel_b = _make_level_state(4)
        thrust_buf, _ = self._run(actions, proj_grav_b, ang_vel_b)
        assert thrust_buf[:, :, 0].abs().max().item() < 1e-6
        assert thrust_buf[:, :, 1].abs().max().item() < 1e-6

    def test_batched_output_shapes(self) -> None:
        """Output buffers must have shape [N, 1, 3]."""
        n = 8
        actions = torch.rand(n, 4) * 2 - 1
        proj_grav_b, ang_vel_b = _make_level_state(n)
        thrust_buf, moment_buf = self._run(actions, proj_grav_b, ang_vel_b)
        assert thrust_buf.shape == (n, 1, 3)
        assert moment_buf.shape == (n, 1, 3)

    def test_derivative_damping_opposes_angular_velocity(self) -> None:
        """With non-zero ang_vel and zero desired attitude, moment should oppose velocity."""
        actions = torch.zeros(1, 4)  # zero desired roll/pitch
        proj_grav_b, _ = _make_level_state(1)
        ang_vel_b = torch.tensor([[2.0, 0.0, 0.0]])  # positive roll rate
        _, moment_buf = self._run(actions, proj_grav_b, ang_vel_b)
        # Damping should produce negative roll moment
        assert float(moment_buf[0, 0, 0]) < 0.0


# ---------------------------------------------------------------------------
# compute_stable_hover_rewards tests
# ---------------------------------------------------------------------------

from contract_logic import (  # noqa: E402
    StableHoverRewardParams,
    compute_stable_hover_rewards,
)


def _make_hover_params(**overrides) -> StableHoverRewardParams:
    defaults = dict(
        rew_scale_pos=15.0,
        rew_scale_vel=-0.05,
        rew_scale_ang_vel=-0.01,
        step_dt=0.02,
        min_height=0.1,
        low_clearance_margin_m=0.2,
        rew_scale_low_clearance=0.0,
        rew_scale_terminated=0.0,
        pos_tanh_sigma=0.8,
    )
    defaults.update(overrides)
    return StableHoverRewardParams(**defaults)


class TestComputeStableHoverRewards:
    def test_output_shape(self) -> None:
        pos = torch.zeros(4, 3, 3)
        desired = torch.zeros(4, 3, 3)
        lin_vel = torch.zeros(4, 3, 3)
        ang_vel = torch.zeros(4, 3, 3)
        rewards = compute_stable_hover_rewards(
            pos_w=pos, desired_pos_w=desired, lin_vel_b=lin_vel, ang_vel_b=ang_vel,
            params=_make_hover_params(),
        )
        assert rewards.shape == (4, 3)

    def test_max_reward_at_goal_with_zero_velocity(self) -> None:
        """At goal (dist=0) with zero velocity, reward = rew_scale_pos * step_dt."""
        pos = torch.zeros(1, 1, 3)
        desired = torch.zeros(1, 1, 3)
        lin_vel = torch.zeros(1, 1, 3)
        ang_vel = torch.zeros(1, 1, 3)
        params = _make_hover_params()
        rewards = compute_stable_hover_rewards(
            pos_w=pos, desired_pos_w=desired, lin_vel_b=lin_vel, ang_vel_b=ang_vel,
            params=params,
        )
        expected = params.rew_scale_pos * params.step_dt  # 1.0 - tanh(0) = 1.0
        assert abs(float(rewards[0, 0]) - expected) < 1e-5

    def test_reward_decreases_with_distance(self) -> None:
        """Reward should be monotonically decreasing as distance increases."""
        params = _make_hover_params()
        lin_vel = torch.zeros(1, 1, 3)
        ang_vel = torch.zeros(1, 1, 3)
        desired = torch.zeros(1, 1, 3)
        rewards = []
        for dist in [0.0, 0.5, 1.0, 2.0, 5.0]:
            pos = torch.tensor([[[dist, 0.0, 0.0]]])
            r = compute_stable_hover_rewards(
                pos_w=pos, desired_pos_w=desired, lin_vel_b=lin_vel, ang_vel_b=ang_vel,
                params=params,
            )
            rewards.append(float(r[0, 0]))
        for i in range(1, len(rewards)):
            assert rewards[i] < rewards[i - 1], f"Reward not decreasing: {rewards}"

    def test_velocity_penalty_reduces_reward(self) -> None:
        """Non-zero velocity should reduce total reward vs zero velocity."""
        params = _make_hover_params()
        pos = desired = torch.zeros(1, 1, 3)
        ang_vel = torch.zeros(1, 1, 3)
        r_zero = compute_stable_hover_rewards(
            pos_w=pos, desired_pos_w=desired, lin_vel_b=torch.zeros(1, 1, 3),
            ang_vel_b=ang_vel, params=params,
        )
        r_moving = compute_stable_hover_rewards(
            pos_w=pos, desired_pos_w=desired, lin_vel_b=torch.ones(1, 1, 3) * 2.0,
            ang_vel_b=ang_vel, params=params,
        )
        assert float(r_moving[0, 0]) < float(r_zero[0, 0])

    def test_ang_vel_penalty_reduces_reward(self) -> None:
        """Spinning drone should get lower reward than stationary."""
        params = _make_hover_params()
        pos = desired = torch.zeros(1, 1, 3)
        lin_vel = torch.zeros(1, 1, 3)
        r_still = compute_stable_hover_rewards(
            pos_w=pos, desired_pos_w=desired, lin_vel_b=lin_vel,
            ang_vel_b=torch.zeros(1, 1, 3), params=params,
        )
        r_spinning = compute_stable_hover_rewards(
            pos_w=pos, desired_pos_w=desired, lin_vel_b=lin_vel,
            ang_vel_b=torch.ones(1, 1, 3) * 5.0, params=params,
        )
        assert float(r_spinning[0, 0]) < float(r_still[0, 0])

    def test_step_dt_scales_reward_proportionally(self) -> None:
        """Doubling step_dt should double the total reward magnitude."""
        pos = desired = torch.zeros(1, 1, 3)
        lin_vel = ang_vel = torch.zeros(1, 1, 3)
        r1 = compute_stable_hover_rewards(
            pos_w=pos, desired_pos_w=desired, lin_vel_b=lin_vel, ang_vel_b=ang_vel,
            params=_make_hover_params(step_dt=0.01),
        )
        r2 = compute_stable_hover_rewards(
            pos_w=pos, desired_pos_w=desired, lin_vel_b=lin_vel, ang_vel_b=ang_vel,
            params=_make_hover_params(step_dt=0.02),
        )
        assert abs(float(r2[0, 0]) - 2.0 * float(r1[0, 0])) < 1e-5

    def test_return_terms_matches_total(self) -> None:
        """return_terms=True should decompose into components summing to total."""
        pos = torch.zeros(2, 2, 3)
        desired = torch.zeros(2, 2, 3)
        lin_vel = torch.zeros(2, 2, 3)
        ang_vel = torch.zeros(2, 2, 3)
        params = _make_hover_params(rew_scale_low_clearance=-1.0)
        pos[:, :, 2] = 0.05
        total, terms = compute_stable_hover_rewards(
            pos_w=pos,
            desired_pos_w=desired,
            lin_vel_b=lin_vel,
            ang_vel_b=ang_vel,
            params=params,
            return_terms=True,
        )
        summed = (
            terms["rew_pos"]
            + terms["rew_vel"]
            + terms["rew_ang_vel"]
            + terms["rew_low_clearance"]
            + terms["rew_terminated"]
        )
        assert torch.allclose(total, summed)

    def test_ground_hit_applies_rew_scale_terminated(self) -> None:
        """Below min_height applies rew_scale_terminated each step; scale 0 disables."""
        desired = torch.zeros(1, 1, 3)
        lin_vel = torch.zeros(1, 1, 3)
        ang_vel = torch.zeros(1, 1, 3)
        pos_ground = torch.zeros(1, 1, 3)
        pos_ground[..., 2] = 0.05  # below default min_height 0.1
        pos_safe = torch.zeros(1, 1, 3)
        pos_safe[..., 2] = 0.5

        params_pen = _make_hover_params(rew_scale_terminated=-5.0)
        total_g, terms_g = compute_stable_hover_rewards(
            pos_w=pos_ground,
            desired_pos_w=desired,
            lin_vel_b=lin_vel,
            ang_vel_b=ang_vel,
            params=params_pen,
            return_terms=True,
        )
        assert torch.allclose(terms_g["rew_terminated"], torch.tensor(-5.0))
        summed_g = (
            terms_g["rew_pos"]
            + terms_g["rew_vel"]
            + terms_g["rew_ang_vel"]
            + terms_g["rew_low_clearance"]
            + terms_g["rew_terminated"]
        )
        assert torch.allclose(total_g, summed_g)

        params_off = _make_hover_params(rew_scale_terminated=0.0)
        _, terms_off = compute_stable_hover_rewards(
            pos_w=pos_ground,
            desired_pos_w=desired,
            lin_vel_b=lin_vel,
            ang_vel_b=ang_vel,
            params=params_off,
            return_terms=True,
        )
        assert torch.allclose(terms_off["rew_terminated"], torch.zeros_like(terms_off["rew_terminated"]))

        _, terms_hi = compute_stable_hover_rewards(
            pos_w=pos_safe,
            desired_pos_w=desired,
            lin_vel_b=lin_vel,
            ang_vel_b=ang_vel,
            params=params_pen,
            return_terms=True,
        )
        assert torch.allclose(terms_hi["rew_terminated"], torch.zeros_like(terms_hi["rew_terminated"]))

    def test_low_clearance_term_negative_when_low(self) -> None:
        """Below clearance band applies rew_scale_low_clearance * depth."""
        pos_hi = torch.zeros(1, 1, 3)
        pos_hi[..., 2] = 1.0
        pos_lo = torch.zeros(1, 1, 3)
        pos_lo[..., 2] = 0.05
        desired = torch.zeros(1, 1, 3)
        z = torch.zeros(1, 1, 3)
        params = _make_hover_params(rew_scale_low_clearance=-2.0)
        _, terms_hi = compute_stable_hover_rewards(
            pos_w=pos_hi,
            desired_pos_w=desired,
            lin_vel_b=z,
            ang_vel_b=z,
            params=params,
            return_terms=True,
        )
        _, terms_lo = compute_stable_hover_rewards(
            pos_w=pos_lo,
            desired_pos_w=desired,
            lin_vel_b=z,
            ang_vel_b=z,
            params=params,
            return_terms=True,
        )
        assert float(terms_lo["rew_low_clearance"][0, 0]) < float(
            terms_hi["rew_low_clearance"][0, 0]
        )

    def test_smaller_sigma_sharper_dropoff(self) -> None:
        """Smaller pos_tanh_sigma makes reward drop faster with distance."""
        z = torch.zeros(1, 1, 3)
        desired = torch.zeros(1, 1, 3)
        pos_half_m = torch.tensor([[[0.5, 0.0, 0.0]]])  # 0.5 m from goal

        r_wide = compute_stable_hover_rewards(
            pos_w=pos_half_m, desired_pos_w=desired, lin_vel_b=z, ang_vel_b=z,
            params=_make_hover_params(pos_tanh_sigma=0.8),
        )
        r_tight = compute_stable_hover_rewards(
            pos_w=pos_half_m, desired_pos_w=desired, lin_vel_b=z, ang_vel_b=z,
            params=_make_hover_params(pos_tanh_sigma=0.25),
        )
        # At 0.5m: sigma=0.8 gives ~45% of max; sigma=0.25 gives ~5% of max
        assert float(r_tight[0, 0]) < float(r_wide[0, 0])
        # Tight sigma should give less than 15% of max reward at 0.5m
        max_r = 15.0 * 0.02  # rew_scale_pos * step_dt
        assert float(r_tight[0, 0]) < 0.15 * max_r

    def test_sigma_does_not_affect_reward_at_goal(self) -> None:
        """At dist=0, sigma should not matter — reward is always max."""
        z = torch.zeros(1, 1, 3)
        r_wide = compute_stable_hover_rewards(
            pos_w=z, desired_pos_w=z, lin_vel_b=z, ang_vel_b=z,
            params=_make_hover_params(pos_tanh_sigma=0.8),
        )
        r_tight = compute_stable_hover_rewards(
            pos_w=z, desired_pos_w=z, lin_vel_b=z, ang_vel_b=z,
            params=_make_hover_params(pos_tanh_sigma=0.25),
        )
        assert abs(float(r_wide[0, 0]) - float(r_tight[0, 0])) < 1e-5


class TestBuildEvalCmdPhase:
    """Tests for updated cli.build_eval_cmd (phase-based signature)."""

    def _make_args(self, **kwargs) -> argparse.Namespace:
        defaults = {
            "no_gnn": False,
            "device": None,
            "num_envs": None,
            "num_agents": None,
            "num_episodes": None,
            "checkpoint": None,
            "video": False,
            "video_length": 200,
            "rendering_mode": None,
            "video_codec": None,
            "video_bitrate": None,
            "video_preset": None,
            "video_ffmpeg_params": None,
        }
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    def test_produces_eval_py_script(self) -> None:
        args = self._make_args()
        cmd = build_eval_cmd("2a", args)
        assert "scripts/eval.py" in cmd

    def test_includes_phase_flag(self) -> None:
        args = self._make_args()
        cmd = build_eval_cmd("2a", args)
        assert "--phase" in cmd
        assert "2a" in cmd

    def test_no_task_flag(self) -> None:
        args = self._make_args()
        cmd = build_eval_cmd("hover", args)
        assert "--task" not in cmd

    def test_headless_included(self) -> None:
        args = self._make_args()
        cmd = build_eval_cmd("2b", args)
        assert "--headless" in cmd

    def test_no_gnn_forwarded(self) -> None:
        args = self._make_args(no_gnn=True)
        cmd = build_eval_cmd("3", args)
        assert "--no-gnn" in cmd

    def test_checkpoint_forwarded(self) -> None:
        args = self._make_args(checkpoint="/tmp/ckpt.pt")
        cmd = build_eval_cmd("2a", args)
        assert "--checkpoint" in cmd
        assert "/tmp/ckpt.pt" in cmd

    def test_num_episodes_forwarded(self) -> None:
        args = self._make_args(num_episodes=10)
        cmd = build_eval_cmd("3", args)
        assert "--num_episodes" in cmd
        assert "10" in cmd

    def test_extra_flags_appended(self) -> None:
        args = self._make_args()
        cmd = build_eval_cmd("3", args, extra_flags=["--kill_agent_step", "50"])
        assert "--kill_agent_step" in cmd
        assert "50" in cmd

    def test_video_sets_default_rendering_quality(self) -> None:
        args = self._make_args(video=True, rendering_mode=None)
        cmd = build_eval_cmd("2a", args)
        assert "--video" in cmd
        assert "--rendering_mode" in cmd
        assert "quality" in cmd


# ---------------------------------------------------------------------------
# composite_collector
# ---------------------------------------------------------------------------

from ggswarm_utils.composite_collector import CompositeCollector  # noqa: E402


class _StubCollector:
    """Minimal PhaseCollector stub that records call order."""

    def __init__(self, name: str, metrics: dict | None = None) -> None:
        self.name = name
        self.metrics = metrics or {}
        self.on_step_calls: list[int] = []
        self.on_episode_end_calls: list[int] = []

    def on_step(self, *, base_env, obs, step, episode_step, episode_num) -> None:
        self.on_step_calls.append(step)

    def on_episode_end(self, episode_num: int) -> None:
        self.on_episode_end_calls.append(episode_num)

    def summarize(self) -> dict[str, float]:
        return self.metrics


class TestCompositeCollector:
    def test_on_step_delegates_to_all_collectors(self) -> None:
        primary = _StubCollector("p")
        extra = _StubCollector("e")
        cc = CompositeCollector(primary, extra)
        cc.on_step(base_env=None, obs=None, step=0, episode_step=0, episode_num=0)
        assert len(primary.on_step_calls) == 1
        assert len(extra.on_step_calls) == 1

    def test_on_step_calls_primary_first(self) -> None:
        order: list[str] = []

        class OrderedStub(_StubCollector):
            def on_step(self, **kw) -> None:
                order.append(self.name)

        p = OrderedStub("primary")
        e = OrderedStub("extra")
        cc = CompositeCollector(p, e)
        cc.on_step(base_env=None, obs=None, step=0, episode_step=0, episode_num=0)
        assert order == ["primary", "extra"]

    def test_on_episode_end_delegates_to_all(self) -> None:
        primary = _StubCollector("p")
        extra = _StubCollector("e")
        cc = CompositeCollector(primary, extra)
        cc.on_episode_end(0)
        assert len(primary.on_episode_end_calls) == 1
        assert len(extra.on_episode_end_calls) == 1

    def test_summarize_returns_primary_metrics(self) -> None:
        primary = _StubCollector("p", {"survival_steps": 100.0})
        extra = _StubCollector("e", {"other": 42.0})
        cc = CompositeCollector(primary, extra)
        assert cc.summarize() == {"survival_steps": 100.0}

    def test_summarize_ignores_extra_collectors(self) -> None:
        primary = _StubCollector("p", {})
        extra = _StubCollector("e", {"should_not_appear": 1.0})
        cc = CompositeCollector(primary, extra)
        assert "should_not_appear" not in cc.summarize()


# ---------------------------------------------------------------------------
# collector_factory
# ---------------------------------------------------------------------------

from ggswarm_utils.collector_factory import make_collector  # noqa: E402


class TestCollectorFactory:
    def test_make_collector_resolves_phase2_collector(self) -> None:
        from ggswarm_utils.sim_helpers import PHASE_REGISTRY

        phase_cfg = PHASE_REGISTRY["2"]
        collector = make_collector(phase_cfg)
        from ggswarm_utils.phases.phase2 import Phase2Collector

        assert isinstance(collector, Phase2Collector)

    def test_make_collector_passes_valid_kwargs(self) -> None:
        from ggswarm_utils.sim_helpers import PHASE_REGISTRY

        phase_cfg = PHASE_REGISTRY["2"]
        collector = make_collector(phase_cfg, airborne_height_margin=0.5)
        assert collector._airborne_margin == 0.5

    def test_make_collector_filters_unknown_kwargs(self) -> None:
        from ggswarm_utils.sim_helpers import PHASE_REGISTRY

        phase_cfg = PHASE_REGISTRY["2"]
        # Should not raise TypeError for unknown param
        collector = make_collector(phase_cfg, bogus_param=True)
        assert collector is not None


# ---------------------------------------------------------------------------
# trajectory_collector
# ---------------------------------------------------------------------------

from ggswarm_utils.trajectory_collector import TrajectoryDataCollector  # noqa: E402


class TestTrajectoryDataCollector:
    def _make_env(self, num_envs: int = 1, num_agents: int = 2) -> object:
        from types import SimpleNamespace

        pos_flat = torch.zeros(num_envs * num_agents, 3)
        quat_flat = torch.zeros(num_envs * num_agents, 4)
        quat_flat[:, 0] = 1.0  # identity quaternion (w, x, y, z)

        class RobotData:
            root_pos_w = pos_flat
            root_quat_w = quat_flat

        class Robot:
            data = RobotData()

        return SimpleNamespace(
            num_envs=num_envs,
            cfg=SimpleNamespace(num_agents=num_agents),
            robot=Robot(),
        )

    def test_on_step_accumulates_positions(self) -> None:
        env = self._make_env()
        env.robot.data.root_pos_w[0, 2] = 1.0  # set z
        col = TrajectoryDataCollector(track_envs=1)
        col.on_step(base_env=env, obs=None, step=0, episode_step=0, episode_num=0)
        col.on_step(base_env=env, obs=None, step=1, episode_step=1, episode_num=0)
        assert len(col.pos_traj[0]) == 2

    def test_on_step_accumulates_attitude_when_euler_fn_provided(self) -> None:
        env = self._make_env()

        def fake_euler(quat):
            n = quat.shape[0]
            return torch.zeros(n), torch.zeros(n), torch.zeros(n)

        col = TrajectoryDataCollector(track_envs=1, euler_fn=fake_euler)
        col.on_step(base_env=env, obs=None, step=0, episode_step=0, episode_num=0)
        assert len(col.roll_traj[0]) == 1
        assert len(col.pitch_traj[0]) == 1

    def test_on_step_skips_attitude_when_no_euler_fn(self) -> None:
        env = self._make_env()
        col = TrajectoryDataCollector(track_envs=1, euler_fn=None)
        col.on_step(base_env=env, obs=None, step=0, episode_step=0, episode_num=0)
        assert len(col.roll_traj[0]) == 0

    def test_track_envs_limits_stored_data(self) -> None:
        env = self._make_env(num_envs=4, num_agents=2)
        col = TrajectoryDataCollector(track_envs=2)
        col.on_step(base_env=env, obs=None, step=0, episode_step=0, episode_num=0)
        # stored tensor should have shape [2, 2, 3] (track_envs=2, agents=2, xyz)
        assert col.pos_traj[0][0].shape[0] == 2

    def test_new_episode_creates_new_key(self) -> None:
        env = self._make_env()
        col = TrajectoryDataCollector(track_envs=1)
        col.on_step(base_env=env, obs=None, step=0, episode_step=0, episode_num=0)
        col.on_step(base_env=env, obs=None, step=1, episode_step=0, episode_num=1)
        assert 0 in col.pos_traj
        assert 1 in col.pos_traj

    def test_summarize_returns_empty_dict(self) -> None:
        col = TrajectoryDataCollector()
        assert col.summarize() == {}

    def test_on_episode_end_is_noop(self) -> None:
        col = TrajectoryDataCollector()
        col.on_episode_end(0)  # should not raise


# ---------------------------------------------------------------------------
# trajectory_plots
# ---------------------------------------------------------------------------

from ggswarm_utils.trajectory_plots import (  # noqa: E402
    generate_all_trajectory_plots,
    plot_altitude,
    plot_attitude,
    plot_xy,
)


class TestTrajectoryPlots:
    def _make_traj_steps(self, T: int = 20, envs: int = 1, agents: int = 2):
        """Synthetic trajectory: list of [envs, agents, 3] tensors."""
        return [torch.randn(envs, agents, 3) for _ in range(T)]

    def _make_attitude_steps(self, T: int = 20, envs: int = 1, agents: int = 2):
        """Synthetic attitude: list of [envs, agents] tensors (degrees)."""
        return [torch.randn(envs, agents) * 10 for _ in range(T)]

    def test_plot_altitude_creates_png(self, tmp_path: Path) -> None:
        out = plot_altitude(
            self._make_traj_steps(), ep=0, out_dir=tmp_path,
            min_height=0.3, max_height=2.5, spawn_z_min=0.5, spawn_z_max=1.0,
        )
        assert out.exists()
        assert out.suffix == ".png"

    def test_plot_xy_creates_png(self, tmp_path: Path) -> None:
        out = plot_xy(self._make_traj_steps(), ep=0, out_dir=tmp_path)
        assert out.exists()
        assert out.suffix == ".png"

    def test_plot_attitude_creates_png(self, tmp_path: Path) -> None:
        out = plot_attitude(
            self._make_attitude_steps(), self._make_attitude_steps(),
            ep=0, out_dir=tmp_path,
        )
        assert out.exists()
        assert out.suffix == ".png"

    def test_generate_all_creates_all_plot_types(self, tmp_path: Path) -> None:
        col = TrajectoryDataCollector(track_envs=1)
        # Manually populate data
        col.pos_traj[0] = self._make_traj_steps()
        col.roll_traj[0] = self._make_attitude_steps()
        col.pitch_traj[0] = self._make_attitude_steps()
        created = generate_all_trajectory_plots(col, tmp_path)
        # altitude + xy + distance + attitude = 4 plots
        assert len(created) == 4
        names = {p.name for p in created}
        assert "altitude_trace_ep0.png" in names
        assert "xy_trace_ep0.png" in names
        assert "distance_trace_ep0.png" in names
        assert "attitude_trace_ep0.png" in names

    def test_generate_all_skips_attitude_when_no_roll_data(self, tmp_path: Path) -> None:
        col = TrajectoryDataCollector(track_envs=1)
        col.pos_traj[0] = self._make_traj_steps()
        col.roll_traj[0] = []
        col.pitch_traj[0] = []
        created = generate_all_trajectory_plots(col, tmp_path)
        # altitude + xy + distance = 3 (no attitude)
        assert len(created) == 3


# ---------------------------------------------------------------------------
# build_assess_cmd
# ---------------------------------------------------------------------------


class TestBuildAssessCmd:
    def _make_args(self, **overrides) -> argparse.Namespace:
        defaults = dict(
            run_dir="logs/skrl/ggswarm_marl/run1",
            num_episodes=5,
            no_sync=False,
            trajectories=False,
            device=None,
            seed=None,
            video=False,
            video_length=None,
            video_codec=None,
            video_bitrate=None,
            video_preset=None,
            video_ffmpeg_params=None,
            rendering_mode=None,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    def test_build_assess_cmd_basic(self) -> None:
        args = self._make_args()
        cmd = build_assess_cmd("scripts/post_train_assess.py", "MyTask", args)
        assert "scripts/post_train_assess.py" in cmd
        assert "--task" in cmd
        assert "MyTask" in cmd
        assert "--headless" in cmd

    def test_build_assess_cmd_trajectories_flag(self) -> None:
        args = self._make_args(trajectories=True)
        cmd = build_assess_cmd("scripts/post_train_assess.py", "MyTask", args)
        assert "--trajectories" in cmd

    def test_build_assess_cmd_no_trajectories_by_default(self) -> None:
        args = self._make_args()
        cmd = build_assess_cmd("scripts/post_train_assess.py", "MyTask", args)
        assert "--trajectories" not in cmd

    def test_build_assess_cmd_no_sync(self) -> None:
        args = self._make_args(no_sync=True)
        cmd = build_assess_cmd("scripts/post_train_assess.py", "MyTask", args)
        assert "--no_sync" in cmd

    def test_build_assess_cmd_video_flags(self) -> None:
        args = self._make_args(video=True)
        cmd = build_assess_cmd("scripts/post_train_assess.py", "MyTask", args)
        assert "--video" in cmd


# ---------------------------------------------------------------------------
# scorecard Unicode
# ---------------------------------------------------------------------------

from ggswarm_utils.scorecard import format_tb_diagnostics  # noqa: E402


class TestScorecardUnicode:
    def test_format_tb_diagnostics_ascii_only(self) -> None:
        scalars = [
            {
                "tag": "Info / rew_pos",
                "first_value": 0.1,
                "last_value": 0.3,
                "first_step": 1000,
                "last_step": 92000,
            }
        ]
        lines = format_tb_diagnostics(scalars)
        text = "\n".join(lines)
        # Must be encodable as ASCII (no Unicode arrows)
        text.encode("ascii")

    def test_format_tb_diagnostics_uses_arrow_ascii(self) -> None:
        scalars = [
            {
                "tag": "test",
                "first_value": 0.0,
                "last_value": 1.0,
                "first_step": 1000,
                "last_step": 2000,
            }
        ]
        lines = format_tb_diagnostics(scalars)
        text = "\n".join(lines)
        assert "->" in text
        assert "\u2192" not in text  # no Unicode arrow
