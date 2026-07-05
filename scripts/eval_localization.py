"""Stage 4b: localization scorecard evaluation (Loc-b gate).

Runs the trained MAPPO policy in `ggswarm-marl-v0` with localization on and
the fault-test/recovery paths enabled (see `localization.py`'s class
docstring and `ggswarm_marl_env.py::_update_localization` for the per-tick
call order: `propagate -> measure -> update_residuals -> run_fault_test ->
correct -> recover`), and prints the Loc-b gate scorecard: steady-state RMSE,
false-positive flag rate, false-negative rate, recovery containment /
detection latency, and formation-collapse count.

Two modes:
  --mode honest  fault_inject_enabled=False. Every drone-step is "never
                 faulted"; measures RMSE + FP rate under normal operation.
  --mode fault   fault_inject_enabled=True. `_fault_mask` designates victim
                 drone(s) per env (`ggswarm_marl_env.py::_reset_idx`); a
                 persistent ranging bias triggers at that env's `_fault_step`
                 (see `ranging.py::inject_fault` docstring: "persistent range
                 bias until reset_idx" — it does NOT self-clear, so a flagged
                 drone dead-reckons for the rest of the episode rather than
                 snapping back to truth once `correct()` gates its links to
                 zero weight).

Recovery is defined two ways because a persistent-bias fault does not let a
drone's estimate "recover" back onto truth the way a transient fault would:
  - containment (primary, matches Task 4's validated gates): the faulted
    drone's estimate error never exceeds 0.20 m from fault onset through
    episode end.
  - time-to-flag: steps from fault onset to the first `run_fault_test` flag
    on that drone, converted to seconds via `step_dt` (0.02 s/step).
Both are reported so the changelog can quote whichever the user prefers.
FN = a faulted drone is never flagged between its fault onset and episode
end (whether or not the fault ever actually onset before an early episode
termination is separately tracked so those non-trials don't inflate the FN
rate). The first (staggered, partial) episode per env is excluded from ALL
fault-trial accounting: the full reset staggers `episode_length_buf` to
randint(0, max_episode_length) (`_reset_idx`) while the bias is injected only
on the EXACT match `episode_length_buf == _fault_step`, so an env whose
staggered start already exceeds `_fault_step` never physically received the
bias that episode and a `>=` onset test there would be spurious. Fault
bookkeeping for an env starts only after its first observed buf wrap — from
the second episode on, buf starts at 0 and `>=` is equivalent to having
passed through the exact-match injection step while the mask was armed. The
printed `episodes` count in fault mode counts only these accounted episodes.
Honest-mode metrics (RMSE/FP) are unaffected and count from the start (they
already use per-env warmup gating and don't depend on fault onset).

RMSE and FP rate are measured from alive, never-faulted, post-warmup
drone-steps in BOTH modes (a fault-mode run still has 7 of 8 honest peers
each episode) — this is the "(steady, honest)" / "(honest)" label on those
two scorecard lines.

AppLauncher header, checkpoint loading, and env construction are mirrored
from `scripts/calibrate_residual_threshold.py` (Task 7); the per-env
`episode_length_buf >= warmup` gating pattern is reused verbatim for the
RMSE/FP mask.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage 4b localization scorecard evaluation")
parser.add_argument("--checkpoint", required=True, help="Actor checkpoint path (capstone or MAPPO format)")
parser.add_argument("--mode", required=True, choices=["honest", "fault"], help="Evaluation mode")
parser.add_argument("--episodes", type=int, default=100, help="Number of episodes to roll out")
parser.add_argument("--warmup_steps", type=int, default=50, help="Steps to discard per episode before harvesting")
parser.add_argument("--num_envs", type=int, default=1, help="Envs to run in parallel")
parser.add_argument(
    "--residual_mu",
    type=float,
    required=True,
    help="Pre-fit residual mean from Task 7's calibrate_residual_threshold.py output (residual_mu)",
)
parser.add_argument(
    "--residual_sigma",
    type=float,
    required=True,
    help="Pre-fit residual std from Task 7's calibrate_residual_threshold.py output (residual_sigma)",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import ggswarm.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from ggswarm.checkpoint_utils import init_state_preprocessor, load_actor_weights  # noqa: E402
from ggswarm.gnn_policy import GgswarmGNNPolicy  # noqa: E402

from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402

from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

NUM_AGENTS = 8
CONTAINMENT_THRESHOLD_M = 0.20  # brief's recovery gate: error must never exceed this from onset


def build_env(num_envs: int, mode: str, residual_mu: float, residual_sigma: float):
    """Build the MARL env with localization + fault-test/recovery on, fault injection gated by mode."""
    env_cfg = parse_env_cfg("ggswarm-marl-v0", num_envs=num_envs, use_fabric=True)
    env_cfg.scene.num_envs = num_envs
    env_cfg.loc_enabled = True
    env_cfg.residual_test_enabled = True
    env_cfg.recovery_enabled = True
    env_cfg.fault_inject_enabled = mode == "fault"
    env_cfg.dropout_enabled = False
    env_cfg.residual_mu = residual_mu
    env_cfg.residual_sigma = residual_sigma
    env = gym.make("ggswarm-marl-v0", cfg=env_cfg, render_mode=None)
    return SkrlVecEnvWrapper(env, ml_framework="torch"), env_cfg


def main() -> int:
    env, env_cfg = build_env(args.num_envs, args.mode, args.residual_mu, args.residual_sigma)
    device = env.device
    N_envs = env.num_envs
    A = env_cfg.num_agents

    agent0 = env.possible_agents[0]
    actor = GgswarmGNNPolicy(
        observation_space=env.observation_space(agent0),
        action_space=env.action_space(agent0),
        device=device,
        num_neighbors=env_cfg.num_neighbors,
        num_agents=NUM_AGENTS,
    )
    GgswarmGNNPolicy.init_edge_cache(memory_size=1, num_envs=env.num_envs * NUM_AGENTS)

    sp_stats = load_actor_weights(args.checkpoint, actor, device)
    actor.to(device)  # force move all params to device after state_dict load
    actor.train(False)
    state_preprocessor = init_state_preprocessor(sp_stats, actor.observation_space, device)

    u = env.unwrapped

    # --- honest-mode metrics (RMSE / FP): accumulated in BOTH modes, from
    # alive, never-faulted, post-warmup drone-steps.
    sq_err_sum = 0.0
    sq_err_count = 0
    fp_flagged = 0
    fp_total = 0

    # --- fault-mode metrics: per-(env, drone) scratch, refreshed at each
    # env's episode boundary (buf wrap). Only meaningful when mode=="fault";
    # allocated unconditionally (cheap, [N_envs, A]) to keep the step loop
    # branch-free.
    first_flag_step = torch.full((N_envs, A), -1, dtype=torch.long, device=device)
    max_err_onset = torch.zeros(N_envs, A, device=device)
    onset_ever = torch.zeros(N_envs, A, dtype=torch.bool, device=device)
    fault_step_for_accum = u._fault_step.clone() if args.mode == "fault" else torch.zeros(N_envs, dtype=torch.long, device=device)
    fault_mask_for_accum = u._fault_mask.clone() if args.mode == "fault" else torch.zeros(N_envs, A, dtype=torch.bool, device=device)
    # Guard against the staggered first episode (see module docstring): the
    # full reset staggers episode_length_buf, but the bias injects only on the
    # exact buf == _fault_step match, so fault accounting for an env is armed
    # only after its first observed buf wrap.
    seen_first_reset = torch.zeros(N_envs, dtype=torch.bool, device=device)

    faulted_trials = 0  # denominator: faulted-drone-episodes where onset actually occurred
    fn_count = 0
    contained_count = 0
    latency_samples: list[int] = []

    n_episodes = 0
    collapses = 0

    steps_per_episode = int(u.max_episode_length)
    total_steps = args.episodes * steps_per_episode

    obs_dict, _ = env.reset()
    with torch.no_grad():
        prev_buf = u.episode_length_buf.clone()
        for step_in_episode in range(total_steps):
            actions = {}
            for agent_id in env.possible_agents:
                o = obs_dict[agent_id]
                if state_preprocessor is not None:
                    o = state_preprocessor(o)
                mean_action, _, _ = actor.compute({"states": o}, role="policy")
                actions[agent_id] = mean_action.clamp(-1.0, 1.0)
            obs_dict, _, _, _, _ = env.step(actions)

            loc = u._localizer
            pos_true_g = (u._robot.data.root_pos_w - u._env_origins_per_drone).reshape(N_envs, A, 3)
            err = (loc.p_hat - pos_true_g).norm(dim=2)  # [N_envs, A]
            flags = loc.flags
            alive_g = u._agent_alive.reshape(N_envs, A)
            faulted = u._fault_mask  # [N_envs, A] (all-False in honest mode)
            warm = u.episode_length_buf >= args.warmup_steps  # [N_envs]

            # --- RMSE / FP (alive, never-faulted, post-warmup)
            honest_mask = alive_g & (~faulted) & warm.unsqueeze(1)  # [N_envs, A]
            if honest_mask.any():
                sq_err_sum += err[honest_mask].square().sum().item()
                sq_err_count += int(honest_mask.sum().item())
                fp_flagged += int(flags[honest_mask].sum().item())
                fp_total += int(honest_mask.sum().item())

            # --- fault-mode onset/flag/containment tracking (no-op in honest
            # mode: fault_mask_for_accum is all-False, so `started` is always
            # all-False and nothing accumulates).
            new_buf = u.episode_length_buf
            started = (
                (new_buf.unsqueeze(1) >= fault_step_for_accum.unsqueeze(1))
                & fault_mask_for_accum
                & seen_first_reset.unsqueeze(1)  # staggered first episode excluded
            )
            onset_ever |= started
            first_flag_step = torch.where(flags & started & (first_flag_step < 0), new_buf.unsqueeze(1), first_flag_step)
            max_err_onset = torch.where(started, torch.maximum(max_err_onset, err), max_err_onset)

            # --- episode boundary detection (per-env buf wrap), mirrors
            # Task 7's continuous-rollout style but per-env rather than a
            # single up-front warmup, since we need to close out faulted-
            # drone trials and formation-collapse counts per episode.
            reset_envs = (new_buf < prev_buf).nonzero(as_tuple=True)[0]
            if reset_envs.numel() > 0:
                # Fault mode counts only accounted episodes: wraps of envs
                # whose staggered first (partial) episode already ended.
                if args.mode == "fault":
                    accounted_envs = reset_envs[seen_first_reset[reset_envs]]
                    n_episodes += int(accounted_envs.numel())
                else:
                    accounted_envs = reset_envs
                    n_episodes += int(reset_envs.numel())

                log0 = u.extras[u._agent_ids[0]].get("log", {})
                collapses += int(log0.get("Episode_Termination/died", 0))

                if args.mode == "fault":
                    for e in accounted_envs.tolist():
                        victims = fault_mask_for_accum[e].nonzero(as_tuple=True)[0].tolist()
                        for a in victims:
                            if not bool(onset_ever[e, a]):
                                continue  # fault never onset this episode (early termination) — not a trial
                            faulted_trials += 1
                            ffs = int(first_flag_step[e, a].item())
                            if ffs >= 0:
                                latency_samples.append(max(ffs - int(fault_step_for_accum[e].item()), 0))
                            else:
                                fn_count += 1
                            if float(max_err_onset[e, a].item()) <= CONTAINMENT_THRESHOLD_M:
                                contained_count += 1

                    # refresh per-env scratch for the new episode that just started
                    first_flag_step[reset_envs, :] = -1
                    max_err_onset[reset_envs, :] = 0.0
                    onset_ever[reset_envs, :] = False
                    fault_step_for_accum[reset_envs] = u._fault_step[reset_envs]
                    fault_mask_for_accum[reset_envs] = u._fault_mask[reset_envs]

                # Env has now completed its staggered first episode; fault
                # accounting is armed for it from here on.
                seen_first_reset[reset_envs] = True

            prev_buf = new_buf.clone()

            if n_episodes >= args.episodes:
                break

    rmse = (sq_err_sum / sq_err_count) ** 0.5 if sq_err_count > 0 else float("nan")
    fp_rate = fp_flagged / fp_total if fp_total > 0 else float("nan")

    if args.mode == "fault" and faulted_trials > 0:
        fn_rate = fn_count / faulted_trials
        containment_rate = contained_count / faulted_trials
        if latency_samples:
            lat = torch.tensor(latency_samples, dtype=torch.float32)
            rec_p50 = lat.quantile(0.5).item() * u.step_dt
            rec_p95 = lat.quantile(0.95).item() * u.step_dt
        else:
            rec_p50 = rec_p95 = float("nan")
    else:
        fn_rate = float("nan")
        containment_rate = float("nan")
        rec_p50 = rec_p95 = float("nan")

    print(f"mode                     : {args.mode}")
    print(f"episodes                 : {n_episodes}")
    print(f"loc RMSE (steady, honest): {rmse:.4f} m      (gate <= 0.10)")
    print(f"FP rate (honest)         : {fp_rate:.4f}     (gate <= 0.01)")
    print(f"FN rate (fault)          : {fn_rate:.4f}     (gate <= 0.05)")
    print(
        f"recovery containment/lat : {containment_rate:.4f} contained, "
        f"p50/p95 time-to-flag {rec_p50:.2f}s / {rec_p95:.2f}s (gate <= 1.0 s)"
    )
    print(f"formation collapses      : {collapses} / {n_episodes} episodes (gate 0)")

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
