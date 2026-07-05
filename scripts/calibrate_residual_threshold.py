"""Stage 4a: calibrate the localizer's pre-fit-residual fault threshold.

Runs the trained MAPPO policy in `ggswarm-marl-v0` with localization on and
the fault-test/injection/dropout paths forced off, harvests the
`DecentralizedLocalizer` PRE-FIT innovation residual (see
`localization.py`'s class docstring and `ggswarm_marl_env.py::_update_localization`
for the per-tick call order: `update_residuals()` computes this residual
*before* any Gauss-Newton correction, which is exactly what `run_fault_test`
consumes at flag time) from alive drones during policy rollout, and prints
`residual_mu` / `residual_sigma` (plus a mu+3*sigma flag-threshold estimate
and the p99.9 sample) for the user to paste into `GgswarmMarlEnvCfg`.

Policy-in-the-loop, not a hover test: the residual is measured over the
policy's real maneuver envelope, not just steady hover, so a per-env warmup period
is discarded to let spawn transients and policy behavior settle first.

AppLauncher header, checkpoint loading, and env construction are mirrored
from `scripts/skrl/replay_gate.py`; only the residual-harvesting measurement
core below is new.
"""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Stage 4a residual-threshold calibration")
parser.add_argument("--checkpoint", required=True, help="Actor checkpoint path (capstone or MAPPO format)")
parser.add_argument("--episodes", type=int, default=50, help="Number of episodes to roll out")
parser.add_argument("--warmup_steps", type=int, default=50, help="Steps to discard per episode before harvesting")
parser.add_argument("--num_envs", type=int, default=1, help="Envs to run in parallel")
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


def build_env(num_envs: int):
    """Build the MARL env with localization on and all fault paths forced off."""
    env_cfg = parse_env_cfg("ggswarm-marl-v0", num_envs=num_envs, use_fabric=True)
    env_cfg.scene.num_envs = num_envs
    env_cfg.loc_enabled = True
    env_cfg.residual_test_enabled = False
    env_cfg.fault_inject_enabled = False
    env_cfg.dropout_enabled = False
    env = gym.make("ggswarm-marl-v0", cfg=env_cfg, render_mode=None)
    return SkrlVecEnvWrapper(env, ml_framework="torch"), env_cfg


def main() -> int:
    env, env_cfg = build_env(args.num_envs)
    device = env.device

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

    # Vectorized envs (IsaacLab DirectMARLEnv) auto-reset internally per-env
    # inside step() — individual envs terminate/reset on their own staggered
    # schedule, so a single reset here plus a single up-front warmup (rather
    # than per-episode reset bookkeeping) covers `args.episodes` worth of
    # steps while still discarding the initial spawn transient.
    steps_per_episode = int(env.unwrapped.max_episode_length)
    total_steps = args.episodes * steps_per_episode
    residual_samples: list[torch.Tensor] = []  # script scope: allocations fine

    obs_dict, _ = env.reset()
    with torch.no_grad():
        for step_in_episode in range(total_steps):
            actions = {}
            for agent_id in env.possible_agents:
                o = obs_dict[agent_id]
                if state_preprocessor is not None:
                    o = state_preprocessor(o)
                mean_action, _, _ = actor.compute({"states": o}, role="policy")
                actions[agent_id] = mean_action.clamp(-1.0, 1.0)
            obs_dict, _, _, _, _ = env.step(actions)

            # After per-env warmup (transients from spawn + policy settle), harvest
            # residuals from alive drones only. Policy-in-the-loop =>
            # maneuver envelope, not hover.
            warm = env.unwrapped.episode_length_buf >= args.warmup_steps  # [N_envs] bool, per-env steps since ITS last reset
            if warm.any():
                loc = env.unwrapped._localizer
                alive = env.unwrapped._agent_alive.reshape(env.unwrapped.num_envs, -1)
                mask = warm.unsqueeze(1) & alive  # [N_envs, A]
                residual_samples.append(loc.residual[mask].flatten().cpu())

            if (step_in_episode + 1) % steps_per_episode == 0:
                print(f"  ~episode {(step_in_episode + 1) // steps_per_episode}/{args.episodes} complete")

    all_res = torch.cat(residual_samples)
    mu, sigma = all_res.mean().item(), all_res.std().item()
    p999 = all_res.quantile(0.999).item()
    print(f"samples          : {all_res.numel()}")
    print(f"residual_mu      = {mu:.4f}")
    print(f"residual_sigma   = {sigma:.4f}")
    print(f"mu + 3*sigma     = {mu + 3 * sigma:.4f}   (flag threshold)")
    print(f"p99.9 residual   = {p999:.4f}   (sanity: should sit near the threshold)")
    print("Paste residual_mu / residual_sigma into GgswarmMarlEnvCfg.")

    env.close()
    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
