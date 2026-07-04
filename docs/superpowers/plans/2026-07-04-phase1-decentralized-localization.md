# Phase 1 Decentralized Localization (Stages 0–4, shadow mode) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the simulated UWB ranging + per-drone localization stack and run it in shadow mode inside `GgswarmMarlEnv` (estimator runs and logs; observations stay ground truth), per the spec `docs/superpowers/specs/2026-07-04-phase1-decentralized-localization-design.md`.

**Architecture:** Two new pure-torch modules (`ranging.py`, `localization.py`) with no Isaac imports, composed by the env behind one `_update_localization()` call at the top of `_get_observations`. Odometry-anchored gauge: seed estimates from spawn poses, dead-reckon on noisy velocity, correct with damped Gauss-Newton against noisy peer ranges and latency-delayed broadcast estimates.

**Tech Stack:** PyTorch (pure-torch modules + pytest), Isaac Lab `DirectMARLEnv` (integration), SKRL replay tooling (`scripts/skrl/replay_gate.py`).

## Global Constraints

- **Per-step allocation ban** (repo CLAUDE.md): no `torch.zeros/ones/empty/full` or `.clone()` inside `_pre_physics_step`, `_apply_action`, `_get_observations`, `_get_rewards`, `_get_dones` — and therefore none inside `measure`/`propagate`/`correct`/`run_fault_test`/`recover` (they run from `_get_observations`). Preallocate in `__init__`; RNG via in-place `.normal_()`/`.bernoulli_()` (`torch.randn_like` is banned). Functional ops (`cdist`, `topk`, arithmetic) are fine — the existing `_get_observations` uses them.
- **Shape comments** on first access of any drone tensor: `# shape: [N_envs, A, 3]` style.
- **All tunables in cfg** — no magic numbers in env core.
- **No `print()` in per-step env code**; `print()` OK in `scripts/` and tests.
- **Frozen trees:** do not touch `source/.../ggswarm_env.py` (legacy single-agent), `docs/capstone/**`.
- **Machines:** pure-torch pytest tasks run anywhere with torch (this Linux box or the Windows venv `env_isaaclab/Scripts/python.exe -m pytest`). Isaac-dependent steps (smoke train, replay gate, calibration) run **on the Windows machine with the local 3070** — mark them and hand off; do not attempt Isaac launches on the Linux server.
- **Windows shell rules** for commands the user runs there: no `&&` chaining (use `;`), no `head` (use `Select-Object -First N`).
- Key constants for tests: `step_dt = 0.02` (sim dt 1/100 × decimation 2), `A = 8`, obs 18D, anchor checkpoint `logs/ref/v1.0.0-capstone/best_agent.pt`.
- Commit after every task; commit messages end with `Co-Authored-By:` line per harness rules.

---

### Task 1: DropoutGuard rename (Stage 0)

**Files:**
- Modify: `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_marl_env.py` (comments at lines 102, 292, 732)
- Modify: `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_marl_env_cfg.py` (comment at line 173)
- Modify: `scripts/skrl/play.py`, `scripts/eval_metrics.py` (whatever `grep -n` finds)

**Interfaces:** none (comments/strings only; zero behavior change).

- [ ] **Step 1: Find every active-path occurrence**

Run: `grep -rn "SwarmRaft" source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_marl_env.py source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_marl_env_cfg.py scripts/`
Expected: hits in `ggswarm_marl_env.py` (3), `ggswarm_marl_env_cfg.py` (1), `scripts/skrl/play.py` (~1), `scripts/eval_metrics.py` (~1–4). Do NOT touch `ggswarm_env.py` or anything under `docs/capstone/`.

- [ ] **Step 2: Replace the term in each hit**

Exact known replacements (comments only):

```python
# ggswarm_marl_env.py:102  — was: # SwarmRaft state (per-drone alive mask, per-env dropout step)
# DropoutGuard state (per-drone alive mask, per-env dropout step)

# ggswarm_marl_env.py:292  — was: # SwarmRaft: trigger agent dropout at scheduled per-env step
# DropoutGuard: trigger agent dropout at scheduled per-env step

# ggswarm_marl_env.py:732  — was: # SwarmRaft: per-env dropout step
# DropoutGuard: per-env dropout step

# ggswarm_marl_env_cfg.py:173  — was: # SwarmRaft agent dropout
# DropoutGuard agent dropout (renamed from "SwarmRaft" — see decentralization_plan.md §2)
```

For the `scripts/` hits, apply the same word substitution (`SwarmRaft` → `DropoutGuard`) in comments/strings; if a hit is a user-facing printed label, keep the rest of the label text unchanged.

- [ ] **Step 3: Verify grep-clean and no code change**

Run: `grep -rn "SwarmRaft" source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_marl_env.py source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_marl_env_cfg.py scripts/`
Expected: no output.
Run: `git diff --stat` — only the four files, comment-line counts only.

- [ ] **Step 4: [Windows] 5-iteration smoke train**

```text
env_isaaclab/Scripts/python.exe scripts/skrl/train.py --headless --task ggswarm-marl-v0 --num_envs 64 --max_iterations 5
```

Expected: trains 5 iterations, no errors, task `ggswarm-marl-v0` loads.

- [ ] **Step 5: Commit**

```bash
git add source/ggswarm/ggswarm/tasks/direct/ggswarm/ scripts/
git commit -m "refactor: rename SwarmRaft dropout mechanism to DropoutGuard (Stage 0)"
```

---

### Task 2: `UwbRangingSim` + tests (Stage 1)

**Files:**
- Create: `source/ggswarm/ggswarm/ranging.py`
- Create: `tests/__init__.py` (empty), `tests/test_ranging.py`
- Create: `tests/conftest.py`

**Interfaces:**
- Produces (Tasks 3, 4, 6 rely on these exact signatures):
  - `UwbRangingSim(num_envs: int, num_agents: int, device, *, noise_std: float, bias: float, dropout_prob: float, latency_steps: int)`
  - `.measure(pos_g: Tensor[E, A, 3]) -> tuple[Tensor[E, A, A], BoolTensor[E, A, A]]` — (held ranges, validity of the delayed reading). Diagonal always invalid. Held = last valid reading per link, so ranges are always finite.
  - `.inject_fault(fault_mask: BoolTensor[E, A], bias_m: float) -> None` — persistent additive range bias on every link touching a faulted drone, until `reset_idx`.
  - `.reset_idx(env_ids: LongTensor, pos_g: Tensor[n_reset, A, 3]) -> None` — seeds ring/held with honest ranges at spawn, clears faults.

- [ ] **Step 1: Write conftest + failing tests**

`tests/conftest.py`:

```python
"""Make the ggswarm package importable without an editable install."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "ggswarm"))
```

`tests/test_ranging.py`:

```python
"""Unit tests for UwbRangingSim (pure torch, no Isaac)."""
import torch

from ggswarm.ranging import UwbRangingSim

E, A = 64, 8
DEV = torch.device("cpu")


def make_sim(**kw):
    defaults = dict(noise_std=0.10, bias=0.05, dropout_prob=0.05, latency_steps=1)
    defaults.update(kw)
    return UwbRangingSim(E, A, DEV, **defaults)


def octagon(radius=1.0):
    theta = torch.arange(A) * (2 * torch.pi / A)
    pts = torch.stack([radius * theta.cos(), radius * theta.sin(), torch.ones(A)], dim=1)
    return pts.unsqueeze(0).expand(E, A, 3).contiguous()


def test_noise_statistics():
    torch.manual_seed(0)
    sim = make_sim(dropout_prob=0.0, latency_steps=0)
    pos = octagon()
    true_d = torch.cdist(pos, pos)
    errs = []
    for _ in range(200):
        ranges, valid = sim.measure(pos)
        offdiag = ~torch.eye(A, dtype=torch.bool).unsqueeze(0).expand(E, A, A)
        errs.append((ranges - true_d)[offdiag])
    err = torch.cat(errs)
    assert abs(err.mean().item() - 0.05) < 0.005          # mean ≈ bias
    assert abs(err.std().item() - 0.10) < 0.01            # std ≈ sigma


def test_symmetry_and_diagonal():
    torch.manual_seed(0)
    sim = make_sim(latency_steps=0)
    ranges, valid = sim.measure(octagon())
    assert torch.allclose(ranges, ranges.transpose(1, 2))
    assert (valid == valid.transpose(1, 2)).all()
    assert not valid.diagonal(dim1=1, dim2=2).any()


def test_dropout_rate():
    torch.manual_seed(0)
    sim = make_sim(dropout_prob=0.20, latency_steps=0)
    pos = octagon()
    rates = []
    for _ in range(200):
        _, valid = sim.measure(pos)
        offdiag = ~torch.eye(A, dtype=torch.bool).unsqueeze(0).expand(E, A, A)
        rates.append(1.0 - valid[offdiag].float().mean().item())
    assert abs(sum(rates) / len(rates) - 0.20) < 0.02


def test_latency_returns_t_minus_L():
    torch.manual_seed(0)
    L = 3
    sim = make_sim(noise_std=0.0, bias=0.0, dropout_prob=0.0, latency_steps=L)
    base = octagon()
    history = []
    for t in range(10):
        pos = base + 0.1 * t  # rigid translation leaves ranges identical -> scale instead
        pos = base * (1.0 + 0.1 * t)
        history.append(torch.cdist(pos, pos))
        ranges, valid = sim.measure(pos)
        if t >= L:
            assert torch.allclose(ranges, history[t - L], atol=1e-5), f"t={t}"
            assert valid[0, 0, 1]


def test_hold_last_valid_when_dropped():
    torch.manual_seed(0)
    sim = make_sim(noise_std=0.0, bias=0.0, dropout_prob=1.0, latency_steps=0)
    pos = octagon()
    sim.reset_idx(torch.arange(E), pos)          # seeds held with honest ranges
    ranges, valid = sim.measure(pos * 2.0)       # all links dropped this step
    assert not valid[0, 0, 1]
    true_seed = torch.cdist(pos, pos)
    assert torch.allclose(ranges, true_seed, atol=1e-5)  # held seed values returned


def test_fault_injection_and_reset():
    torch.manual_seed(0)
    sim = make_sim(noise_std=0.0, bias=0.0, dropout_prob=0.0, latency_steps=0)
    pos = octagon()
    mask = torch.zeros(E, A, dtype=torch.bool)
    mask[:, 3] = True
    sim.inject_fault(mask, 1.0)
    ranges, _ = sim.measure(pos)
    true_d = torch.cdist(pos, pos)
    assert torch.allclose(ranges[:, 3, 4] - true_d[:, 3, 4], torch.ones(E), atol=1e-5)
    assert torch.allclose(ranges[:, 0, 1], true_d[:, 0, 1], atol=1e-5)
    sim.reset_idx(torch.arange(E), pos)
    ranges, _ = sim.measure(pos)
    assert torch.allclose(ranges[:, 3, 4], true_d[:, 3, 4], atol=1e-5)


def test_no_allocations_after_warmup():
    sim = make_sim()
    pos = octagon()
    r1, v1 = sim.measure(pos)
    ptr_r = r1.data_ptr()
    r2, v2 = sim.measure(pos)
    assert r2.data_ptr() == ptr_r  # held buffer is reused, not reallocated
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_ranging.py -q` (or the Windows venv python if torch is missing here)
Expected: `ModuleNotFoundError: No module named 'ggswarm.ranging'` (collection error counts as failing).

- [ ] **Step 3: Implement `source/ggswarm/ggswarm/ranging.py`**

```python
"""Simulated UWB peer-ranging channel. Pure torch — no Isaac imports.

Model per undirected link (i, j): d = ||x_i - x_j|| + bias + N(0, sigma),
symmetric per-link Bernoulli dropout, fixed latency via ring buffer, and a
persistent per-drone fault bias (inject_fault) for FN/recovery evaluation.
Noise defaults calibrated to Crazyflie LPS/TWR literature (see
docs/ggswarm_live/decentralization_plan.md §3).

All buffers are preallocated in __init__ and mutated in place — measure() is
called from the env step path, which bans explicit tensor construction.
"""

from __future__ import annotations

import torch


class UwbRangingSim:
    """Vectorized UWB ranging simulator over [num_envs, num_agents] drones."""

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        device: torch.device,
        *,
        noise_std: float,
        bias: float,
        dropout_prob: float,
        latency_steps: int,
    ) -> None:
        E, A = num_envs, num_agents
        L = max(0, int(latency_steps))
        self._E, self._A, self._L = E, A, L
        self.noise_std = float(noise_std)
        self.bias = float(bias)
        self.dropout_prob = float(dropout_prob)
        self._t = 0

        self._noise = torch.zeros(E, A, A, device=device)  # shape: [E, A, A]
        self._keep = torch.zeros(E, A, A, device=device)  # shape: [E, A, A]
        self._ring = torch.zeros(L + 1, E, A, A, device=device)  # shape: [L+1, E, A, A]
        self._ring_valid = torch.zeros(L + 1, E, A, A, dtype=torch.bool, device=device)
        self._held = torch.zeros(E, A, A, device=device)  # last valid range per link
        self._fault_bias = torch.zeros(E, A, device=device)  # shape: [E, A]
        self._eye = torch.eye(A, dtype=torch.bool, device=device).unsqueeze(0)  # [1, A, A]

    def measure(self, pos_g: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """pos_g: [E, A, 3] true env-local positions -> (ranges [E, A, A], valid [E, A, A])."""
        true_d = torch.cdist(pos_g, pos_g)  # shape: [E, A, A]

        # Symmetric Gaussian noise: draw full matrix, mirror the upper triangle.
        self._noise.normal_(0.0, self.noise_std)
        n_upper = torch.triu(self._noise, diagonal=1)
        sym_noise = n_upper + n_upper.transpose(1, 2)

        # Per-drone fault bias applies to every link touching a faulted drone.
        fb = self._fault_bias.unsqueeze(2) + self._fault_bias.unsqueeze(1)  # [E, A, A]

        ranges_now = true_d + self.bias + sym_noise + fb

        # Symmetric per-link Bernoulli dropout (upper triangle mirrored).
        self._keep.bernoulli_(1.0 - self.dropout_prob)
        k_upper = torch.triu(self._keep, diagonal=1)
        valid_now = (k_upper + k_upper.transpose(1, 2)) > 0.5
        valid_now = valid_now & ~self._eye

        # Ring buffer: write t, read t - L.
        w = self._t % (self._L + 1)
        r = (self._t - self._L) % (self._L + 1)
        self._ring[w].copy_(ranges_now)
        self._ring_valid[w].copy_(valid_now)
        ranges_d = self._ring[r]
        valid_d = self._ring_valid[r]
        self._t += 1

        # Hold-last-valid so returned ranges are always finite.
        self._held.copy_(torch.where(valid_d, ranges_d, self._held))
        return self._held, valid_d

    def inject_fault(self, fault_mask: torch.Tensor, bias_m: float) -> None:
        """fault_mask: [E, A] bool — persistent range bias until reset_idx."""
        self._fault_bias[fault_mask] = float(bias_m)

    def reset_idx(self, env_ids: torch.Tensor, pos_g: torch.Tensor) -> None:
        """Seed ring/held with honest spawn ranges; clear faults. pos_g: [n_reset, A, 3]."""
        true_d = torch.cdist(pos_g, pos_g) + self.bias  # honest reading at spawn
        self._ring[:, env_ids] = true_d.unsqueeze(0)
        self._ring_valid[:, env_ids] = ~self._eye
        self._held[env_ids] = true_d
        self._fault_bias[env_ids] = 0.0
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_ranging.py -q`
Expected: all 7 tests PASS. (If `test_latency_returns_t_minus_L` fails on the first `L` steps, the ring read of pre-seed zeros is leaking — check the `reset_idx` seeding and the read index math.)

- [ ] **Step 5: Commit**

```bash
git add source/ggswarm/ggswarm/ranging.py tests/
git commit -m "feat(loc): UwbRangingSim — simulated UWB peer ranging with noise, dropout, latency, faults (Stage 1)"
```

---

### Task 3: `DecentralizedLocalizer` — propagate / correct / diagnostics + tests (Stage 2a)

**Files:**
- Create: `source/ggswarm/ggswarm/localization.py`
- Create: `tests/test_localization.py`

**Interfaces:**
- Consumes: `UwbRangingSim.measure` output `(ranges [E, A, A], valid [E, A, A])` from Task 2.
- Produces (Tasks 4, 6 rely on these):
  - `DecentralizedLocalizer(num_envs, num_agents, device, *, correct_iters: int, damping: float, odom_noise_std: float, recovery_irls_iters: int, recovery_huber_delta: float, min_recovery_peers: int)`
  - `.p_hat: Tensor[E, A, 3]` (public estimate), `.flags: BoolTensor[E, A]`
  - `.propagate(lin_vel_w: Tensor[E*A, 3], dt: float, noise_scale: float = 1.0) -> None`
  - `.correct(ranges, valid, alive_g: BoolTensor[E, A]) -> None` — also refreshes `.residual: Tensor[E, A]` and re-publishes broadcasts.
  - `.rmse(pos_true_g: Tensor[E, A, 3], alive_g) -> Tensor[E]`, `.gauge_drift(pos_true_g, alive_g) -> Tensor[E]`
  - `.reset_idx(env_ids, pos_true_g: Tensor[n_reset, A, 3]) -> None`
  - (Task 4 adds `run_fault_test` / `recover`.)

- [ ] **Step 1: Write failing tests**

`tests/test_localization.py`:

```python
"""Synthetic-trajectory tests for DecentralizedLocalizer (pure torch, no Isaac)."""
import math

import torch

from ggswarm.localization import DecentralizedLocalizer
from ggswarm.ranging import UwbRangingSim

E, A, DT = 32, 8, 0.02
DEV = torch.device("cpu")


def make_stack(noise_std=0.10, bias=0.05, dropout=0.05, latency=1, odom_std=0.02):
    rng = UwbRangingSim(E, A, DEV, noise_std=noise_std, bias=bias,
                        dropout_prob=dropout, latency_steps=latency)
    loc = DecentralizedLocalizer(E, A, DEV, correct_iters=3, damping=0.5,
                                 odom_noise_std=odom_std, recovery_irls_iters=5,
                                 recovery_huber_delta=0.10, min_recovery_peers=4)
    return rng, loc


def octagon_traj(t):
    """Octagon of radius 1 whose centroid orbits slowly — pos [E, A, 3], vel [E*A, 3]."""
    theta = torch.arange(A) * (2 * math.pi / A)
    base = torch.stack([theta.cos(), theta.sin(), torch.ones(A)], dim=1)  # [A, 3]
    cx, cy = 0.5 * math.sin(0.2 * t), 0.5 * math.cos(0.2 * t)
    vx, vy = 0.5 * 0.2 * math.cos(0.2 * t), -0.5 * 0.2 * math.sin(0.2 * t)
    pos = base.unsqueeze(0).expand(E, A, 3).contiguous()
    pos = pos + torch.tensor([cx, cy, 0.0])
    vel = torch.tensor([vx, vy, 0.0]).expand(E * A, 3).contiguous()
    return pos, vel


def run_honest(steps=500, **kw):
    torch.manual_seed(0)
    rng, loc = make_stack(**kw)
    pos0, _ = octagon_traj(0.0)
    loc.reset_idx(torch.arange(E), pos0)
    rng.reset_idx(torch.arange(E), pos0)
    alive = torch.ones(E, A, dtype=torch.bool)
    for s in range(steps):
        pos, vel = octagon_traj(s * DT)
        loc.propagate(vel, DT)
        ranges, valid = rng.measure(pos)
        loc.correct(ranges, valid, alive)
    return rng, loc, pos, alive


def test_steady_state_rmse():
    _, loc, pos, alive = run_honest()
    rmse = loc.rmse(pos, alive)
    assert rmse.mean().item() <= 0.10, f"RMSE {rmse.mean():.3f} > 0.10 m"


def test_no_reflection_flips():
    _, loc, pos, alive = run_honest()
    per_drone_err = (loc.p_hat - pos).norm(dim=2)  # [E, A]
    assert per_drone_err.max().item() < 0.5  # a mirror flip would be ~2 m on an r=1 octagon


def test_gauge_drift_within_random_walk_envelope():
    _, loc, pos, alive = run_honest()
    drift = loc.gauge_drift(pos, alive)
    # Common-mode random walk: odom_std * dt * sqrt(steps / A) ~ 0.003 m; allow 20x slack.
    assert drift.mean().item() < 0.06


def test_dead_reckon_fallback_no_valid_links():
    torch.manual_seed(0)
    rng, loc = make_stack(dropout=1.0, latency=0, odom_std=0.0)
    pos0, _ = octagon_traj(0.0)
    loc.reset_idx(torch.arange(E), pos0)
    rng.reset_idx(torch.arange(E), pos0)
    alive = torch.ones(E, A, dtype=torch.bool)
    for s in range(50):
        pos, vel = octagon_traj(s * DT)
        loc.propagate(vel, DT)
        ranges, valid = rng.measure(pos)
        loc.correct(ranges, valid, alive)  # all links invalid -> pure dead reckoning
    assert loc.rmse(pos, alive).mean().item() < 0.02  # noiseless odom integrates cleanly


def test_nan_free_at_10x_noise():
    _, loc, pos, alive = run_honest(steps=300, noise_std=1.0, bias=0.5, odom_std=0.2)
    assert torch.isfinite(loc.p_hat).all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python3 -m pytest tests/test_localization.py -q`
Expected: collection error `No module named 'ggswarm.localization'`.

- [ ] **Step 3: Implement `source/ggswarm/ggswarm/localization.py`**

```python
"""Per-drone decentralized localization. Pure torch — no Isaac imports.

Odometry-anchored gauge (docs/ggswarm_live/decentralization_plan.md §3):
estimates seed from spawn truth, dead-reckon on noisy velocity odometry, and
each drone refines only its own estimate with damped Gauss-Newton steps
against peers' latency-delayed broadcast estimates and noisy UWB ranges.
No joint solve, no leader — the correct/test/recover math is the
companion-computer algorithm.

Called from the env step path: no explicit tensor construction here; scratch
buffers preallocated in __init__, RNG in-place.
"""

from __future__ import annotations

import torch


class DecentralizedLocalizer:
    """Vectorized per-drone estimator over [num_envs, num_agents] drones."""

    def __init__(
        self,
        num_envs: int,
        num_agents: int,
        device: torch.device,
        *,
        correct_iters: int,
        damping: float,
        odom_noise_std: float,
        recovery_irls_iters: int,
        recovery_huber_delta: float,
        min_recovery_peers: int,
    ) -> None:
        E, A = num_envs, num_agents
        self._E, self._A = E, A
        self.correct_iters = int(correct_iters)
        self.damping = float(damping)
        self.odom_noise_std = float(odom_noise_std)
        self.recovery_irls_iters = int(recovery_irls_iters)
        self.recovery_huber_delta = float(recovery_huber_delta)
        self.min_recovery_peers = int(min_recovery_peers)

        self.p_hat = torch.zeros(E, A, 3, device=device)  # shape: [E, A, 3]
        self._p_broadcast = torch.zeros(E, A, 3, device=device)  # peers' delayed view
        self._odom_noise = torch.zeros(E * A, 3, device=device)  # shape: [E*A, 3]
        self.residual = torch.zeros(E, A, device=device)  # shape: [E, A]
        self.flags = torch.zeros(E, A, dtype=torch.bool, device=device)

    # ------------------------------------------------------------ core steps

    def propagate(self, lin_vel_w: torch.Tensor, dt: float, noise_scale: float = 1.0) -> None:
        """lin_vel_w: [E*A, 3] world-frame velocity (sim-perfect odom before noise)."""
        self._odom_noise.normal_(0.0, self.odom_noise_std * noise_scale)
        self.p_hat += ((lin_vel_w + self._odom_noise) * dt).reshape(self._E, self._A, 3)

    def _link_weights(self, valid: torch.Tensor, alive_g: torch.Tensor) -> torch.Tensor:
        """Valid link AND both endpoints alive. -> float [E, A, A]."""
        both_alive = alive_g.unsqueeze(1) & alive_g.unsqueeze(2)
        return (valid & both_alive).float()

    def _gn_step(self, ranges: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """One scaled-gradient Gauss-Newton step for every drone at once.

        Each drone i minimizes sum_j w_ij (||p_i - b_j|| - d_ij)^2 over its own
        p_i, holding peer broadcasts b_j fixed. Returns step [E, A, 3].
        """
        diff = self.p_hat.unsqueeze(2) - self._p_broadcast.unsqueeze(1)  # [E, A, A, 3]
        dist = diff.norm(dim=3).clamp(min=1e-6)  # shape: [E, A, A]
        r = dist - ranges
        u = diff / dist.unsqueeze(3)  # unit vectors i -> j
        g = ((w * r).unsqueeze(3) * u).sum(dim=2)  # [E, A, 3]
        wsum = w.sum(dim=2, keepdim=True).clamp(min=1e-6)  # [E, A, 1]
        return g / wsum

    def correct(self, ranges: torch.Tensor, valid: torch.Tensor, alive_g: torch.Tensor) -> None:
        """Refine own estimates against delayed peer broadcasts; refresh residuals."""
        w = self._link_weights(valid, alive_g)
        for _ in range(self.correct_iters):
            self.p_hat -= self.damping * self._gn_step(ranges, w)

        # Per-drone residual: mean |range inconsistency| over usable links.
        diff = self.p_hat.unsqueeze(2) - self._p_broadcast.unsqueeze(1)
        r_abs = (diff.norm(dim=3) - ranges).abs() * w
        self.residual.copy_(r_abs.sum(dim=2) / w.sum(dim=2).clamp(min=1.0))

        # Publish for the next step's peers (1-step broadcast latency).
        self._p_broadcast.copy_(self.p_hat)

    # ---------------------------------------------------------- diagnostics

    def rmse(self, pos_true_g: torch.Tensor, alive_g: torch.Tensor) -> torch.Tensor:
        """Position RMSE vs truth over alive drones, per env -> [E]."""
        sq = (self.p_hat - pos_true_g).square().sum(dim=2) * alive_g.float()
        n = alive_g.float().sum(dim=1).clamp(min=1.0)
        return (sq.sum(dim=1) / n).sqrt()

    def gauge_drift(self, pos_true_g: torch.Tensor, alive_g: torch.Tensor) -> torch.Tensor:
        """Norm of the common-mode (mean) estimate error over alive drones -> [E]."""
        err = (self.p_hat - pos_true_g) * alive_g.unsqueeze(2).float()
        n = alive_g.float().sum(dim=1).clamp(min=1.0).unsqueeze(1)
        return (err.sum(dim=1) / n).norm(dim=1)

    def reset_idx(self, env_ids: torch.Tensor, pos_true_g: torch.Tensor) -> None:
        """Seed estimates from spawn truth (takeoff-layout datum). pos_true_g: [n_reset, A, 3]."""
        self.p_hat[env_ids] = pos_true_g
        self._p_broadcast[env_ids] = pos_true_g
        self.residual[env_ids] = 0.0
        self.flags[env_ids] = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python3 -m pytest tests/test_localization.py tests/test_ranging.py -q`
Expected: all PASS. If `test_steady_state_rmse` is marginally over 0.10, first check that the range **bias** (0.05 m) is being absorbed — with bias present the GN fixed point sits ~bias/2 off per link; if that alone breaks the gate, subtract `bias` inside `correct` via a `known_bias` constructor arg defaulting to the cfg bias (drones can calibrate a static bias on the ground — hardware-legal), and re-run.

- [ ] **Step 5: Commit**

```bash
git add source/ggswarm/ggswarm/localization.py tests/test_localization.py
git commit -m "feat(loc): DecentralizedLocalizer — dead-reckon + distributed GN correction (Stage 2a)"
```

---

### Task 4: Residual fault test + IRLS recovery + tests (Stage 2b)

**Files:**
- Modify: `source/ggswarm/ggswarm/localization.py` (add two methods)
- Modify: `tests/test_localization.py` (append tests)

**Interfaces:**
- Produces (Task 6 relies on):
  - `.run_fault_test(mu: float, sigma: float, k: float) -> BoolTensor[E, A]` — sets and returns `.flags` (`residual > mu + k*sigma`).
  - `.recover(ranges, valid, alive_g) -> None` — IRLS/Huber re-multilateration of flagged drones from non-flagged peers; skips drones with `< min_recovery_peers` usable peers (dead-reckon hold).

- [ ] **Step 1: Append failing tests to `tests/test_localization.py`**

```python
def run_with_fault(fault_drone=3, fault_bias=1.0, steps=300, fault_at=150,
                   mu=0.05, sigma=0.02, recover=True):
    torch.manual_seed(0)
    rng, loc = make_stack()
    pos0, _ = octagon_traj(0.0)
    loc.reset_idx(torch.arange(E), pos0)
    rng.reset_idx(torch.arange(E), pos0)
    alive = torch.ones(E, A, dtype=torch.bool)
    mask = torch.zeros(E, A, dtype=torch.bool)
    mask[:, fault_drone] = True
    flagged_at = None
    for s in range(steps):
        if s == fault_at:
            rng.inject_fault(mask, fault_bias)
        pos, vel = octagon_traj(s * DT)
        loc.propagate(vel, DT)
        ranges, valid = rng.measure(pos)
        loc.correct(ranges, valid, alive)
        loc.run_fault_test(mu, sigma, 3.0)
        if flagged_at is None and loc.flags[:, fault_drone].float().mean() > 0.5:
            flagged_at = s
        if recover:
            loc.recover(ranges, valid, alive)
    return loc, pos, alive, flagged_at


def test_fault_is_flagged():
    loc, pos, alive, flagged_at = run_with_fault(recover=False)
    assert flagged_at is not None and flagged_at - 150 <= 25  # flagged within 0.5 s


def test_false_positive_rate_honest():
    _, loc, pos, alive = run_honest(steps=300)
    loc.run_fault_test(mu=0.05, sigma=0.02, k=3.0)
    assert loc.flags.float().mean().item() <= 0.01


def test_recovery_bounds_error():
    loc, pos, alive, _ = run_with_fault(recover=True)
    err_faulted = (loc.p_hat[:, 3] - pos[:, 3]).norm(dim=1)
    assert err_faulted.mean().item() < 0.20  # recovered despite 1.0 m ranging bias


def test_recovery_skipped_below_min_peers():
    torch.manual_seed(0)
    rng, loc = make_stack(dropout=1.0, latency=0)  # no usable links at all
    pos0, _ = octagon_traj(0.0)
    loc.reset_idx(torch.arange(E), pos0)
    rng.reset_idx(torch.arange(E), pos0)
    alive = torch.ones(E, A, dtype=torch.bool)
    loc.flags[:, 2] = True
    before = loc.p_hat.clone()
    ranges, valid = rng.measure(pos0)
    loc.recover(ranges, valid, alive)
    assert torch.allclose(loc.p_hat, before)  # dead-reckon hold, no update
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `python3 -m pytest tests/test_localization.py -q`
Expected: new tests error with `AttributeError: ... has no attribute 'run_fault_test'`.

- [ ] **Step 3: Add the two methods to `DecentralizedLocalizer`**

Append below `correct` (before the diagnostics section):

```python
    # ------------------------------------------------- fault test + recovery

    def run_fault_test(self, mu: float, sigma: float, k: float) -> torch.Tensor:
        """Flag drones whose residual exceeds mu + k*sigma (per-drone local verdict)."""
        self.flags.copy_(self.residual > (mu + k * sigma))
        return self.flags

    def recover(self, ranges: torch.Tensor, valid: torch.Tensor, alive_g: torch.Tensor) -> None:
        """IRLS/Huber re-multilateration of flagged drones from non-flagged peers.

        Mirrors the SwarmRaft paper's Stage-2 recovery without its Raft
        transport. Drones with < min_recovery_peers usable peers keep their
        dead-reckoned estimate (matches the paper's INS-fallback note).
        """
        if not self.flags.any():
            return
        # Usable target links: valid, both alive, target NOT flagged.
        w_ok = self._link_weights(valid, alive_g) * (~self.flags).unsqueeze(1).float()
        enough = w_ok.sum(dim=2) >= float(self.min_recovery_peers)  # [E, A]
        upd_mask = (self.flags & enough & alive_g).unsqueeze(2).float()  # [E, A, 1]
        if upd_mask.sum() == 0:
            return
        for _ in range(self.recovery_irls_iters):
            diff = self.p_hat.unsqueeze(2) - self._p_broadcast.unsqueeze(1)
            dist = diff.norm(dim=3).clamp(min=1e-6)
            r = dist - ranges
            huber = torch.clamp(self.recovery_huber_delta / r.abs().clamp(min=1e-6), max=1.0)
            w = w_ok * huber
            step = self._gn_step_with(ranges, w, diff, dist, r)
            self.p_hat -= self.damping * step * upd_mask
        self._p_broadcast.copy_(self.p_hat)

    def _gn_step_with(self, ranges, w, diff, dist, r) -> torch.Tensor:
        """GN step from precomputed geometry (avoids recomputing diff/dist)."""
        u = diff / dist.unsqueeze(3)
        g = ((w * r).unsqueeze(3) * u).sum(dim=2)
        wsum = w.sum(dim=2, keepdim=True).clamp(min=1e-6)
        return g / wsum
```

Also refactor `_gn_step` to call `_gn_step_with` (compute `diff`/`dist`/`r` then delegate) so the math lives once.

- [ ] **Step 4: Run all tests**

Run: `python3 -m pytest tests/ -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add source/ggswarm/ggswarm/localization.py tests/test_localization.py
git commit -m "feat(loc): residual fault test + IRLS multilateration recovery (Stage 2b)"
```

---

### Task 5: Cfg parameters + init-time guard (Loc-b start)

**Files:**
- Modify: `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_marl_env_cfg.py` (append after the DropoutGuard block, line ~178)

**Interfaces:**
- Produces: cfg attribute names Task 6 reads verbatim: `loc_enabled`, `loc_obs_source`, `uwb_range_noise_std_m`, `uwb_range_bias_m`, `uwb_link_dropout_prob`, `uwb_latency_steps`, `odom_vel_noise_std_mps`, `loc_correct_iters`, `loc_gn_damping`, `residual_test_enabled`, `residual_mu`, `residual_sigma`, `residual_k`, `recovery_enabled`, `recovery_irls_iters`, `recovery_huber_delta`, `loc_min_recovery_peers`, `fault_inject_enabled`, `fault_bias_m`, `fault_step_min`, `fault_step_max`, `fault_count`, `loc_noise_anneal_start`, `loc_noise_anneal_end`, `loc_noise_scale_min`.

- [ ] **Step 1: Append the cfg block**

```python
    # Decentralized localization (Phase 1 Goal A — shadow mode in this build;
    # see docs/superpowers/specs/2026-07-04-phase1-decentralized-localization-design.md)
    loc_enabled = False
    loc_obs_source = "ground_truth"   # "ground_truth" only until Stage 5 lands
    uwb_range_noise_std_m = 0.10
    uwb_range_bias_m = 0.05
    uwb_link_dropout_prob = 0.05
    uwb_latency_steps = 1
    odom_vel_noise_std_mps = 0.02
    loc_correct_iters = 3
    loc_gn_damping = 0.5
    # residual fault test — mu/sigma come from scripts/calibrate_residual_threshold.py
    residual_test_enabled = False
    residual_mu = 0.0
    residual_sigma = 0.0
    residual_k = 3.0
    recovery_enabled = False
    recovery_irls_iters = 5
    recovery_huber_delta = 0.10
    loc_min_recovery_peers = 4
    # fault injection (FN / recovery-time evaluation)
    fault_inject_enabled = False
    fault_bias_m = 1.0
    fault_step_min = 200
    fault_step_max = 350
    fault_count = 1
    # noise-anneal curriculum (Stage 5 — declared now, unused in shadow mode)
    loc_noise_anneal_start = 0
    loc_noise_anneal_end = 5000
    loc_noise_scale_min = 0.1
```

- [ ] **Step 2: Verify import still works**

Run: `python3 -c "import sys; sys.path.insert(0, 'source/ggswarm'); import ast; ast.parse(open('source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_marl_env_cfg.py').read()); print('OK')"`
Expected: `OK` (full import needs Isaac; syntax check suffices here — the Windows smoke in Task 6 covers the real import).

- [ ] **Step 3: Commit**

```bash
git add source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_marl_env_cfg.py
git commit -m "feat(loc): localization cfg parameters (shadow-mode defaults, all off)"
```

---

### Task 6: Env integration — shadow mode (Stage 3)

**Files:**
- Modify: `source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_marl_env.py`:
  - `__init__` (after the DropoutGuard state block, ~line 105)
  - `_get_observations` (top, line ~388)
  - `_reset_idx` (end, after line 798)
  - new method `_update_localization` (place after `_get_observations`'s helper `_draw_debug_overlay`)

**Interfaces:**
- Consumes: `UwbRangingSim` and `DecentralizedLocalizer` exactly as defined in Tasks 2–4; cfg names from Task 5.
- Produces: TB log keys `Metrics/loc_rmse_m`, `Metrics/loc_gauge_drift_m`, `Metrics/loc_flag_rate` (Tasks 7–8 read these), and env attributes `self._localizer`, `self._ranging`, `self._fault_step [N_envs]`, `self._fault_mask [N_envs, A]` (the calibration/eval scripts reach into `env.unwrapped._localizer`).

- [ ] **Step 1: `__init__` additions**

Insert after the DropoutGuard state block:

```python
        # Decentralized localization (shadow mode — obs remain ground truth)
        if self.cfg.loc_obs_source != "ground_truth":
            raise ValueError(
                "loc_obs_source='estimate' is Stage 5 (obs swap); this build is shadow-mode only."
            )
        if self.cfg.loc_enabled:
            from ggswarm.localization import DecentralizedLocalizer  # noqa: PLC0415
            from ggswarm.ranging import UwbRangingSim  # noqa: PLC0415

            self._ranging = UwbRangingSim(
                N_envs, A, device,
                noise_std=self.cfg.uwb_range_noise_std_m,
                bias=self.cfg.uwb_range_bias_m,
                dropout_prob=self.cfg.uwb_link_dropout_prob,
                latency_steps=self.cfg.uwb_latency_steps,
            )
            self._localizer = DecentralizedLocalizer(
                N_envs, A, device,
                correct_iters=self.cfg.loc_correct_iters,
                damping=self.cfg.loc_gn_damping,
                odom_noise_std=self.cfg.odom_vel_noise_std_mps,
                recovery_irls_iters=self.cfg.recovery_irls_iters,
                recovery_huber_delta=self.cfg.recovery_huber_delta,
                min_recovery_peers=self.cfg.loc_min_recovery_peers,
            )
            self._fault_step = torch.zeros(N_envs, dtype=torch.long, device=device)  # [N_envs]
            self._fault_mask = torch.zeros(N_envs, A, dtype=torch.bool, device=device)  # [N_envs, A]
```

- [ ] **Step 2: Hook at the top of `_get_observations`**

First statement of the method body (before `A = self._A`... keep that line, insert after the locals):

```python
        if self.cfg.loc_enabled:
            self._update_localization()
```

- [ ] **Step 3: Add `_update_localization`**

```python
    def _update_localization(self) -> None:
        """Shadow-mode localization tick: estimator runs and logs; obs untouched.

        Order: fault trigger -> propagate -> measure -> correct -> test ->
        recover -> log. Runs once per env step (DirectMARLEnv calls
        _get_observations once per step, after _reset_idx re-seeds fresh envs).
        """
        A = self._A
        N_envs = self.num_envs
        alive_g = self._agent_alive.reshape(N_envs, A)  # shape: [N_envs, A]
        pos_true_g = (self._robot.data.root_pos_w - self._env_origins_per_drone).reshape(
            N_envs, A, 3
        )  # shape: [N_envs, A, 3]

        # Fault injection at the scheduled per-env step (DropoutGuard scheduling pattern).
        if self.cfg.fault_inject_enabled:
            trigger = (self.episode_length_buf == self._fault_step) & (self._fault_step > 0)
            if trigger.any():
                self._ranging.inject_fault(
                    self._fault_mask & trigger.unsqueeze(1), self.cfg.fault_bias_m
                )

        self._localizer.propagate(self._robot.data.root_lin_vel_w, self.step_dt)
        ranges, valid = self._ranging.measure(pos_true_g)
        self._localizer.correct(ranges, valid, alive_g)
        if self.cfg.residual_test_enabled:
            self._localizer.run_fault_test(
                self.cfg.residual_mu, self.cfg.residual_sigma, self.cfg.residual_k
            )
            if self.cfg.recovery_enabled:
                self._localizer.recover(ranges, valid, alive_g)

        log0 = self.extras[self._agent_ids[0]].setdefault("log", {})
        log0["Metrics/loc_rmse_m"] = self._localizer.rmse(pos_true_g, alive_g).mean().item()
        log0["Metrics/loc_gauge_drift_m"] = (
            self._localizer.gauge_drift(pos_true_g, alive_g).mean().item()
        )
        if self.cfg.residual_test_enabled:
            log0["Metrics/loc_flag_rate"] = self._localizer.flags.float().mean().item()
```

- [ ] **Step 4: `_reset_idx` additions (very end, after `write_joint_state_to_sim`)**

```python
        # Localization: seed estimates from spawn truth; clear channel state.
        if self.cfg.loc_enabled:
            pos_spawn_g = (
                default_root_state[:, :3] - self._env_origins_per_drone[drone_ids]
            ).reshape(n_envs_reset, A, 3)  # shape: [n_reset, A, 3]
            self._localizer.reset_idx(env_ids, pos_spawn_g)
            self._ranging.reset_idx(env_ids, pos_spawn_g)
            if self.cfg.fault_inject_enabled:
                self._fault_step[env_ids] = torch.randint(
                    self.cfg.fault_step_min,
                    self.cfg.fault_step_max + 1,
                    (n_envs_reset,),
                    device=self.device,
                )
                self._fault_mask[env_ids] = False
                for e_idx in range(n_envs_reset):
                    victims = torch.randperm(A, device=self.device)[: self.cfg.fault_count]
                    self._fault_mask[env_ids[e_idx], victims] = True
```

(Reset path — explicit constructors are allowed here, matching the existing `torch.randint` dropout scheduling above it.)

- [ ] **Step 5: [Windows] Off-means-off replay gate**

```text
env_isaaclab/Scripts/python.exe scripts/skrl/replay_gate.py --headless --task ggswarm-marl-v0 --checkpoint logs/ref/v1.0.0-capstone/best_agent.pt
```

Run once on this branch and once on the pre-Task-6 commit (`git stash` or checkout); metrics must be **identical** (`loc_enabled=False` never touches the step path).

- [ ] **Step 6: [Windows] Shadow-mode replay**

Temporarily set `loc_enabled = True` in the cfg (or via play-script cfg override if available) and re-run the replay gate command. Expected: formation metrics within 2σ of the Step 5 run; TensorBoard/console shows `Metrics/loc_rmse_m` ≤ 0.10 steady-state; step time within noise of Step 5 (allocation ban holds). Revert `loc_enabled` to `False` afterwards.

- [ ] **Step 7: [Windows] 5-iteration smoke train** (same command as Task 1 Step 4). Expected: clean.

- [ ] **Step 8: Commit**

```bash
git add source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_marl_env.py
git commit -m "feat(loc): shadow-mode localization integration in GgswarmMarlEnv (Stage 3)"
```

---

### Task 7: `scripts/calibrate_residual_threshold.py` (Stage 4a)

**Files:**
- Create: `scripts/calibrate_residual_threshold.py`
- Reference (read before writing): `scripts/skrl/replay_gate.py` — copy its AppLauncher argparse/boilerplate, env construction, checkpoint loading (`ggswarm.checkpoint_utils.load_actor_weights` + `init_state_preprocessor`), and rollout loop structure exactly; only the measurement core below is new.

**Interfaces:**
- Consumes: `env.unwrapped._localizer.residual` (`[N_envs, A]`, refreshed by `correct` each step), `env.unwrapped._agent_alive`.
- Produces: printed `residual_mu` / `residual_sigma` values the user pastes into `ggswarm_marl_env_cfg.py`.

- [ ] **Step 1: Write the script**

Structure (AppLauncher header and checkpoint/env sections mirrored from `replay_gate.py`; new args: `--episodes` default 50, `--warmup_steps` default 50). Force these cfg values on `env_cfg` before `gym.make`: `loc_enabled=True`, `residual_test_enabled=False`, `fault_inject_enabled=False`, `dropout_enabled=False`. Measurement core inside the rollout loop:

```python
    # After warmup (transients from spawn + policy settle), harvest residuals
    # from alive drones only. Policy-in-the-loop => maneuver envelope, not hover.
    residual_samples: list[torch.Tensor] = []  # script scope: allocations fine
    step_in_episode = 0
    ...
    if step_in_episode >= args.warmup_steps:
        loc = env.unwrapped._localizer
        alive = env.unwrapped._agent_alive.reshape(env.unwrapped.num_envs, -1)
        residual_samples.append(loc.residual[alive].flatten().cpu())
```

And after the loop:

```python
    all_res = torch.cat(residual_samples)
    mu, sigma = all_res.mean().item(), all_res.std().item()
    p999 = all_res.quantile(0.999).item()
    print(f"samples          : {all_res.numel()}")
    print(f"residual_mu      = {mu:.4f}")
    print(f"residual_sigma   = {sigma:.4f}")
    print(f"mu + 3*sigma     = {mu + 3 * sigma:.4f}   (flag threshold)")
    print(f"p99.9 residual   = {p999:.4f}   (sanity: should sit near the threshold)")
    print("Paste residual_mu / residual_sigma into GgswarmMarlEnvCfg.")
```

- [ ] **Step 2: [Windows] Run it**

```text
env_isaaclab/Scripts/python.exe scripts/calibrate_residual_threshold.py --headless --task ggswarm-marl-v0 --checkpoint logs/ref/v1.0.0-capstone/best_agent.pt --num_envs 64 --episodes 50
```

Expected: prints mu/sigma; `mu + 3*sigma` should land in the same order of magnitude as the honest-noise residual scale from the Task 4 unit tests (~0.05–0.15 m). Paste the two values into the cfg defaults and set nothing else.

- [ ] **Step 3: Commit**

```bash
git add scripts/calibrate_residual_threshold.py source/ggswarm/ggswarm/tasks/direct/ggswarm/ggswarm_marl_env_cfg.py
git commit -m "feat(loc): residual-threshold calibration script + calibrated mu/sigma (Stage 4a)"
```

---

### Task 8: `scripts/eval_localization.py` (Stage 4b)

**Files:**
- Create: `scripts/eval_localization.py`
- Reference: same boilerplate sources as Task 7; metric-aggregation helpers in `scripts/eval_metrics.py` may be reused where they fit.

**Interfaces:**
- Consumes: everything Task 7 consumes, plus `env.unwrapped._localizer.flags`, `env.unwrapped._fault_mask`, `env.unwrapped._fault_step`, `env.unwrapped._localizer.p_hat`, TB keys from Task 6.
- Produces: a printed scorecard — the Loc-b gate numbers.

- [ ] **Step 1: Write the script**

Two modes, `--mode honest` and `--mode fault` (new args: `--episodes` default 100). Cfg forcing: `loc_enabled=True`, `residual_test_enabled=True`, `recovery_enabled=True`; `fault_inject_enabled = (mode == "fault")`; `dropout_enabled=False`. Per-step collection core:

```python
    # honest mode: FP = fraction of alive, never-faulted drone-steps flagged.
    # fault mode:  per faulted drone, detection latency (steps from fault_step to
    #   first flag), recovery time (steps from fault_step until estimate error
    #   re-enters 0.10 m), FN = faulted drones never flagged before episode end.
    u = env.unwrapped
    loc, A = u._localizer, u.cfg.num_agents
    pos_true_g = (u._robot.data.root_pos_w - u._env_origins_per_drone).reshape(u.num_envs, A, 3)
    err = (loc.p_hat - pos_true_g).norm(dim=2)                      # [N_envs, A]
    flags, faulted = loc.flags.cpu(), u._fault_mask.cpu()
    started = (u.episode_length_buf.unsqueeze(1).cpu() >= u._fault_step.unsqueeze(1).cpu()) & faulted
```

Aggregate across episodes and print:

```python
    print(f"episodes                 : {n_episodes}")
    print(f"loc RMSE (steady, honest): {rmse:.4f} m      (gate <= 0.10)")
    print(f"FP rate (honest)         : {fp_rate:.4f}     (gate <= 0.01)")
    print(f"FN rate (fault)          : {fn_rate:.4f}     (gate <= 0.05)")
    print(f"recovery time p50/p95    : {rec_p50:.2f}s / {rec_p95:.2f}s (gate <= 1.0 s)")
    print(f"formation collapses      : {collapses} / {n_episodes} episodes (gate 0)")
```

`recovery seconds = steps * env.unwrapped.step_dt` (0.02 s/step ⇒ gate is 50 steps). "Formation collapse" = any episode terminating via the existing collision/altitude `died_env` before timeout.

- [ ] **Step 2: [Windows] Run both modes**

```text
env_isaaclab/Scripts/python.exe scripts/eval_localization.py --headless --task ggswarm-marl-v0 --checkpoint logs/ref/v1.0.0-capstone/best_agent.pt --num_envs 64 --episodes 100 --mode honest
env_isaaclab/Scripts/python.exe scripts/eval_localization.py --headless --task ggswarm-marl-v0 --checkpoint logs/ref/v1.0.0-capstone/best_agent.pt --num_envs 64 --episodes 100 --mode fault
```

Expected: all five gate lines pass. If FP > 0.01, re-run Task 7 with more episodes and check the warmup window covers reset transients; if recovery > 1.0 s, raise `recovery_irls_iters` (cfg) before touching the math.

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_localization.py
git commit -m "feat(loc): localization scorecard evaluation script (Stage 4b)"
```

---

### Task 9: Gate record + docs

**Files:**
- Modify: `docs/ggswarm_live/status/changelog.md` (new dated entry), `docs/ggswarm_live/status/log.md` (short note)
- Modify: `docs/ggswarm_live/decentralization_plan.md` (mark Stages 0–4 done, record scorecard numbers)
- Modify: `docs/ggswarm_live/phases/phase1_sim.md` (Goal A: peer-to-peer localization → "shadow mode complete, obs swap pending")
- Modify: `cspell.json` if any new terms flag (e.g., `multilateration` variants, `IRLS`)

**Interfaces:** none.

- [ ] **Step 1: Write the changelog entry** — date, the two module names, cfg params added, calibrated `residual_mu`/`residual_sigma`, and the full measured scorecard vs gates (RMSE, FP, FN, recovery p50/p95, collapses), plus the replay-gate off/shadow comparison results. Follow the existing entry format in the file.

- [ ] **Step 2: Update the other three docs** as listed (status flips + scorecard table in `decentralization_plan.md` §6; one-line pointer in `log.md`).

- [ ] **Step 3: Markdown hygiene check** — GFM tables have `| :--- |` separators; blank lines around any fenced block added.

- [ ] **Step 4: Commit**

```bash
git add docs/ cspell.json
git commit -m "docs(loc): record Loc-a/Loc-b gates, scorecard, and stage status (Stages 0-4 complete)"
```

---

## Self-Review Record

- **Spec coverage:** Stage 0 → Task 1; Stage 1 → Task 2; Stage 2 → Tasks 3–4; Stage 3 → Tasks 5–6; Stage 4 → Tasks 7–8; spec §6 gates → Task 6 Steps 5–7 (Loc-b parts 1–2), Task 8 (part 3), recorded in Task 9. Error-handling spec §5: ill-conditioning → Task 4 `min_recovery_peers` skip; all-links-dropped → Task 3 dead-reckon test; NaN guard → Task 3 10× noise test; misconfiguration → Task 6 Step 1 init guard.
- **Known deviation:** `loc_min_recovery_peers` cfg name (spec table implied `recovery_*` prefix family) — recorded here as the canonical name; Tasks 4/5/6 use it consistently.
- **Type consistency:** `measure -> (held_ranges, valid_d)` consumed identically in Tasks 3, 4, 6; `alive_g [E, A]` bool everywhere; `p_hat`/`residual`/`flags` names match across Tasks 3, 4, 6, 7, 8.
