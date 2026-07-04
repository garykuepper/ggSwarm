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
