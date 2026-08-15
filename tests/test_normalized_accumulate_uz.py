import pytest
import torch

from neural_shift_cuda import (
    accumulate_uz_reference,
    normalized_accumulate_uz,
    normalized_accumulate_uz_reference,
)


def _periodic_shifts(radius: int) -> torch.Tensor:
    return torch.tensor(
        [(dx, dy, 0)
         for dx in range(-radius, radius + 1)
         for dy in range(-radius, radius + 1)],
        dtype=torch.int32,
    )


def _halfplane_shifts(radius: int) -> torch.Tensor:
    rows = []
    for dx in range(-radius, radius + 1):
        for dy in range(1, radius + 1):
            rows.append((dx, dy, 1))
    for dx in range(0, radius + 1):
        rows.append((dx, 0, 0 if dx == 0 else 1))
    return torch.tensor(rows, dtype=torch.int32)


def test_stable_positive_matches_exact_ratio():
    torch.manual_seed(0)
    B, C, H, W = 2, 3, 7, 6
    shifts = _halfplane_shifts(2)
    S = shifts.size(0)
    x = torch.randn(B, C, H, W, dtype=torch.double)
    weights = torch.rand(S * B, C, H, W, dtype=torch.double) + 0.1

    exact_u, exact_z = accumulate_uz_reference(x, weights, shifts)
    stable = normalized_accumulate_uz_reference(x, weights, shifts)
    torch.testing.assert_close(stable, exact_u / exact_z, rtol=1e-12, atol=1e-12)


def test_stable_positive_avoids_sum_overflow():
    B, C, H, W = 1, 1, 5, 4
    shifts = _periodic_shifts(1)
    S = shifts.size(0)
    x = torch.linspace(-1, 1, H * W).reshape(B, C, H, W)
    weights = torch.full((S * B, C, H, W), 1.0e38, dtype=torch.float32)

    # The requested unscaled degree is outside float32, while D is perfectly
    # representable. The stable path never forms this overflowing sum.
    assert torch.isinf(weights.view(S, B, C, H, W).sum(dim=0)).any()
    denoised, log_degree = normalized_accumulate_uz_reference(
        x, weights, shifts, return_log_degree=True)
    expected = torch.stack([
        torch.roll(x, shifts=(-int(s[0]), -int(s[1])), dims=(-2, -1))
        for s in shifts
    ]).mean(dim=0)
    assert torch.isfinite(denoised).all()
    assert torch.isfinite(log_degree).all()
    torch.testing.assert_close(denoised, expected, rtol=2e-6, atol=2e-6)


def test_log_weight_mode_handles_huge_logits_and_offset_invariance():
    torch.manual_seed(1)
    B, C, H, W = 1, 2, 6, 5
    shifts = _periodic_shifts(2)
    S = shifts.size(0)
    x = torch.randn(B, C, H, W)
    logits = 10_000.0 + 3.0 * torch.randn(S * B, C, H, W)

    d1, log_c1 = normalized_accumulate_uz_reference(
        x, logits, shifts, log_weights=True, return_log_degree=True)
    d2, log_c2 = normalized_accumulate_uz_reference(
        x, logits + 5_000.0, shifts,
        log_weights=True, return_log_degree=True)
    assert torch.isfinite(d1).all()
    torch.testing.assert_close(d1, d2, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(
        log_c2 - log_c1, torch.full_like(log_c1, 5_000.0),
        rtol=0, atol=2e-3)


@pytest.mark.parametrize("log_weights", [False, True])
def test_stable_reference_gradcheck(log_weights):
    torch.manual_seed(2)
    B, C, H, W = 1, 1, 4, 4
    shifts = _periodic_shifts(1)
    S = shifts.size(0)
    x = torch.randn(B, C, H, W, dtype=torch.double, requires_grad=True)
    base = torch.randn(S * B, C, H, W, dtype=torch.double)
    weights = (base if log_weights else base.exp()).requires_grad_(True)

    def fn(xx, ww):
        return normalized_accumulate_uz_reference(
            xx, ww, shifts, log_weights=log_weights,
            return_log_degree=True)

    assert torch.autograd.gradcheck(fn, (x, weights), eps=1e-6, atol=2e-5)


def test_accumulator_is_promoted_before_reduction():
    shifts = _periodic_shifts(1)
    S = shifts.size(0)
    x = torch.ones(1, 1, 4, 4, dtype=torch.float16)
    weights = torch.ones(S, 1, 4, 4, dtype=torch.float16)
    u, z = accumulate_uz_reference(x, weights, shifts)
    d = normalized_accumulate_uz(x, weights, shifts)
    assert u.dtype == torch.float32
    assert z.dtype == torch.float32
    assert d.dtype == torch.float32


def test_weight_contract_validation():
    shifts = _periodic_shifts(0)
    x = torch.ones(1, 1, 2, 2)
    with pytest.raises(ValueError, match="nonnegative"):
        normalized_accumulate_uz_reference(
            x, -torch.ones(1, 1, 2, 2), shifts)
    with pytest.raises(ValueError, match=r"not NaN or \+inf"):
        normalized_accumulate_uz_reference(
            x, torch.full((1, 1, 2, 2), torch.inf), shifts,
            log_weights=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("log_weights", [False, True])
def test_cuda_stable_matches_reference_and_gradients(log_weights):
    torch.manual_seed(3)
    B, C, H, W = 1, 2, 8, 7
    shifts = _periodic_shifts(2)
    S = shifts.size(0)
    x0 = torch.randn(B, C, H, W, dtype=torch.double)
    raw = torch.randn(S * B, C, H, W, dtype=torch.double)
    w0 = raw if log_weights else raw.exp()

    xc = x0.cuda().requires_grad_(True)
    wc = w0.cuda().requires_grad_(True)
    xr = x0.requires_grad_(True)
    wr = w0.requires_grad_(True)

    dc, lc = normalized_accumulate_uz(
        xc, wc, shifts.cuda(), log_weights=log_weights,
        return_log_degree=True)
    dr, lr = normalized_accumulate_uz_reference(
        xr, wr, shifts, log_weights=log_weights,
        return_log_degree=True)
    torch.testing.assert_close(dc.cpu(), dr, rtol=1e-10, atol=1e-10)
    torch.testing.assert_close(lc.cpu(), lr, rtol=1e-10, atol=1e-10)

    gd = torch.randn_like(dc)
    gl = torch.randn_like(lc)
    (dc * gd + lc * gl).sum().backward()
    (dr * gd.cpu() + lr * gl.cpu()).sum().backward()
    torch.testing.assert_close(xc.grad.cpu(), xr.grad, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(wc.grad.cpu(), wr.grad, rtol=1e-9, atol=1e-9)
