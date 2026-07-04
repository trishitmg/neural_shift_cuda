# tests/test_accumulate_uz_scalar.py
#
# Tests for the scalar-per-transform accumulate op used by the GASD forward.
# These exercise the pure-PyTorch reference (CPU) and, when a CUDA build with
# the 0.4.0 symbols is present, the CUDA kernels against that reference.

import pytest
import torch

from neural_shift_cuda.ops import (
    accumulate_uz_scalar,
    accumulate_uz_scalar_reference,
    _has_scalar_ext,
)


def _window_shifts(R):
    return torch.tensor(
        [[dx, dy] for dx in range(-R, R + 1) for dy in range(-R, R + 1)],
        dtype=torch.int32,
    )


def _naive(x, weights, shifts):
    """Independent naive implementation (gather convention)."""
    B, C, H, W = x.shape
    S = shifts.size(0)
    U = torch.zeros_like(x)
    Z = x.new_zeros(B, C, 1, 1)
    for s in range(S):
        dx, dy = int(shifts[s, 0]), int(shifts[s, 1])
        xs = torch.roll(x, shifts=(-dx, -dy), dims=(-2, -1))
        w = weights[s].view(B, C, 1, 1)
        U = U + w * xs
        Z = Z + w
    return U, Z.expand(B, C, H, W)


def test_reference_matches_naive():
    torch.manual_seed(0)
    B, C, H, W, R = 2, 3, 11, 9, 2
    shifts = _window_shifts(R)
    S = shifts.size(0)
    x = torch.randn(B, C, H, W, dtype=torch.double)
    w = torch.rand(S, B, C, dtype=torch.double) + 0.1
    U1, Z1 = accumulate_uz_scalar_reference(x, w, shifts)
    U2, Z2 = _naive(x, w, shifts)
    assert torch.allclose(U1, U2, atol=1e-12)
    assert torch.allclose(Z1, Z2, atol=1e-12)


def test_reference_gradcheck():
    torch.manual_seed(1)
    B, C, H, W, R = 1, 2, 7, 7, 1
    shifts = _window_shifts(R)
    S = shifts.size(0)
    x = torch.randn(B, C, H, W, dtype=torch.double, requires_grad=True)
    w = (torch.rand(S, B, C, dtype=torch.double) + 0.1).requires_grad_(True)

    def f(xx, ww):
        return accumulate_uz_scalar_reference(xx, ww, shifts)

    assert torch.autograd.gradcheck(f, (x, w), atol=1e-6, rtol=1e-4)


def test_shape_validation():
    x = torch.randn(2, 3, 8, 8)
    shifts = _window_shifts(1)
    S = shifts.size(0)
    with pytest.raises(ValueError):
        accumulate_uz_scalar_reference(x, torch.rand(S, 2, 4), shifts)  # wrong C
    with pytest.raises(ValueError):
        accumulate_uz_scalar_reference(x, torch.rand(S + 1, 2, 3), shifts)  # wrong S


@pytest.mark.skipif(not torch.cuda.is_available() or not _has_scalar_ext(),
                    reason="CUDA build with 0.4.0 scalar symbols required")
def test_cuda_matches_reference():
    torch.manual_seed(2)
    B, C, H, W, R = 2, 3, 16, 16, 3
    shifts = _window_shifts(R)
    S = shifts.size(0)
    x = torch.randn(B, C, H, W, device="cuda")
    w = (torch.rand(S, B, C, device="cuda") + 0.1)

    xc = x.clone().requires_grad_(True)
    wc = w.clone().requires_grad_(True)
    xr = x.clone().cpu().requires_grad_(True)
    wr = w.clone().cpu().requires_grad_(True)

    Uc, Zc = accumulate_uz_scalar(xc, wc, shifts)                    # CUDA
    Ur, Zr = accumulate_uz_scalar_reference(xr, wr, shifts.cpu())    # reference

    assert torch.allclose(Uc.cpu(), Ur, atol=1e-4)
    assert torch.allclose(Zc.cpu(), Zr, atol=1e-4)

    g = torch.randn_like(Uc)
    (Uc * g).sum().backward()
    (Ur * g.cpu()).sum().backward()
    assert torch.allclose(xc.grad.cpu(), xr.grad, atol=1e-4)
    assert torch.allclose(wc.grad.cpu(), wr.grad, atol=1e-4)
