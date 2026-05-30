# tests/test_shift_gather.py
import itertools
import pytest
import torch

from neural_shift_cuda import shift_gather, shift_gather_reference

CUDA_OK = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA_OK, reason="CUDA not available")


def _all_shifts(R: int) -> torch.Tensor:
    """Match nekre._collect_shifts (upper half + horizontal axis)."""
    rows = []
    for dx in range(-R, R + 1):
        for dy in range(1, R + 1):
            rows.append((dx, dy, 1))
    for dx in range(0, R + 1):
        rows.append((dx, 0, 0 if dx == 0 else 1))
    return torch.tensor(rows, dtype=torch.int32)


SHAPES = [
    (1, 8, 16, 16),
    (2, 16, 31, 29),
    (4, 32, 64, 64),
]
RADII = [1, 2, 5]


# ---------------------------------------------------------------------------
# (1) test_shift_gather_matches_reference
# ---------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("shape,R", list(itertools.product(SHAPES, RADII)))
def test_shift_gather_matches_reference(shape, R):
    B, C, H, W = shape
    torch.manual_seed(0)
    guide = torch.randn(B, C, H, W, device="cuda", dtype=torch.float32)
    shifts = _all_shifts(R).to("cuda")

    gs_cuda, mask_cuda = shift_gather(guide, shifts)
    gs_ref, mask_ref = shift_gather_reference(guide, shifts)

    torch.testing.assert_close(gs_cuda, gs_ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# (2) test_mask_matches_reference
# ---------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("shape,R", list(itertools.product(SHAPES, RADII)))
def test_mask_matches_reference(shape, R):
    B, C, H, W = shape
    torch.manual_seed(0)
    guide = torch.randn(B, C, H, W, device="cuda", dtype=torch.float32)
    shifts = _all_shifts(R).to("cuda")

    _, mask_cuda = shift_gather(guide, shifts)
    _, mask_ref = shift_gather_reference(guide, shifts)

    torch.testing.assert_close(mask_cuda, mask_ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# float64 forward equivalence
# ---------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("R", [1, 2])
def test_shift_gather_float64(R):
    B, C, H, W = 2, 4, 12, 10
    torch.manual_seed(0)
    guide = torch.randn(B, C, H, W, device="cuda", dtype=torch.float64)
    shifts = _all_shifts(R).to("cuda")

    gs_cuda, _ = shift_gather(guide, shifts)
    gs_ref, _ = shift_gather_reference(guide, shifts)
    torch.testing.assert_close(gs_cuda, gs_ref, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# (4) test_gradcheck_shift_gather  -- analytic vs numeric Jacobian
# ---------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("R", [1, 2])
def test_gradcheck_shift_gather(R):
    # gradcheck wants float64 and very small tensors.
    B, C, H, W = 1, 2, 5, 5
    guide = torch.randn(B, C, H, W, device="cuda", dtype=torch.float64,
                        requires_grad=True)
    shifts = _all_shifts(R).to("cuda")

    def fn(g):
        gs, _ = shift_gather(g, shifts)
        return gs

    assert torch.autograd.gradcheck(fn, (guide,), eps=1e-6, atol=1e-5)


# ---------------------------------------------------------------------------
# Analytic backward also matches PyTorch autograd through the reference impl
# ---------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("R", [1, 3])
def test_shift_gather_backward_matches_reference(R):
    B, C, H, W = 2, 4, 13, 11
    torch.manual_seed(7)
    g1 = torch.randn(B, C, H, W, device="cuda", dtype=torch.float64,
                     requires_grad=True)
    g2 = g1.detach().clone().requires_grad_(True)
    shifts = _all_shifts(R).to("cuda")

    out1, _ = shift_gather(g1, shifts)
    out2, _ = shift_gather_reference(g2, shifts)

    # Same upstream gradient.
    grad_out = torch.randn_like(out1)
    out1.backward(grad_out)
    out2.backward(grad_out)

    torch.testing.assert_close(g1.grad, g2.grad, rtol=1e-6, atol=1e-6)
