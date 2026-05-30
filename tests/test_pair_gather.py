# tests/test_pair_gather.py
import itertools
import pytest
import torch

from neural_shift_cuda import (
    pair_gather,
    pair_gather_reference,
    shift_gather_reference,
)

CUDA_OK = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA_OK, reason="CUDA not available")


def _all_shifts(R: int) -> torch.Tensor:
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
    (3, 12, 33, 47),
]
RADII = [1, 2, 4]


# ---------------------------------------------------------------------------
# (3) test_pair_gather_matches_reference
# ---------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("shape,R", list(itertools.product(SHAPES, RADII)))
def test_pair_gather_matches_reference(shape, R):
    B, C, H, W = shape
    torch.manual_seed(0)
    guide = torch.randn(B, C, H, W, device="cuda", dtype=torch.float32)
    shifts = _all_shifts(R).to("cuda")

    pair_cuda, mask_cuda = pair_gather(guide, shifts)
    pair_ref, mask_ref = pair_gather_reference(guide, shifts)

    torch.testing.assert_close(pair_cuda, pair_ref, rtol=0, atol=0)
    torch.testing.assert_close(mask_cuda, mask_ref, rtol=0, atol=0)

    # Also matches the manual construction torch.cat([g_repeat, gs_batch], dim=1)
    gs_ref, _ = shift_gather_reference(guide, shifts)
    S = gs_ref.size(0) // B
    g_batch = guide.unsqueeze(0).expand(S, B, *guide.shape[1:]).reshape_as(gs_ref)
    pair_manual = torch.cat([g_batch, gs_ref], dim=1).contiguous()
    torch.testing.assert_close(pair_cuda, pair_manual, rtol=0, atol=0)


# ---------------------------------------------------------------------------
# gradcheck
# ---------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("R", [1, 2])
def test_gradcheck_pair_gather(R):
    B, C, H, W = 1, 2, 5, 5
    guide = torch.randn(B, C, H, W, device="cuda", dtype=torch.float64,
                        requires_grad=True)
    shifts = _all_shifts(R).to("cuda")

    def fn(g):
        p, _ = pair_gather(g, shifts)
        return p

    assert torch.autograd.gradcheck(fn, (guide,), eps=1e-6, atol=1e-5)


@requires_cuda
@pytest.mark.parametrize("R", [1, 3])
def test_pair_gather_backward_matches_reference(R):
    B, C, H, W = 2, 4, 13, 11
    torch.manual_seed(7)
    g1 = torch.randn(B, C, H, W, device="cuda", dtype=torch.float64,
                     requires_grad=True)
    g2 = g1.detach().clone().requires_grad_(True)
    shifts = _all_shifts(R).to("cuda")

    p1, _ = pair_gather(g1, shifts)
    p2, _ = pair_gather_reference(g2, shifts)

    grad_out = torch.randn_like(p1)
    p1.backward(grad_out)
    p2.backward(grad_out)

    torch.testing.assert_close(g1.grad, g2.grad, rtol=1e-6, atol=1e-6)
