# tests/test_accumulate_uz.py
import itertools
import pytest
import torch

from neural_shift_cuda import accumulate_uz, accumulate_uz_reference

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
    (1, 3, 16, 16),
    (2, 3, 31, 29),
    (3, 6, 33, 47),
]
RADII = [1, 2, 4]


def _premasked_weights(B, C, H, W, R, shifts, dtype, device):
    """Make a random weights tensor already multiplied by the forward mask."""
    S = shifts.size(0)
    w = torch.randn(S * B, C, H, W, device=device, dtype=dtype)

    import torch.nn.functional as F
    box = F.pad(torch.ones(1, 1, H, W, device=device, dtype=dtype),
                (R, R, R, R), mode='constant', value=0)
    masks = []
    for i in range(S):
        dx = int(shifts[i, 0].item())
        dy = int(shifts[i, 1].item())
        masks.append(box[:, :, R + dx:R + dx + H, R + dy:R + dy + W])
    mask = torch.cat(masks, dim=0)            # (S, 1, H, W)
    mask = mask.unsqueeze(1).expand(S, B, 1, H, W).reshape(S * B, 1, H, W)
    return w * mask


@requires_cuda
@pytest.mark.parametrize("shape,R", list(itertools.product(SHAPES, RADII)))
def test_accumulate_uz_forward(shape, R):
    B, C, H, W = shape
    torch.manual_seed(0)
    shifts = _all_shifts(R).to("cuda")
    x = torch.randn(B, C, H, W, device="cuda", dtype=torch.float32)
    w = _premasked_weights(B, C, H, W, R, shifts, torch.float32, "cuda")

    U_cuda, Z_cuda = accumulate_uz(x, w, shifts)
    U_ref, Z_ref = accumulate_uz_reference(x, w, shifts)

    torch.testing.assert_close(U_cuda, U_ref, rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(Z_cuda, Z_ref, rtol=1e-5, atol=1e-5)


@requires_cuda
@pytest.mark.parametrize("R", [1, 2])
def test_gradcheck_accumulate_uz(R):
    B, C, H, W = 1, 2, 6, 5
    torch.manual_seed(0)
    shifts = _all_shifts(R).to("cuda")
    x = torch.randn(B, C, H, W, device="cuda", dtype=torch.float64,
                    requires_grad=True)
    w = _premasked_weights(B, C, H, W, R, shifts, torch.float64, "cuda")
    w.requires_grad_(True)

    def fn(x_, w_):
        U, Z = accumulate_uz(x_, w_, shifts)
        return U, Z

    assert torch.autograd.gradcheck(fn, (x, w), eps=1e-6, atol=1e-5)


@requires_cuda
@pytest.mark.parametrize("R", [1, 3])
def test_accumulate_uz_backward_matches_reference(R):
    B, C, H, W = 2, 3, 13, 11
    torch.manual_seed(7)
    shifts = _all_shifts(R).to("cuda")

    x1 = torch.randn(B, C, H, W, device="cuda", dtype=torch.float64,
                     requires_grad=True)
    w1_raw = _premasked_weights(B, C, H, W, R, shifts, torch.float64, "cuda")
    w1 = w1_raw.detach().requires_grad_(True)

    x2 = x1.detach().clone().requires_grad_(True)
    w2 = w1.detach().clone().requires_grad_(True)

    U1, Z1 = accumulate_uz(x1, w1, shifts)
    U2, Z2 = accumulate_uz_reference(x2, w2, shifts)

    g_U = torch.randn_like(U1)
    g_Z = torch.randn_like(Z1)

    (U1 * g_U + Z1 * g_Z).sum().backward()
    (U2 * g_U + Z2 * g_Z).sum().backward()

    torch.testing.assert_close(x1.grad, x2.grad, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(w1.grad, w2.grad, rtol=1e-6, atol=1e-6)
