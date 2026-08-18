"""Equivalence tests for the three fixed-guide cached CUDA patch functions.

The direct helpers dispatch to the compiled tree reduction on CUDA and to its
pure-PyTorch reference on CPU, so the same tests cover both paths when a GPU is
available.  These fixtures intentionally contain only final cached weights:
no feature extractor or attention network is evaluated.
"""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from neural_shift_cuda.integration import gasd_drunet_attn_patch as gasd_patch
from neural_shift_cuda.integration import nekde_drunet_attn_patch as nekde_patch
from neural_shift_cuda.integration import nkd_metropolis_attn_patch as mp_patch


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _box(batch, channels, height, width, radius, device):
    return F.pad(
        torch.ones(batch, channels, height, width, device=device),
        (radius, radius, radius, radius), mode="constant", value=0.0)


def _shift(x, dx, dy):
    """Gather convention: out[h,w] = x[h+dx,w+dy], circularly."""
    return torch.roll(x, shifts=(-dx, -dy), dims=(-2, -1))


def _symmetric_serial(x, weights, shifts, radius, box, use_box):
    U = torch.zeros_like(x)
    Z = torch.zeros_like(x)
    H, W = x.shape[-2:]
    for w_fwd, (dx, dy, has_inv) in zip(weights, shifts):
        U = U + w_fwd * _shift(x, dx, dy)
        Z = Z + w_fwd
        if has_inv:
            w_inv = torch.roll(w_fwd, shifts=(dx, dy), dims=(-2, -1))
            if use_box:
                w_inv = w_inv * box[
                    :, :, radius - dx:radius - dx + H,
                    radius - dy:radius - dy + W]
            U = U + w_inv * _shift(x, -dx, -dy)
            Z = Z + w_inv
    return U, Z


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_box", [True, False])
def test_nekde_forward_cached_parallel(device, use_box):
    torch.manual_seed(1)
    B, C, H, W, R = 2, 3, 7, 8, 1
    x = torch.randn(B, C, H, W, device=device)
    shifts = [
        (0, 0, False), (0, 1, True), (1, -1, True),
        (1, 0, True), (1, 1, True),
    ]
    box = _box(B, C, H, W, R, device)
    weights = []
    for dx, dy, _ in shifts:
        w = torch.rand_like(x) + 0.1
        if use_box:
            w = w * box[:, :, R + dx:R + dx + H, R + dy:R + dy + W]
        weights.append(w)
    Kx_ref, D = _symmetric_serial(x, weights, shifts, R, box, use_box)
    cache = SimpleNamespace(
        C=C, shifts=shifts, w_fwd_list=weights,
        comp_box=use_box, D=D)
    model = SimpleNamespace()
    out, degree = nekde_patch._forward_cached_cuda(model, x, cache, True)
    torch.testing.assert_close(
        out, Kx_ref / D.clamp_min(1e-6), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(degree, D)

    packed = model._nekde_parallel_weight_cache[4]
    nekde_patch._forward_cached_cuda(model, x, cache, False)
    assert model._nekde_parallel_weight_cache[4] is packed


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_box", [True, False])
def test_gasd_forward_and_transpose_cached_parallel(device, use_box):
    torch.manual_seed(2)
    B, C, H, W, R = 2, 3, 7, 8, 1
    x = torch.randn(B, C, H, W, device=device)
    shifts = [(dx, dy) for dx in range(-R, R + 1)
              for dy in range(-R, R + 1)]
    box = _box(B, C, H, W, R, device)
    weights = []
    degree = torch.zeros_like(x)
    Kx_ref = torch.zeros_like(x)
    KT_ref = torch.zeros_like(x)
    for dx, dy in shifts:
        w = torch.rand_like(x) + 0.1
        if use_box:
            w = w * box[:, :, R + dx:R + dx + H, R + dy:R + dy + W]
        weights.append(w)
        degree = degree + w
        Kx_ref = Kx_ref + w * _shift(x, dx, dy)
        KT_ref = KT_ref + torch.roll(
            w * x, shifts=(dx, dy), dims=(-2, -1))

    cache = SimpleNamespace(C=C, shifts=shifts, w_list=weights, Z=degree)
    model = SimpleNamespace()
    out, returned_degree = gasd_patch._forward_cached_cuda(
        model, x, cache, True)
    KT = gasd_patch._kt_action_cached_cuda(model, x, cache)
    torch.testing.assert_close(
        out, Kx_ref / degree.clamp_min(1e-6), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(returned_degree, degree)
    torch.testing.assert_close(KT, KT_ref, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("use_box", [True, False])
def test_metropolis_forward_cached_parallel(device, use_box):
    torch.manual_seed(3)
    B, C, H, W, R = 2, 3, 7, 8, 1
    x = torch.randn(B, C, H, W, device=device)
    shifts = [
        (0, 0, False), (0, 1, True), (1, -1, True),
        (1, 0, True), (1, 1, True),
    ]
    box = _box(B, C, H, W, R, device)
    weights = []
    for dx, dy, _ in shifts:
        w = 0.08 * torch.rand_like(x)
        if use_box:
            w = w * box[:, :, R + dx:R + dx + H, R + dy:R + dy + W]
        weights.append(w)
    K_hat_x, degree_hat = _symmetric_serial(
        x, weights, shifts, R, box, use_box)
    cache = SimpleNamespace(
        C=C, shifts=shifts, w_hat_fwd_list=weights,
        comp_box=use_box, degree_hat=degree_hat)
    model = SimpleNamespace()
    out, returned_degree = mp_patch._forward_cached_cuda(
        model, x, cache, True)
    reference = x - degree_hat * x + K_hat_x
    torch.testing.assert_close(out, reference, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(returned_degree, degree_hat)
