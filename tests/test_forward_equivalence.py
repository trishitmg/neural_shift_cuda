# tests/test_forward_equivalence.py
#
# End-to-end test: instantiate the user's `nekre` model, run the original
# batched=True forward path, then monkey-patch the model to use the CUDA
# shift_gather / pair_gather operators, and confirm U and Z match.

import argparse
import importlib.util
import os
import pathlib
import sys
import types

import pytest
import torch
import torch.nn.functional as F

from neural_shift_cuda import shift_gather, pair_gather, accumulate_uz
from neural_shift_cuda import accumulate_uz_reference

CUDA_OK = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA_OK, reason="CUDA not available")


# ---------------------------------------------------------------------------
# Load the user's nekre module dynamically.
# We look for it on a few common paths so the test runs both inside the
# extension repo and standalone.
# ---------------------------------------------------------------------------

_CANDIDATES = [
    os.environ.get("NKD_MODELS_PATH", ""),
    "NKD_models_symm.py",
    "NKD_models_symm__2_.py",
    "/mnt/user-data/uploads/NKD_models_symm__2_.py",
    str(pathlib.Path(__file__).resolve().parent.parent.parent /
        "NKD_models_symm.py"),
]


def _load_nekre():
    for p in _CANDIDATES:
        if p and os.path.exists(p):
            spec = importlib.util.spec_from_file_location("nkd_models_symm", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    pytest.skip("NKD_models_symm.py not found; set NKD_MODELS_PATH to enable.")


def _make_args(model_type="concat",
               residual_depth=2,
               proj_depth=1,
               latent_dim=8,
               window_rad=2,
               in_channel=3,
               patch_rad=1,
               output_activation="exp",
               blind=True):
    ns = argparse.Namespace()
    ns.model_type = model_type
    ns.residual_depth = residual_depth
    ns.proj_depth = proj_depth
    ns.latent_dim = latent_dim
    ns.window_rad = window_rad
    ns.in_channel = in_channel
    ns.patch_rad = patch_rad
    ns.output_activation = output_activation
    ns.blind = blind
    return ns


# ---------------------------------------------------------------------------
# (5) test_model_forward_equivalence
# ---------------------------------------------------------------------------

def _shift_tensor(model, device):
    rows = []
    for dx, dy, hi in model._collect_shifts():
        rows.append((dx, dy, int(bool(hi))))
    return torch.tensor(rows, dtype=torch.int32, device=device)


def _new_forward_concat(model, x, sig=None, guide=None):
    """CUDA-accelerated equivalent of forward(..., batched=True) for model_type='concat'."""
    B, C, H, W = x.shape
    if guide is None:
        guide = model.pre_activation(x, sigma=sig)
    else:
        guide = model.pre_activation(guide, sigma=sig)
    shifts = _shift_tensor(model, x.device)

    pair_batch, mask_batch = pair_gather(guide.contiguous(), shifts)
    weights = model.compute_weights_from_pair(pair_batch)
    weights = weights * mask_batch        # broadcast (S*B, 1, H, W) -> (S*B, C, H, W)

    # Accumulate via either CUDA accumulate_uz or the reference impl.
    U_num, Z = accumulate_uz(x.contiguous(), weights.contiguous(), shifts)
    U = U_num / Z
    return U, Z


def _compute_weights_from_pair(model, pair_batch):
    """Mirror of compute_weights but skipping the torch.cat (pair already cat'd)."""
    x = pair_batch
    from einops import rearrange
    for proj in model.proj:
        for i, sub_layer in enumerate(proj):
            if i == 1:  # LayerNorm
                x = rearrange(x, 'b c h w -> b h w c')
                x = sub_layer(x)
                x = rearrange(x, 'b h w c -> b c h w')
            else:
                x = sub_layer(x)
    x = model.out(x)
    if model.output_activation == 'exp':
        x = torch.exp(x / model.sigma ** 2)
    elif model.output_activation == 'control_sigmoid':
        k, a, b = torch.chunk(x, dim=1, chunks=3)
        x = model.controlled_sigmoid(k, a, b) + 1e-8
    elif model.output_activation == 'sig':
        normalization_factor = -(model.sigma * model.sigma)
        return torch.nn.functional.sigmoid(k / normalization_factor)
    return x


@requires_cuda
@pytest.mark.parametrize("R", [1, 2])
def test_model_forward_equivalence(R):
    nkd = _load_nekre()
    torch.manual_seed(0)
    args = _make_args(window_rad=R, output_activation='exp', model_type='concat')
    model = nkd.nekre(args, device='cuda').to('cuda').eval()

    # Bind the helper used by the new path
    model.compute_weights_from_pair = types.MethodType(_compute_weights_from_pair, model)

    B, C, H, W = 2, args.in_channel, 24, 20
    x = torch.randn(B, C, H, W, device='cuda', dtype=torch.float32)

    with torch.no_grad():
        U_old, Z_old = model.forward(x, batched=True)
        U_new, Z_new = _new_forward_concat(model, x)

    torch.testing.assert_close(U_new, U_old, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(Z_new, Z_old, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# (6) test_training_backward
# ---------------------------------------------------------------------------

@requires_cuda
def test_training_backward():
    nkd = _load_nekre()
    torch.manual_seed(0)
    args = _make_args(window_rad=1, output_activation='exp', model_type='concat')
    model = nkd.nekre(args, device='cuda').to('cuda').train()
    model.compute_weights_from_pair = types.MethodType(_compute_weights_from_pair, model)

    B, C, H, W = 1, args.in_channel, 16, 16
    x = torch.randn(B, C, H, W, device='cuda', dtype=torch.float32,
                    requires_grad=True)

    U, Z = _new_forward_concat(model, x)
    loss = U.sum() + Z.sum()
    loss.backward()

    # Every named parameter should receive a finite gradient.
    n_with_grad = 0
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
        if p.grad.abs().sum().item() > 0:
            n_with_grad += 1
    # at least the projection and output convs (and pre_activation layers) should fire
    assert n_with_grad > 0
    assert x.grad is not None and torch.isfinite(x.grad).all()
