# tests/test_moments_patches.py
#
# Equivalence tests for the moments-branch integration patches:
#   * nekde_moments_patch  (NeKDeMoments, NKD_moments)
#   * gasd_moments_patch   (GASDMoments, GASD_moments)
#   * nkd_mp_moments_patch (NeKDeMetropolisMoments, NKD_mp_moments)
#
# The ops layer dispatches CUDA tensors to the compiled kernels and CPU
# tensors to the bit-exact reference implementations, and the patch code is
# device-agnostic. So the patch MATH (chunked shift_gather weight build,
# descriptor alignment, joint softmax, head mixing, fused U/Z or Metropolis
# accumulation, comp_box both ways, gradients) is validated here in fp64 on
# CPU by calling the patched paths directly; on a CUDA machine the same
# tests also run end-to-end through the compiled kernels.

import importlib.util
import os
import pathlib

import pytest
import torch

from neural_shift_cuda.integration import nekde_moments_patch
from neural_shift_cuda.integration import gasd_moments_patch
from neural_shift_cuda.integration import nkd_mp_moments_patch

CUDA_OK = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA_OK, reason="CUDA not available")

_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent


def _load(env_var, fname, modname):
    candidates = [os.environ.get(env_var, ""), fname, str(_ROOT / fname)]
    for p in candidates:
        if p and os.path.exists(p):
            spec = importlib.util.spec_from_file_location(modname, p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    pytest.skip(f"{fname} not found; set {env_var} to enable.")


def _load_nekde():
    return _load("NKD_MOMENTS_PATH", "NKD_moments.py", "nkd_moments")


def _load_gasd():
    return _load("GASD_MOMENTS_PATH", "GASD_moments.py", "gasd_moments")


def _load_mp():
    return _load("NKD_MP_MOMENTS_PATH", "NKD_mp_moments.py", "nkd_mp_moments")


def _make_model(cls, seed=0, **kw):
    defaults = dict(in_channels=3, window_rad=2)
    defaults.update(kw)
    model = cls(**defaults).double()
    torch.manual_seed(seed)
    # Randomize so the weights genuinely vary across configs.
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.5)
    return model


def _data(C, H, W, B=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(B, C, H, W, dtype=torch.float64, generator=g)
    guide = torch.rand(B, C, H, W, dtype=torch.float64, generator=g)
    sig = torch.rand(B, 1, 1, 1, dtype=torch.float64, generator=g) * 50.0 / 255.0
    return x, guide, sig


CONFIGS = [
    # (name, model kwargs, spatial)
    ("softmax_default", dict(output_activation="softmax", use_dist=True), (12, 14)),
    ("control_sigmoid", dict(output_activation="control_sigmoid",
                             use_dist=False), (12, 14)),
    ("cs_with_dist", dict(output_activation="control_sigmoid",
                          use_dist=True), (12, 14)),
    ("heads4_concat", dict(output_activation="softmax", n_heads=4,
                           head_mix="concat", head_mix_pos_act="softplus",
                           use_dist=True), (12, 14)),
    ("heads4_broadcast", dict(output_activation="softmax", n_heads=4,
                              head_mix="broadcast_or_mean",
                              use_dist=True), (12, 14)),
    ("fixed_moments", dict(weight_arch="fixed_moments",
                           output_activation="softmax"), (12, 14)),
    ("no_descriptor", dict(use_transform_descriptor=False,
                           output_activation="softmax"), (12, 14)),
    ("layer_norm", dict(layer_norm=True, output_activation="softmax"), (12, 14)),
    ("chunked_small", dict(max_batch_shifts=3,
                           output_activation="softmax"), (12, 14)),
    ("no_comp_box", dict(comp_box=False, output_activation="softmax"), (12, 14)),
    ("grad_ckpt", dict(use_grad_checkpoint=True,
                       output_activation="softmax"), (12, 14)),
]

_IDS = [c[0] for c in CONFIGS]


# ---------------------------------------------------------------------------
# NeKDeMoments (half-plane, symmetric U/Z)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,kw,hw", CONFIGS, ids=_IDS)
def test_nekde_forward_equivalence_fp64(name, kw, hw):
    mod = _load_nekde()
    model = _make_model(mod.NeKDeMoments, **kw).eval()
    x, guide, sig = _data(3, *hw)

    with torch.no_grad():
        ref, ref_Z = model(x, guide, sig=sig, return_D=True)
        out, out_Z = nekde_moments_patch._forward_cuda(
            model, x, guide=guide, sig=sig, return_D=True)

    err = (ref - out).abs().max().item()
    zerr = (ref_Z - out_Z).abs().max().item()
    assert err < 1e-12, f"{name}: forward mismatch {err:.3e}"
    assert zerr < 1e-12, f"{name}: Z mismatch {zerr:.3e}"


# ---------------------------------------------------------------------------
# GASDMoments (full window, row-stochastic)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,kw,hw", CONFIGS, ids=_IDS)
def test_gasd_forward_equivalence_fp64(name, kw, hw):
    mod = _load_gasd()
    model = _make_model(mod.GASDMoments, **kw).eval()
    x, guide, sig = _data(3, *hw)

    with torch.no_grad():
        ref, ref_Z = model(x, guide, sig=sig, return_D=True)
        out, out_Z = gasd_moments_patch._forward_cuda(
            model, x, guide=guide, sig=sig, return_D=True)

    err = (ref - out).abs().max().item()
    zerr = (ref_Z - out_Z).abs().max().item()
    assert err < 1e-12, f"{name}: forward mismatch {err:.3e}"
    assert zerr < 1e-12, f"{name}: Z mismatch {zerr:.3e}"


# ---------------------------------------------------------------------------
# NeKDeMetropolisMoments (Metropolis-symmetrized)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("name,kw,hw", CONFIGS, ids=_IDS)
def test_mp_forward_equivalence_fp64(name, kw, hw):
    mod = _load_mp()
    model = _make_model(mod.NeKDeMetropolisMoments, **kw).eval()
    x, guide, sig = _data(3, *hw)

    with torch.no_grad():
        ref, ref_D = model(x, guide, sig=sig, return_D=True)
        out, out_D = nkd_mp_moments_patch._forward_cuda(
            model, x, guide=guide, sig=sig, return_D=True)

    err = (ref - out).abs().max().item()
    derr = (ref_D - out_D).abs().max().item()
    assert err < 1e-12, f"{name}: forward mismatch {err:.3e}"
    assert derr < 1e-12, f"{name}: degree_hat mismatch {derr:.3e}"


# ---------------------------------------------------------------------------
# Gradient equivalence (one representative config per family)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["nekde", "gasd", "mp"])
def test_gradient_equivalence_fp64(family):
    if family == "nekde":
        mod = _load_nekde()
        cls, fwd = mod.NeKDeMoments, nekde_moments_patch._forward_cuda
    elif family == "gasd":
        mod = _load_gasd()
        cls, fwd = mod.GASDMoments, gasd_moments_patch._forward_cuda
    else:
        mod = _load_mp()
        cls, fwd = mod.NeKDeMetropolisMoments, nkd_mp_moments_patch._forward_cuda

    x, guide, sig = _data(3, 12, 14)
    kw = dict(output_activation="softmax", use_dist=True, n_heads=2,
              head_mix="concat", head_mix_pos_act="softplus")

    grads = []
    for use_patched in (False, True):
        model = _make_model(cls, seed=0, **kw).train()
        xi = x.clone().requires_grad_(True)
        if use_patched:
            out, _ = fwd(model, xi, guide=guide, sig=sig)
        else:
            out, _ = model(xi, guide, sig=sig)
        (out.square().sum()).backward()
        grads.append((
            xi.grad.clone(),
            [p.grad.clone() for p in model.parameters()],
        ))

    gx_err = (grads[0][0] - grads[1][0]).abs().max().item()
    assert gx_err < 1e-11, f"{family}: x-grad mismatch {gx_err:.3e}"
    for gr, gp in zip(grads[0][1], grads[1][1]):
        gerr = (gr - gp).abs().max().item()
        assert gerr < 1e-11, f"{family}: param-grad mismatch {gerr:.3e}"


# ---------------------------------------------------------------------------
# Installer routing / idempotence / architecture rejection
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("family", ["nekde", "gasd", "mp"])
def test_installer_routing_and_idempotence(family):
    if family == "nekde":
        mod = _load_nekde()
        cls, installer = mod.NeKDeMoments, nekde_moments_patch.install_cuda_shift
        flag = "_cuda_shift_installed"
    elif family == "gasd":
        mod = _load_gasd()
        cls, installer = mod.GASDMoments, gasd_moments_patch.install_cuda_shift
        flag = "_cuda_shift_installed"
    else:
        mod = _load_mp()
        cls, installer = (mod.NeKDeMetropolisMoments,
                          nkd_mp_moments_patch.install_cuda_shift)
        flag = "_metropolis_cuda_forward_installed"

    orig_forward = cls.forward
    try:
        installer(cls)
        assert getattr(cls, flag)
        first_forward = cls.forward
        installer(cls)                                # idempotent
        assert cls.forward is first_forward

        # CPU tensors must route to the ORIGINAL forward (kernel path is
        # CUDA-only); output must match a pristine reference bit-for-bit.
        model = _make_model(cls, output_activation="softmax",
                            use_dist=True).eval()
        x, guide, sig = _data(3, 12, 14)
        with torch.no_grad():
            out, _ = model(x, guide, sig=sig)
        cls.forward = orig_forward
        ref_model = _make_model(cls, output_activation="softmax",
                                use_dist=True).eval()
        with torch.no_grad():
            ref, _ = ref_model(x, guide, sig=sig)
        assert torch.equal(out, ref)
    finally:
        cls.forward = orig_forward
        if hasattr(cls, flag):
            delattr(cls, flag)


@pytest.mark.parametrize("installer", [
    nekde_moments_patch.install_cuda_shift,
    gasd_moments_patch.install_cuda_shift,
    nkd_mp_moments_patch.install_cuda_shift,
])
def test_installer_rejects_wrong_architecture(installer):
    class NotMoments(torch.nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(AttributeError):
        installer(NotMoments)


# ---------------------------------------------------------------------------
# CUDA end-to-end (same tests through the compiled kernels)
# ---------------------------------------------------------------------------

@requires_cuda
@pytest.mark.parametrize("family", ["nekde", "gasd", "mp"])
def test_forward_equivalence_cuda_fp64(family):
    """End-to-end through the compiled kernels; fp64 so the comparison to the
    CPU-reference forward is tight."""
    if family == "nekde":
        mod = _load_nekde()
        cls, fwd = mod.NeKDeMoments, nekde_moments_patch._forward_cuda
    elif family == "gasd":
        mod = _load_gasd()
        cls, fwd = mod.GASDMoments, gasd_moments_patch._forward_cuda
    else:
        mod = _load_mp()
        cls, fwd = mod.NeKDeMetropolisMoments, nkd_mp_moments_patch._forward_cuda

    model = _make_model(cls, output_activation="softmax", use_dist=True).eval()
    x, guide, sig = _data(3, 12, 14)

    with torch.no_grad():
        ref, _ = model(x, guide, sig=sig, return_D=True)
        out, _ = fwd(model.cuda(), x.cuda(), guide=guide.cuda(),
                     sig=sig.cuda(), return_D=True)

    err = (ref - out.cpu()).abs().max().item()
    assert err < 1e-10, f"{family}: CUDA forward mismatch {err:.3e}"
