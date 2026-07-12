# tests/test_gadsd_moments_patch.py
#
# Equivalence test for the GADSDMoments integration patch.
#
# The ops layer dispatches CUDA tensors to the compiled kernels and CPU
# tensors to the bit-exact reference implementations, and the patch code is
# device-agnostic. So the patch's MATH (chunked shift_gather stack build,
# descriptor alignment, head/mix/finalize plumbing, fused scalar
# accumulation, D4 handling, gradients) is validated here in fp64 on CPU by
# calling the patched paths directly; on a CUDA machine the same tests also
# run end-to-end through the compiled kernels.

import importlib.util
import os
import pathlib

import pytest
import torch

from neural_shift_cuda.integration.gadsd_moments_patch import (
    _forward_cuda,
    _transform_weights_cuda,
    install_cuda_shift,
)

CUDA_OK = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA_OK, reason="CUDA not available")

_CANDIDATES = [
    os.environ.get("GADSD_MOMENTS_PATH", ""),
    "GADSD_moments.py",
    str(pathlib.Path(__file__).resolve().parent.parent.parent /
        "GADSD_moments.py"),
]


def _load_module():
    for p in _CANDIDATES:
        if p and os.path.exists(p):
            spec = importlib.util.spec_from_file_location("gadsd_moments", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    pytest.skip("GADSD_moments.py not found; "
                "set GADSD_MOMENTS_PATH to enable.")


def _make_model(mod, **kw):
    seed = kw.pop("seed", 0)
    defaults = dict(in_channels=3, window_rad=2)
    defaults.update(kw)
    model = mod.GADSDMoments(**defaults).double()
    torch.manual_seed(seed)
    # Randomize so per-channel weights genuinely vary across configs.
    for p in model.parameters():
        torch.nn.init.normal_(p, std=0.5)
    return model


def _data(C, H, W, B=2, seed=1):
    g = torch.Generator().manual_seed(seed)
    x = torch.rand(B, C, H, W, dtype=torch.float64, generator=g)
    guide = torch.rand(B, C, H, W, dtype=torch.float64, generator=g)
    sig = torch.rand(B, dtype=torch.float64, generator=g) * 50.0 / 255.0
    return x, guide, sig


CONFIGS = [
    # (name, model kwargs, spatial)
    ("default_softmax", dict(), (12, 14)),
    ("control_sigmoid_avg",
     dict(output_activation="control_sigmoid", stochasticize="avg",
          epsilon_floor=0.05), (12, 14)),
    ("heads4_concat",
     dict(n_heads=4, head_mix="concat", head_mix_pos_act="softplus",
          use_dist=False), (12, 14)),
    ("heads4_broadcast", dict(n_heads=4, head_mix="broadcast_or_mean"), (12, 14)),
    ("shared_weights", dict(weight_sharing="shared", n_heads=2), (12, 14)),
    ("fixed_moments", dict(weight_arch="fixed_moments"), (12, 14)),
    ("no_descriptor", dict(use_transform_descriptor=False), (12, 14)),
    ("pool_max", dict(shift_weight_pool="max"), (12, 14)),
    ("chunked_small", dict(max_batch_shifts=3), (12, 14)),
    ("chunked_auto_tiny", dict(auto_chunk_elements=4096), (12, 14)),
    ("d4_family", dict(transform_family="d4"), (12, 12)),
    ("translations_d4",
     dict(transform_family="translations+d4", window_rad=1,
          max_batch_shifts=4), (12, 12)),  # mixed chunks at the boundary
    ("stride2", dict(translation_stride=2, window_rad=2), (14, 14)),
    ("grad_ckpt", dict(use_grad_checkpoint=True), (12, 14)),
]


@pytest.mark.parametrize("name,kw,hw", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_forward_equivalence_fp64(name, kw, hw):
    mod = _load_module()
    model = _make_model(mod, **kw).eval()
    x, guide, sig = _data(3, *hw)

    with torch.no_grad():
        ref, ref_D = model(x, guide, sig=sig, return_D=True)
        out, out_D = _forward_cuda(model, x, guide=guide, sig=sig,
                                   return_D=True)

    err = (ref - out).abs().max().item()
    assert err < 1e-12, f"{name}: forward mismatch {err:.3e}"
    assert torch.equal(ref_D, out_D)


@pytest.mark.parametrize("name,kw,hw", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_transform_weights_equivalence_fp64(name, kw, hw):
    mod = _load_module()
    model = _make_model(mod, **kw).eval()
    _, guide, sig = _data(3, *hw)

    with torch.no_grad():
        sig_n = mod._normalise_sigma(sig, guide)
        phi = model.pre_activation(guide, sigma=sig_n)
        raw_ref = model._transform_weights(phi, sig_n, 3)
        raw_pat = _transform_weights_cuda(model, phi, sig_n, 3)

    err = (raw_ref - raw_pat).abs().max().item()
    assert err < 1e-12, f"{name}: raw score mismatch {err:.3e}"


@pytest.mark.parametrize("ckpt", [False, True], ids=["plain", "grad_ckpt"])
def test_gradient_equivalence_fp64(ckpt):
    mod = _load_module()
    x, guide, sig = _data(3, 12, 14)

    grads = []
    for use_patched in (False, True):
        model = _make_model(mod, use_grad_checkpoint=ckpt, seed=0).train()
        xi = x.clone().requires_grad_(True)
        if use_patched:
            out, _ = _forward_cuda(model, xi, guide=guide, sig=sig)
        else:
            out, _ = model(xi, guide, sig=sig)
        (out.square().sum()).backward()
        grads.append((
            xi.grad.clone(),
            [p.grad.clone() for p in model.parameters()],
        ))

    gx_err = (grads[0][0] - grads[1][0]).abs().max().item()
    assert gx_err < 1e-12, f"x-grad mismatch {gx_err:.3e}"
    for gr, gp in zip(grads[0][1], grads[1][1]):
        gerr = (gr - gp).abs().max().item()
        assert gerr < 1e-12, f"param-grad mismatch {gerr:.3e}"


def test_installer_routing_and_idempotence():
    mod = _load_module()
    cls = mod.GADSDMoments
    orig_forward = cls.forward
    orig_tw = cls._transform_weights
    try:
        install_cuda_shift(cls)
        assert cls._cuda_shift_installed
        first_forward = cls.forward
        install_cuda_shift(cls)                       # idempotent
        assert cls.forward is first_forward

        # CPU tensors must route to the ORIGINAL methods (kernel path is
        # CUDA-only); output must match a pristine reference bit-for-bit.
        model = _make_model(mod).eval()
        x, guide, sig = _data(3, 12, 14)
        with torch.no_grad():
            out, _ = model(x, guide, sig=sig)
        cls.forward = orig_forward
        cls._transform_weights = orig_tw
        ref_model = _make_model(mod).eval()
        with torch.no_grad():
            ref, _ = ref_model(x, guide, sig=sig)
        assert torch.equal(out, ref)
    finally:
        cls.forward = orig_forward
        cls._transform_weights = orig_tw
        if hasattr(cls, "_cuda_shift_installed"):
            del cls._cuda_shift_installed


def test_installer_rejects_wrong_architecture():
    class NotMoments(torch.nn.Module):
        def forward(self, x):
            return x

    with pytest.raises(AttributeError):
        install_cuda_shift(NotMoments)


@requires_cuda
@pytest.mark.parametrize("name,kw,hw", CONFIGS, ids=[c[0] for c in CONFIGS])
def test_forward_equivalence_cuda_fp64(name, kw, hw):
    """End-to-end through the compiled kernels; fp64 so the comparison to the
    CPU-reference forward is tight."""
    mod = _load_module()
    model = _make_model(mod, **kw).eval()
    x, guide, sig = _data(3, *hw)

    with torch.no_grad():
        ref, _ = model(x, guide, sig=sig)

    install_cuda_shift(mod.GADSDMoments)
    model_c = model.cuda()
    with torch.no_grad():
        out, _ = model_c(x.cuda(), guide.cuda(), sig=sig.cuda())

    err = (ref - out.cpu()).abs().max().item()
    assert err < 1e-10, f"{name}: CUDA forward mismatch {err:.3e}"
