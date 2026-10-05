"""GASD (both shift_modes) and NKD_mp_dss (kernel_modes x construction / scale_mode / diag_correction)
patch paths vs the models' own PyTorch reference methods.

Model files are loaded by path:

    NSC_MODEL_DIR=/path/to/core/models pytest -q tests/test_gasd_dss_patches.py

(default /mnt/user-data/uploads; a missing file or a failing import, e.g. no
mamba_ssm for the Mamba variants, skips that case). The patch functions are
called directly, so on CPU they run through the reference ops and check the
patch maths without a GPU; on CUDA they run the compiled kernels.
"""

import functools
import importlib.util
import os

import pytest
import torch

from neural_shift_cuda.integration import gasd_drunet_attn_patch as gasd_patch
from neural_shift_cuda.integration import nkd_metropolis_dss_attn_patch as dss_patch

MODEL_DIR = os.environ.get("NSC_MODEL_DIR", "/mnt/user-data/uploads")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
TOL = dict(rtol=1e-10, atol=1e-12)                         # float64 throughout

_BASE = dict(in_channels=3, window_rad=2, feat_ch=16, qk_ch=8, n_heads=2)
_DRUNET = dict(drunet_nc=(8, 16), drunet_nb=1)
_MAMBA = dict(mamba_nc=(16, 32), mamba_nb=1, d_state=4, chunk_size=16)

GASD_ARCHS = {
    "drunet":   ("GASD_drunet_attn_v2.py", "GASDDRUNetAttn", _DRUNET),
    "featattn": ("GASD_drunet_featattn_v2.py", "GASDDRUNetFeatAttn",
                 dict(_DRUNET, fe_bottleneck_heads=4)),
    "mamba":    ("GASD_mamba_attn.py", "GASDMambaAttn", _MAMBA),
}
DSS_ARCHS = {
    "drunet":   ("NKD_mp_dss_drunet_attn_v2.py", "NeKDeDSSDRUNetAttn", _DRUNET),
    "featattn": ("NKD_mp_dss_drunet_featattn_v2.py", "NeKDeDSSDRUNetFeatAttn",
                 dict(_DRUNET, fe_bottleneck_heads=4)),
    "mamba":    ("NKD_mp_dss_mamba_attn.py", "NeKDeDSSMambaAttn", _MAMBA),
}


@functools.lru_cache(maxsize=None)
def _load(fname, cls_name, installer):
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        return None, f"{path} not found"
    spec = importlib.util.spec_from_file_location(f"_{fname[:-3]}_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:
        return None, f"{fname}: {exc}"
    cls = getattr(mod, cls_name)
    installer(cls)
    return cls, None


def _model(table, arch, installer, device, **kw):
    fname, cls_name, extra = table[arch]
    cls, why = _load(fname, cls_name, installer)
    if cls is None:
        pytest.skip(why)
    torch.manual_seed(0)
    return cls(**_BASE, **extra, **kw).to(device=device, dtype=torch.float64).eval()


def _inputs(device):
    g = torch.Generator().manual_seed(1)
    x, z, y = (torch.randn(2, 3, 12, 12, generator=g, dtype=torch.float64).to(device)
               for _ in range(3))
    return x, z, y, torch.tensor([0.1, 0.2], dtype=torch.float64, device=device)


def _ref(model, fn):
    model.use_cuda_shift = False
    try:
        return fn()
    finally:
        model.use_cuda_shift = True


def _grads(model, x, fn):
    model.zero_grad(set_to_none=True)
    xg = x.detach().clone().requires_grad_(True)
    fn(xg).square().sum().backward()
    return [xg.grad] + [p.grad for p in model.parameters()]


# ---------------------------------------------------------------------------
# GASD
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shift_mode", ["inverse_pair", "legacy"])
@pytest.mark.parametrize("arch", list(GASD_ARCHS))
def test_gasd_patch_matches_reference(device, shift_mode, arch):
    m = _model(GASD_ARCHS, arch, gasd_patch.install_cuda_shift, device,
               shift_mode=shift_mode)
    x, z, y, sig = _inputs(device)

    U, Z = gasd_patch._forward_cuda(m, x, guide=z, sig=sig, return_D=True)
    U_r, Z_r = _ref(m, lambda: m.forward(x, z, sig=sig, return_D=True))
    torch.testing.assert_close(U, U_r, **TOL)
    torch.testing.assert_close(Z, Z_r, **TOL)
    torch.testing.assert_close(gasd_patch._kt_action_cuda(m, y, guide=z, sig=sig),
                               _ref(m, lambda: m._KT_action(y, z, sig)), **TOL)
    WT, D = gasd_patch._nlm_transpose_cuda(m, x, guide=z, sig=sig, return_D=True)
    WT_r, D_r = _ref(m, lambda: m.NLM_transpose(x, z, sig=sig, return_D=True))
    torch.testing.assert_close(WT, WT_r, **TOL)
    torch.testing.assert_close(D, D_r, **TOL)
    torch.testing.assert_close(gasd_patch._laplacian_grw_cuda(m, x, z, sig=sig),
                               _ref(m, lambda: m.laplacian_grw(x, z, sig=sig)), **TOL)

    cache = m.build_weight_cache(z, sig)
    torch.testing.assert_close(gasd_patch._forward_cached_cuda(m, x, cache, True)[0],
                               _ref(m, lambda: m.forward_cached(x, cache)[0]), **TOL)
    torch.testing.assert_close(gasd_patch._kt_action_cached_cuda(m, y, cache),
                               _ref(m, lambda: m._KT_action_cached(y, cache)), **TOL)
    torch.testing.assert_close(gasd_patch._laplacian_grw_cached_cuda(m, x, cache),
                               _ref(m, lambda: m.laplacian_grw_cached(x, cache)), **TOL)

    m.train()
    got = _grads(m, x, lambda v: gasd_patch._forward_cuda(m, v, guide=z, sig=sig)[0])
    want = _ref(m, lambda: _grads(m, x, lambda v: m.forward(v, z, sig=sig)[0]))
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, **TOL)


# ---------------------------------------------------------------------------
# NKD_mp_dss
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("construction,scale_mode,diag_correction", [
    ("metropolis", "local", False), ("metropolis", "local", True),
    ("metropolis", "global", False), ("metropolis", "global", True),
    ("optimal_transport", "global", False)])
@pytest.mark.parametrize("kernel_mode,shift_mode", [
    ("asymmetric", "inverse_pair"), ("symmetric", "inverse_pair"), ("asymmetric", "legacy")])
@pytest.mark.parametrize("arch", list(DSS_ARCHS))
def test_dss_patch_matches_reference(device, kernel_mode, shift_mode, construction, scale_mode,
                                      diag_correction, arch):
    m = _model(DSS_ARCHS, arch, dss_patch.install_cuda_shift, device,
               kernel_mode=kernel_mode, shift_mode=shift_mode, construction=construction,
               diag_correction=diag_correction, scale_mode=scale_mode)
    x, z, y, sig = _inputs(device)

    Wx, We = dss_patch._forward_cuda(m, x, guide=z, sig=sig, return_D=True)
    Wx_r, We_r = _ref(m, lambda: m.forward(x, z, sig=sig, return_D=True))
    torch.testing.assert_close(Wx, Wx_r, **TOL)
    torch.testing.assert_close(We, We_r, **TOL)
    torch.testing.assert_close(dss_patch._adjoint_cuda(m, y, guide=z, sig=sig),
                               _ref(m, lambda: m.adjoint(y, z, sig=sig)), **TOL)
    torch.testing.assert_close(dss_patch._laplacian_grw_cuda(m, x, z, sig=sig),
                               _ref(m, lambda: m.laplacian_grw(x, z, sig=sig)), **TOL)

    cache = m.build_weight_cache(z, sig)
    Wc, Wec = dss_patch._forward_cached_cuda(m, x, cache, return_D=True)
    Wc_r, Wec_r = _ref(m, lambda: m.forward_cached(x, cache, return_D=True))
    torch.testing.assert_close(Wc, Wc_r, **TOL)
    torch.testing.assert_close(Wec, Wec_r, **TOL)
    torch.testing.assert_close(dss_patch._adjoint_cached_cuda(m, y, cache),
                               _ref(m, lambda: m.adjoint_cached(y, cache)), **TOL)

    m.train()  # checkpointed operator build on the patch path
    got = _grads(m, x, lambda v: dss_patch._forward_cuda(m, v, guide=z, sig=sig)[0])
    want = _ref(m, lambda: _grads(m, x, lambda v: m.forward(v, z, sig=sig)[0]))
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, **TOL)
    got = _grads(m, x, lambda v: dss_patch._laplacian_grw_cuda(m, v, z, sig=sig))
    want = _ref(m, lambda: _grads(m, x, lambda v: m.laplacian_grw(v, z, sig=sig)))
    for a, b in zip(got, want):
        torch.testing.assert_close(a, b, **TOL)


@pytest.mark.parametrize("kernel_mode", ["asymmetric", "symmetric"])
def test_dss_installer_routes_cpu_to_reference(kernel_mode):
    m = _model(DSS_ARCHS, "drunet", dss_patch.install_cuda_shift, "cpu",
               kernel_mode=kernel_mode)
    x, z, y, sig = _inputs("cpu")
    assert type(m)._dss_cuda_installed
    # CPU tensors never take the CUDA path, whatever use_cuda_shift says.
    torch.testing.assert_close(m.forward(x, z, sig=sig)[0],
                               _ref(m, lambda: m.forward(x, z, sig=sig)[0]), rtol=0, atol=0)
