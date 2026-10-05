"""GASD (shift_modes x direction_modes) and NKD_mp_dss (kernel_modes x
direction_modes x construction / scale_mode / diag_correction) patch paths vs
the models' own PyTorch reference methods, plus the direction_mode
equivalences (directional == legacy for the same weights; gated at a zero gate
== the symmetric raw kernel).

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


def _model(table, arch, installer, device, active_gate=True, **kw):
    fname, cls_name, extra = table[arch]
    cls, why = _load(fname, cls_name, installer)
    if cls is None:
        pytest.skip(why)
    kw = {k: v for k, v in kw.items() if v is not None}  # None -> the model's default
    torch.manual_seed(0)
    m = cls(**_BASE, **extra, **kw).to(device=device, dtype=torch.float64).eval()
    gate = getattr(m, "direction_gate", None)
    if gate is not None and active_gate:
        # The gate's output projection starts at zero (g = 1/2, the symmetric
        # kernel); randomise it so the directed path is actually exercised.
        g = torch.Generator().manual_seed(2)
        with torch.no_grad():
            for p in gate.parameters():
                p.copy_(0.5 * torch.randn(p.shape, generator=g, dtype=p.dtype))
    return m


def _twin(table, arch, installer, device, src, **kw):
    """A second model of the same arch whose shared parameters are copied from
    `src` (parameters only one of the two has, i.e. the gate, are left out)."""
    m = _model(table, arch, installer, device, **kw)
    missing, unexpected = m.load_state_dict(src.state_dict(), strict=False)
    assert all(k.startswith("direction_gate.") for k in missing + unexpected), (missing, unexpected)
    return m


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

_DIRECTED = ["directional", "gated", "gated_potential"]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shift_mode,direction_mode",
                         [("inverse_pair", d) for d in _DIRECTED] + [("legacy", "directional")])
@pytest.mark.parametrize("arch", list(GASD_ARCHS))
def test_gasd_patch_matches_reference(device, shift_mode, direction_mode, arch):
    m = _model(GASD_ARCHS, arch, gasd_patch.install_cuda_shift, device,
               shift_mode=shift_mode, direction_mode=direction_mode)
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


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("arch", list(GASD_ARCHS))
def test_gasd_directional_matches_legacy(device, arch):
    """Same weights, no new parameters: inverse_pair + directional == legacy."""
    leg = _model(GASD_ARCHS, arch, gasd_patch.install_cuda_shift, device, shift_mode="legacy")
    dirn = _twin(GASD_ARCHS, arch, gasd_patch.install_cuda_shift, device, leg,
                 direction_mode="directional")
    assert sum(p.numel() for p in dirn.parameters()) == sum(p.numel() for p in leg.parameters())
    x, z, y, sig = _inputs(device)
    for a, b in zip(gasd_patch._forward_cuda(dirn, x, guide=z, sig=sig, return_D=True),
                    gasd_patch._forward_cuda(leg, x, guide=z, sig=sig, return_D=True)):
        torch.testing.assert_close(a, b, **TOL)
    torch.testing.assert_close(gasd_patch._kt_action_cuda(dirn, y, guide=z, sig=sig),
                               gasd_patch._kt_action_cuda(leg, y, guide=z, sig=sig), **TOL)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("direction_mode", ["gated", "gated_potential"])
@pytest.mark.parametrize("arch", list(GASD_ARCHS))
def test_gasd_zero_gate_is_symmetric(device, direction_mode, arch):
    """At init (zero gate logits, g = 1/2) a+ = b and a- = P_{pi^{-1}} b, so the
    raw kernel is D_id + sum (D_pi P_pi + P_{pi^{-1}} D_pi) = K^T."""
    m = _model(GASD_ARCHS, arch, gasd_patch.install_cuda_shift, device, active_gate=False,
               direction_mode=direction_mode)
    x, z, y, sig = _inputs(device)
    U, D = gasd_patch._forward_cuda(m, y, guide=z, sig=sig, return_D=True)
    torch.testing.assert_close(gasd_patch._kt_action_cuda(m, y, guide=z, sig=sig), U * D, **TOL)


# ---------------------------------------------------------------------------
# NKD_mp_dss
# ---------------------------------------------------------------------------

_DSS_ROUTES = [
    ("metropolis", "local", False), ("metropolis", "local", True),
    ("metropolis", "global", False), ("metropolis", "global", True),
    ("optimal_transport", "global", False)]


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("construction,scale_mode,diag_correction", _DSS_ROUTES)
@pytest.mark.parametrize("kernel_mode,shift_mode,direction_mode",
                         [("asymmetric", "inverse_pair", d) for d in _DIRECTED]
                         + [("symmetric", "inverse_pair", None), ("asymmetric", "legacy", None)])
@pytest.mark.parametrize("arch", list(DSS_ARCHS))
def test_dss_patch_matches_reference(device, kernel_mode, shift_mode, direction_mode,
                                      construction, scale_mode, diag_correction, arch):
    m = _model(DSS_ARCHS, arch, dss_patch.install_cuda_shift, device,
               kernel_mode=kernel_mode, shift_mode=shift_mode, direction_mode=direction_mode,
               construction=construction, diag_correction=diag_correction,
               scale_mode=scale_mode)
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


def _dss_ops(m, x, y, z, sig):
    """(W x, W e, W^T y, cached W x, cached W^T y) on the patch path."""
    cache = m.build_weight_cache(z, sig)
    return (*dss_patch._forward_cuda(m, x, guide=z, sig=sig, return_D=True),
            dss_patch._adjoint_cuda(m, y, guide=z, sig=sig),
            dss_patch._forward_cached_cuda(m, x, cache)[0],
            dss_patch._adjoint_cached_cuda(m, y, cache))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("construction,scale_mode,diag_correction", _DSS_ROUTES)
@pytest.mark.parametrize("arch", list(DSS_ARCHS))
def test_dss_directional_matches_legacy(device, construction, scale_mode, diag_correction, arch):
    """Same weights, no new parameters: asymmetric inverse_pair + directional ==
    asymmetric legacy, on every normalisation route."""
    kw = dict(kernel_mode="asymmetric", construction=construction, scale_mode=scale_mode,
              diag_correction=diag_correction)
    leg = _model(DSS_ARCHS, arch, dss_patch.install_cuda_shift, device, shift_mode="legacy", **kw)
    dirn = _twin(DSS_ARCHS, arch, dss_patch.install_cuda_shift, device, leg,
                 direction_mode="directional", **kw)
    assert sum(p.numel() for p in dirn.parameters()) == sum(p.numel() for p in leg.parameters())
    x, z, y, sig = _inputs(device)
    for a, b in zip(_dss_ops(dirn, x, y, z, sig), _dss_ops(leg, x, y, z, sig)):
        torch.testing.assert_close(a, b, **TOL)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("construction,scale_mode,diag_correction", _DSS_ROUTES)
@pytest.mark.parametrize("direction_mode", ["gated", "gated_potential"])
@pytest.mark.parametrize("arch", list(DSS_ARCHS))
def test_dss_zero_gate_matches_symmetric(device, direction_mode, construction, scale_mode,
                                         diag_correction, arch):
    """At init (zero gate logits, g = 1/2) both directed affinities equal b, so
    the gated asymmetric model reproduces kernel_mode='symmetric'."""
    kw = dict(construction=construction, scale_mode=scale_mode, diag_correction=diag_correction)
    gated = _model(DSS_ARCHS, arch, dss_patch.install_cuda_shift, device, active_gate=False,
                   kernel_mode="asymmetric", direction_mode=direction_mode, **kw)
    sym = _twin(DSS_ARCHS, arch, dss_patch.install_cuda_shift, device, gated,
                kernel_mode="symmetric", **kw)
    x, z, y, sig = _inputs(device)
    for a, b in zip(_dss_ops(gated, x, y, z, sig), _dss_ops(sym, x, y, z, sig)):
        torch.testing.assert_close(a, b, **TOL)


@pytest.mark.parametrize("kernel_mode", ["asymmetric", "symmetric"])
def test_dss_installer_routes_cpu_to_reference(kernel_mode):
    m = _model(DSS_ARCHS, "drunet", dss_patch.install_cuda_shift, "cpu",
               kernel_mode=kernel_mode)
    x, z, y, sig = _inputs("cpu")
    assert type(m)._dss_cuda_installed
    # 'directional' is the default for the asymmetric kernel; none for the symmetric one.
    assert m.direction_mode == ("directional" if kernel_mode == "asymmetric" else None)
    # CPU tensors never take the CUDA path, whatever use_cuda_shift says.
    torch.testing.assert_close(m.forward(x, z, sig=sig)[0],
                               _ref(m, lambda: m.forward(x, z, sig=sig)[0]), rtol=0, atol=0)
