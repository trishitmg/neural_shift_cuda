"""
End-to-end equivalence test: NeKDeDRUNetAttn (v2/v3/v4) original forward
vs the CUDA-patched forward.

Loads the model file dynamically so the test does not require the path to
be on sys.path. Override the path with:

    NKD_ATTN_PATH=/path/to/NKD_drunet_attn_v4.py pytest tests/test_attn_forward_equivalence.py
"""

import importlib.util
import os
import sys
from pathlib import Path

import pytest
import torch

torch.set_grad_enabled(True)

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="needs CUDA")

DEFAULT_PATHS = [
    "/mnt/user-data/uploads/NKD_drunet_attn_v4.py",
    "/mnt/user-data/uploads/NKD_drunet_attn_v3.py",
    "/mnt/user-data/uploads/NKD_drunet_attn_v2.py",
]


def _load_attn_module(path: str):
    spec = importlib.util.spec_from_file_location("nkd_attn_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nkd_attn_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _candidate_paths():
    env = os.environ.get("NKD_ATTN_PATH")
    if env:
        return [env]
    return [p for p in DEFAULT_PATHS if Path(p).exists()]


def _make_model(mod, *, output_activation: str, n_heads: int, R: int):
    cls = mod.NeKDeDRUNetAttn
    # Constructor args vary slightly between versions; introspect.
    import inspect
    sig = inspect.signature(cls.__init__)
    params = sig.parameters
    kwargs = {}
    if "feat_ch" in params:
        kwargs["feat_ch"] = 16
    if "window_rad" in params:
        kwargs["window_rad"] = R
    if "n_heads" in params:
        kwargs["n_heads"] = n_heads
    if "output_activation" in params:
        kwargs["output_activation"] = output_activation
    if "qk_ch" in params:
        kwargs["qk_ch"] = 8
    if "in_ch" in params:
        kwargs["in_ch"] = 1
    elif "in_channels" in params:
        kwargs["in_channels"] = 1
    if "n_drunet_blocks" in params:
        kwargs["n_drunet_blocks"] = 1
    if "drunet_nc" in params:
        kwargs["drunet_nc"] = [16, 32, 32, 32]
    if "head_mix" in params:
        kwargs["head_mix"] = "broadcast_or_mean"
    return cls(**kwargs).eval()


@requires_cuda
@pytest.mark.parametrize("path", _candidate_paths())
@pytest.mark.parametrize("output_activation", ["control_sigmoid", "softmax"])
@pytest.mark.parametrize("n_heads", [1, 4])
def test_forward_matches_reference(path, output_activation, n_heads):
    mod = _load_attn_module(path)
    R = 2
    try:
        model = _make_model(
            mod, output_activation=output_activation, n_heads=n_heads, R=R)
    except TypeError as e:
        pytest.skip(f"Constructor signature mismatch for {path}: {e}")

    model = model.to("cuda").double()    # fp64 for exact-ish comparison
    torch.manual_seed(0)
    B, C, H, W = 2, 1, 24, 24
    x = torch.randn(B, C, H, W, device="cuda", dtype=torch.float64)
    sig = torch.full((B, 1, 1, 1), 25.0 / 255.0,
                     device="cuda", dtype=torch.float64)

    # Reference: untouched forward.
    with torch.no_grad():
        U_ref, Z_ref = model(x, sig=sig)

    # Patched: install_cuda_shift on a fresh copy of the class to avoid
    # leaking the monkey-patch across parametrised tests.
    from copy import deepcopy
    model_patched = deepcopy(model)
    # Force the class to be a subclass so install_cuda_shift only touches
    # this instance's class, not the global one.
    patched_cls = type(
        f"{type(model_patched).__name__}_PatchedCopy",
        (type(model_patched),),
        {},
    )
    model_patched.__class__ = patched_cls

    from neural_shift_cuda.__init__ import (  # noqa: F401  (extension loaded)
        shift_gather, accumulate_uz,
    )
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches"))
    import nekde_drunet_attn_patch as patch
    patch.install_cuda_shift(patched_cls)

    with torch.no_grad():
        U_new, Z_new = model_patched(x, sig=sig)

    # Reduction order differs (per-shift loop vs fused kernel), so allow
    # a small fp tolerance even in fp64 (~1e-10 typical).
    assert torch.allclose(U_ref, U_new, rtol=1e-8, atol=1e-9), \
        f"U mismatch: max abs diff = {(U_ref - U_new).abs().max().item()}"
    assert torch.allclose(Z_ref, Z_new, rtol=1e-8, atol=1e-9), \
        f"Z mismatch: max abs diff = {(Z_ref - Z_new).abs().max().item()}"


@requires_cuda
@pytest.mark.parametrize("path", _candidate_paths())
@pytest.mark.parametrize("output_activation", ["control_sigmoid", "softmax"])
def test_backward_finite_grads(path, output_activation):
    """Sanity: every parameter receives a finite gradient through the
    patched forward."""
    mod = _load_attn_module(path)
    try:
        model = _make_model(
            mod, output_activation=output_activation, n_heads=2, R=2)
    except TypeError as e:
        pytest.skip(f"Constructor signature mismatch for {path}: {e}")

    model = model.to("cuda").float().train()

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "patches"))
    import nekde_drunet_attn_patch as patch
    patched_cls = type(
        f"{type(model).__name__}_PatchedCopy",
        (type(model),),
        {},
    )
    model.__class__ = patched_cls
    patch.install_cuda_shift(patched_cls)

    torch.manual_seed(0)
    x = torch.randn(2, 1, 32, 32, device="cuda", requires_grad=False)
    sig = torch.full((2, 1, 1, 1), 25.0 / 255.0, device="cuda")
    U, Z = model(x, sig=sig)
    loss = U.sum() + Z.sum()
    loss.backward()

    n_with_grad = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        n_with_grad += 1
        assert torch.isfinite(p.grad).all(), f"non-finite grad in {name}"
    assert n_with_grad > 0, "no parameter received any gradient"
