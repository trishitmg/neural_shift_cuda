"""
Integration patch for the DSG-variant NeKDeDRUNetAttn (NKD_dsg_drunet_attn_v2).

Why a separate installer
------------------------
The DSG file swaps the roles of two methods relative to v2/v3/v4:

  * ``forward``  is now the DSG-NLM DENOISER (the trained operator). It runs the
    guide network ONCE via ``build_weight_cache`` and then drives THREE
    row-stochastic NLM matvecs through ``forward_cached``. It is pure
    orchestration (sqrt, spatial amax, elementwise) around those matvecs and
    MUST stay in Python -- there is no shift/reduction work in it to fuse.

  * ``NLM``      is now the row-stochastic operator W x = D^{-1} K x -- the body
    that used to be ``forward`` in v2/v3/v4, and the thing the CUDA path in
    ``nekde_drunet_attn_patch`` actually accelerates.

The base ``install_cuda_shift`` patches ``forward``. On this class that would
overwrite the DSG denoiser with the operator CUDA path and destroy the
three-pass structure. So this installer routes:

    NLM            -> _forward_cuda          (operator; one network pass + fused reduction)
    forward_cached -> _forward_cached_cuda   (fixed-guide cached matvec, one reduction launch)
    _KT_action     -> _kt_action_cuda        (adjoint; K = K^T so no new kernel)

and leaves ``forward`` untouched. Because the DSG ``forward`` calls
``build_weight_cache`` (whose internal degree pass and the subsequent two
matvecs all go through ``forward_cached``), the denoiser's heavy work runs on
the CUDA cached path automatically, while the single guide-network evaluation
stays where it belongs (torch ops on the GPU).

No new CUDA kernels
-------------------
None are required. The DSG denoiser is fixed arithmetic wrapped around NLM
matvecs, and those matvecs are already ``normalized_accumulate_uz`` /
``accumulate_uz``. Every implementation function is imported unchanged from
``nekde_drunet_attn_patch`` -- this file only re-wires which method each
replaces.

Usage
-----
    from NKD_dsg_drunet_attn_v2 import NeKDeDRUNetAttn
    from neural_shift_cuda.integration import install_cuda_shift_dsg_attn
    install_cuda_shift_dsg_attn(NeKDeDRUNetAttn)

Disable per-instance with ``model.use_cuda_shift = False`` (honoured by the
denoiser too, since it flows through the patched ``forward_cached``).

Apply this ONLY to the DSG file. For v2/v3/v4 (operator in ``forward``) use
``install_cuda_shift_attn`` from ``nekde_drunet_attn_patch``.
"""

from __future__ import annotations

import inspect

# Reuse the base CUDA implementations verbatim: they are self-contained (they
# call the model's own _halfplane_weights / _collect_shifts / pre_activation and
# the shared module-level _head_mix_mat / _mix_heads_vec helpers), so nothing
# about them depends on WHICH method name they are installed under.
from .nekde_drunet_attn_patch import (
    _get_shift_tensor,
    _forward_cuda,
    _forward_cached_cuda,
    _kt_action_cuda,
)


def install_cuda_shift(model_cls):
    """Monkey-patch the DSG NeKDeDRUNetAttn class to use neural_shift_cuda.

    Idempotent. Patches ``NLM`` (operator), ``forward_cached`` (fixed-guide
    matvec) and ``_KT_action`` (adjoint); deliberately does NOT patch
    ``forward`` (the DSG denoiser), which orchestrates the cached matvecs.
    """
    if getattr(model_cls, "_cuda_shift_installed", False):
        return model_cls

    if getattr(model_cls, "NLM", None) is None:
        raise AttributeError(
            f"{model_cls.__name__} has no NLM method. This installer targets "
            "the DSG variant where the operator lives in NLM; for v2/v3/v4 use "
            "install_cuda_shift_attn (nekde_drunet_attn_patch) instead.")

    model_cls._get_shift_tensor = _get_shift_tensor

    # ---- NLM operator (old `forward` body) -> _forward_cuda ----
    original_nlm = model_cls.NLM
    _nlm_sig = inspect.signature(original_nlm)
    _nlm_has_return_D = "return_D" in _nlm_sig.parameters
    _nlm_return_D_default = (
        _nlm_sig.parameters["return_D"].default if _nlm_has_return_D else True)

    def patched_nlm(self, x, guide=None, sig=None, return_D=_nlm_return_D_default):
        if not hasattr(self, "_cached_shift_tensor"):
            self.use_cuda_shift = True
            self._cached_shift_tensor = None
            self._cached_shift_device = None
        if getattr(self, "use_cuda_shift", True) and x.is_cuda:
            return _forward_cuda(self, x, guide=guide, sig=sig,
                                 return_D=return_D)
        if _nlm_has_return_D:
            return original_nlm(self, x, guide=guide, sig=sig,
                                return_D=return_D)
        return original_nlm(self, x, guide=guide, sig=sig)

    model_cls.NLM = patched_nlm

    # ---- Fixed-guide cached matvec -> _forward_cached_cuda ----
    # This is what the DSG `forward` (and build_weight_cache's degree pass)
    # calls, so patching it here is what CUDA-accelerates the denoiser itself.
    original_forward_cached = getattr(model_cls, "forward_cached", None)
    if original_forward_cached is not None:
        _c_sig = inspect.signature(original_forward_cached)
        _c_has_return_D = "return_D" in _c_sig.parameters
        _c_return_D_default = (
            _c_sig.parameters["return_D"].default if _c_has_return_D else True)

        def patched_forward_cached(self, x, cache, return_D=_c_return_D_default):
            if not hasattr(self, "use_cuda_shift"):
                self.use_cuda_shift = True
            if getattr(self, "use_cuda_shift", True) and x.is_cuda:
                return _forward_cached_cuda(self, x, cache, return_D=return_D)
            if _c_has_return_D:
                return original_forward_cached(self, x, cache, return_D=return_D)
            return original_forward_cached(self, x, cache)

        model_cls.forward_cached = patched_forward_cached

    # ---- Adjoint (_KT_action / NLM_transpose route here) -> _kt_action_cuda ----
    _orig_kt = getattr(model_cls, "_KT_action", None)

    def patched_kt_action(self, y, guide=None, sig=None,
                          return_D=False, apply_Dinv=False):
        if not hasattr(self, "_cached_shift_tensor"):
            self.use_cuda_shift = True
            self._cached_shift_tensor = None
            self._cached_shift_device = None
        if getattr(self, "use_cuda_shift", True) and y.is_cuda:
            return _kt_action_cuda(self, y, guide=guide, sig=sig,
                                   return_D=return_D, apply_Dinv=apply_Dinv)
        if _orig_kt is None:
            raise AttributeError(
                f"{type(self).__name__} has no reference _KT_action; add the "
                "arch-level _KT_action/NLM_transpose before installing.")
        return _orig_kt(self, y, guide=guide, sig=sig,
                        return_D=return_D, apply_Dinv=apply_Dinv)

    model_cls._KT_action = patched_kt_action

    model_cls._cuda_shift_installed = True
    return model_cls
