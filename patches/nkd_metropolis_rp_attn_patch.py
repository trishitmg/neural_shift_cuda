"""Integration patch for NeKDeMetropolisRPDRUNetAttn (NKD_mp_RP_drunet_attn_v2).

Why the base ``nkd_metropolis_attn_patch`` is NOT compatible
-----------------------------------------------------------
The RP model adds a random-permutation block: p permutations of the H*W lattice
plus their inverse twins. Its contribution is COUPLED to the translation block
through the shared combined degree  d = (K_trans + K_perm) e  that the Metropolis
scaling  max(d_i, d_j)  uses for every edge. Two facts follow:

  1. Permutation edges are arbitrary index gathers, not translations. None of
     the shift kernels -- ``shift_gather`` / ``accumulate_uz`` /
     ``metropolis_aggregate`` -- can express them; they all assume (dx, dy)
     circular shifts. So there is no shift kernel to fuse the perm block into,
     and none needs to be modified.

  2. ``metropolis_aggregate`` computes the raw degree d = K e INTERNALLY from the
     translation half-plane and bakes the max(d_i, d_j) scaling in. It neither
     accepts an external perm-degree contribution nor exposes the raw degree, so
     it cannot be reused as a black box even for the translation part once perms
     are present: the translation edges would be scaled by a translation-only
     degree instead of the combined one.

Consequently, installing the base metropolis patch on the RP class SILENTLY
DROPS every permutation edge (verified: the fused path returns the n_perms=0
operator, diverging by O(1) from the true RP operator once perms are on). That
is a silent correctness bug, not a crash -- hence this dedicated installer.

What this patch does (no new CUDA kernel, no kernel changes)
-----------------------------------------------------------
The permutation block already runs as native ATen CUDA ops (index_select /
maximum / elementwise), which are efficient; the only shift-fusable work is the
translation block, and it is only separable from the perms when there are none.
So the installer gates PER INSTANCE / PER CACHE at call time:

  * n_perms == 0  -> the RP operator is bit-for-bit the base Metropolis operator,
    so route ``forward`` / ``forward_cached`` through the base metropolis CUDA
    path (``metropolis_aggregate`` / ``accumulate_uz``). Full acceleration,
    numerically identical (verified to ~1e-16 in fp64).

  * n_perms  > 0  -> no correct shift fusion exists; run the model's own native
    (ATen, GPU) ``forward`` / ``forward_cached``. Correct and already on-device.

This reuses the base patch's implementation functions verbatim and adds only the
gate, so it inherits every fix to that path. Set ``model.use_cuda_shift = False``
to force native everywhere.

If a fused RP operator is ever wanted for the p>0 case, it would need a NEW
kernel that (a) gathers by an index permutation rather than a shift and (b)
accepts the externally combined degree so the two blocks share max(d_i, d_j);
the existing shift kernels cannot be adapted to it. Because the perm gathers are
memory-bound scatters that ATen ``index_select`` already handles well, the
practical speed lever for p>0 is vectorising the Python perm loop (a single
``index_select`` over the (p, N) index) or CUDA graphs -- not a bespoke kernel.

Usage
-----
    from NKD_mp_RP_drunet_attn_v2 import NeKDeMetropolisRPDRUNetAttn
    from neural_shift_cuda.integration import install_cuda_shift_metropolis_rp
    install_cuda_shift_metropolis_rp(NeKDeMetropolisRPDRUNetAttn)
"""

from __future__ import annotations

import inspect

# Reuse the base Metropolis CUDA implementations verbatim; they are correct for
# the translation-only (n_perms == 0) operator, which is exactly the RP operator
# in that case.
from .nkd_metropolis_attn_patch import (
    _forward_cuda as _mp_forward_cuda,
    _forward_cached_cuda as _mp_forward_cached_cuda,
)


def install_cuda_shift(model_cls):
    """Patch the RP model with an n_perms-gated CUDA path. Idempotent.

    n_perms == 0 routes through the base metropolis CUDA kernels; n_perms > 0
    (or a cache carrying a non-empty permutation block) falls back to the
    model's native ATen forward, which is correct and already GPU-resident.
    """
    if getattr(model_cls, "_metropolis_rp_cuda_installed", False):
        return model_cls
    if not hasattr(model_cls, "forward"):
        raise TypeError(
            "model_cls must provide forward(x, guide=None, sig=None).")

    original_forward = model_cls.forward
    original_forward_cached = getattr(model_cls, "forward_cached", None)

    _fwd_sig = inspect.signature(original_forward)
    _has_return_D = "return_D" in _fwd_sig.parameters
    _return_D_default = (
        _fwd_sig.parameters["return_D"].default if _has_return_D else False)

    def patched_forward(self, x, guide=None, sig=None,
                        return_D=_return_D_default):
        if not hasattr(self, "use_cuda_shift"):
            self.use_cuda_shift = True
        # Only the perm-free operator is expressible with the shift kernels.
        if (self.use_cuda_shift and x.is_cuda
                and int(getattr(self, "n_perms", 0)) == 0):
            return _mp_forward_cuda(self, x, guide=guide, sig=sig,
                                    return_D=return_D)
        if _has_return_D:
            return original_forward(self, x, guide=guide, sig=sig,
                                    return_D=return_D)
        return original_forward(self, x, guide=guide, sig=sig)

    model_cls.forward = patched_forward

    if original_forward_cached is not None:
        _c_sig = inspect.signature(original_forward_cached)
        _c_has_return_D = "return_D" in _c_sig.parameters
        _c_return_D_default = (
            _c_sig.parameters["return_D"].default
            if _c_has_return_D else False)

        def patched_forward_cached(self, x, cache,
                                   return_D=_c_return_D_default):
            if not hasattr(self, "use_cuda_shift"):
                self.use_cuda_shift = True
            # Gate on the CACHE's frozen perm block (authoritative for this
            # solve) rather than the live n_perms, in case they differ.
            perm_list = getattr(cache, "perm_w_hat_fwd_list", None)
            n_perm = len(perm_list) if perm_list is not None else 0
            if (getattr(self, "use_cuda_shift", True) and x.is_cuda
                    and n_perm == 0):
                return _mp_forward_cached_cuda(self, x, cache, return_D=return_D)
            if _c_has_return_D:
                return original_forward_cached(self, x, cache, return_D=return_D)
            return original_forward_cached(self, x, cache)

        model_cls.forward_cached = patched_forward_cached

    model_cls._metropolis_rp_cuda_installed = True
    return model_cls
