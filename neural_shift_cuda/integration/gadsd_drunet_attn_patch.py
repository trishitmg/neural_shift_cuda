"""
Integration patch for GADSDDRUNetAttn (GADSD_drunet_attn) to use
neural_shift_cuda kernels.

Architecture
------------
GADSD is a fully-circular, group-averaged DOUBLY-stochastic denoiser::

    W x = sum_{g in G} alpha_g P_g x ,   alpha_g > 0, sum_g alpha_g = 1

where alpha_g is a PER-CHANNEL scalar per transform: the head emits a
per-pixel, per-head map; ``_mix_heads`` combines heads into C channels (still
per-pixel); ``_pool_spatial`` pools over (H, W) ONLY, keeping the channel axis,
so each transform g yields an (B, C) vector of weights -- C*S weights per
image. ``_finalize_weights`` normalizes over the transform axis PER CHANNEL
(sum_g alpha_g = 1 for each channel), so W is exactly stochastic and the degree
map Z is identically one (no U/Z division).

This is a drop-in replacement for ``forward``. Same math, same gradients, but:

  * the per-transform feature gather ``torch.cat([g.apply(phi) for g ...])``
    inside ``_transform_weights`` is replaced, for the TRANSLATION transforms,
    by a single ``shift_gather`` CUDA kernel per chunk (no per-shift
    ``torch.roll`` and no ``torch.cat``);
  * the per-channel weighted accumulation loop ``U += alpha_g * P_g x`` in
    ``forward`` is replaced, for the TRANSLATION transforms, by a single fused
    ``accumulate_uz_scalar`` call with (St, B, C) weights -- one scalar per
    (transform, image, channel), never materialized per-pixel. The kernel's Z
    output is discarded (weights are pre-normalized, so there is no division).

D4 transforms (``rot90``/``flip``/``transpose`` ...) are index permutations, not
translations, so they are handled with the model's own ``g.apply`` in a small
torch pass (at most 8 of them). Transform ordering is preserved so
``w[:, idx]`` still lines up with ``self.transforms[idx]``.

Model-owned pieces
------------------
Head calls (``attn_head.logit`` / ``attn_head.weight``), head mixing
(``self._mix_heads`` with ``self._head_mix_mat()``), spatial pooling
(``self._pool_spatial``), the control-sigmoid clamp and the transform-axis
normalization (``self._finalize_weights``) are all invoked on the model itself,
so any change to those stays in sync automatically. The per-chunk pipeline
mirrors ``_transform_weights`` exactly: head -> _mix_heads (per-pixel) ->
_pool_spatial; head-mix BEFORE pooling matters for n_heads > 1 and for
'sum'/'max' pooling, where the two orders do not commute.

max_batch_shifts / gradient checkpointing
-----------------------------------------
Both are honoured HERE, exactly as the reference ``_transform_weights`` does:
  * ``self.max_batch_shifts`` chunks the head evaluation (translations AND D4)
    so peak activation memory is bounded by the chunk, not by S.
  * ``self.use_grad_checkpoint`` wraps each per-chunk head call in
    ``torch.utils.checkpoint`` (``use_reentrant=False``); mixing and pooling
    stay outside the checkpoint, as in the reference.

Shift convention
----------------
``shift_gather`` / ``accumulate_uz_scalar`` use GATHER indexing
``out[h,w] = in[(h+dx)%H, (w+dy)%W]``, whereas ``ShiftTransform(dx, dy).apply``
uses ``torch.roll(t, (dx, dy))`` = ``in[(h-dx)%H, (w-dy)%W]``. To reproduce the
model's transform EXACTLY (per index, so equivalence is tight, not just
set-equal), we pass ``(-dx, -dy)`` to the kernels for a ``ShiftTransform(dx,
dy)``. The same negated shift is used for both the phi gather and the x
accumulate, so each transform index maps to the same permutation on both sides.

Usage
-----
    from GADSD_drunet_attn import GADSDDRUNetAttn
    from gadsd_drunet_attn_patch import install_cuda_shift
    install_cuda_shift(GADSDDRUNetAttn)

After this, every instance routes its forward through the CUDA path when the
input is on a CUDA device. To disable per-instance:

    model.use_cuda_shift = False
"""

from __future__ import annotations

import inspect
from typing import List, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from neural_shift_cuda import shift_gather, accumulate_uz_scalar


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_head(head_fn, phi_c, phi_s, sig, use_ckpt):
    """Call the attention head, optionally under gradient checkpointing."""
    if use_ckpt:
        return grad_checkpoint(head_fn, phi_c, phi_s, sig, use_reentrant=False)
    return head_fn(phi_c, phi_s, sig)


def _gadsd_transform_partition(self):
    """Split self.transforms into translation entries and 'other' (D4) entries,
    preserving original indices. Cached; invalidated if the list identity
    changes (it is built once in __init__)."""
    cache = getattr(self, "_gadsd_part_cache", None)
    if cache is not None and cache[0] == id(self.transforms):
        return cache[1], cache[2]
    trans_meta: List[Tuple[int, int, int]] = []   # (orig_idx, dx, dy)
    other_meta: List[Tuple[int, object]] = []      # (orig_idx, transform_obj)
    for i, g in enumerate(self.transforms):
        if hasattr(g, "dx") and hasattr(g, "dy"):
            trans_meta.append((i, int(g.dx), int(g.dy)))
        else:
            other_meta.append((i, g))
    self._gadsd_part_cache = (id(self.transforms), trans_meta, other_meta)
    return trans_meta, other_meta


def _gadsd_trans_shifts(self, trans_meta, device: torch.device) -> torch.Tensor:
    """Cache the (St, 2) int32 NEGATED shift tensor (gather convention) on
    ``device`` -- see the 'Shift convention' note in the module docstring."""
    cache = getattr(self, "_gadsd_shift_cache", None)
    if (cache is not None and cache[0] == id(self.transforms)
            and cache[1] == device):
        return cache[2]
    rows = [(-dx, -dy) for (_, dx, dy) in trans_meta]
    shifts = (torch.tensor(rows, dtype=torch.int32, device=device)
              if rows else torch.empty(0, 2, dtype=torch.int32, device=device))
    self._gadsd_shift_cache = (id(self.transforms), device, shifts)
    return shifts


# ---------------------------------------------------------------------------
# Forward replacement
# ---------------------------------------------------------------------------

def _forward_cuda(
    self,
    x: torch.Tensor,
    guide: Optional[torch.Tensor] = None,
    sig: Optional[torch.Tensor] = None,
    return_D: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """CUDA-accelerated drop-in for ``GADSDDRUNetAttn.forward``
    (per-channel, C*S-weights-per-image architecture).

    Numerics: identical to the reference forward up to floating-point reduction
    order (~1e-5 fp32, exact fp64).
    """
    B, C, H, W = x.shape

    if getattr(self, "_requires_square", False) and H != W:
        raise ValueError(
            f"transform_family={getattr(self, 'transform_family', '?')!r} uses "
            f"D4 transforms, which require a square image, got H={H}, W={W}.")

    # ---- Normalise sigma to (B, 1, 1, 1) (verbatim from the model) ----
    if sig is not None and not torch.is_tensor(sig):
        sig = x.new_full((B, 1, 1, 1), float(sig))
    elif sig is not None:
        if sig.dim() == 0:
            sig = sig.view(1, 1, 1, 1).expand(B, 1, 1, 1)
        elif sig.shape[0] == 1 and B > 1:
            sig = sig.expand(B, -1, -1, -1)

    # ---- Guide features ----
    g_input = x if guide is None else guide
    phi = self.pre_activation(g_input, sigma=sig).contiguous()      # (B, F, H, W)

    trans_meta, other_meta = _gadsd_transform_partition(self)
    S = len(self.transforms)
    chunk = self.max_batch_shifts if self.max_batch_shifts is not None else max(S, 1)

    head_fn = (self.attn_head.logit
               if self.output_activation == "softmax"
               else self.attn_head.weight)
    head_mix_mat = self._head_mix_mat()

    # Per-channel weight tile (B, C, 1, 1) per transform, placed by orig index.
    # Per-chunk pipeline mirrors _transform_weights: head -> _mix_heads
    # (per-pixel, heads -> C channels) -> _pool_spatial over (H, W).
    chan_tiles: List[Optional[torch.Tensor]] = [None] * S

    # ---- (a) translation transforms: shift_gather + chunked head ----
    trans_shifts = _gadsd_trans_shifts(self, trans_meta, x.device)   # (St, 2) int32
    St = trans_shifts.size(0)
    for start in range(0, St, chunk):
        end = min(start + chunk, St)
        n = end - start
        shifts_chunk = trans_shifts[start:end].contiguous()
        phi_s_batch, _ = shift_gather(phi, shifts_chunk)            # (n*B, F, H, W)
        phi_c_batch = phi.repeat(n, 1, 1, 1)
        sig_batch = sig.repeat(n, 1, 1, 1) if sig is not None else None

        m = _run_head(head_fn, phi_c_batch, phi_s_batch, sig_batch,
                      self.use_grad_checkpoint)                     # (n*B, h, H, W)
        m = self._mix_heads(m, C, head_mix_mat)                    # (n*B, C, H, W)
        m = self._pool_spatial(m)                                  # (n*B, C, 1, 1)
        for j, tile in enumerate(m.split(B, dim=0)):
            chan_tiles[trans_meta[start + j][0]] = tile

    # ---- (b) D4 / other transforms: torch permutation + chunked head ----
    for start in range(0, len(other_meta), chunk):
        grp = other_meta[start:start + chunk]
        n = len(grp)
        phi_s_batch = torch.cat([g.apply(phi) for (_, g) in grp], dim=0)
        phi_c_batch = phi.repeat(n, 1, 1, 1)
        sig_batch = sig.repeat(n, 1, 1, 1) if sig is not None else None

        m = _run_head(head_fn, phi_c_batch, phi_s_batch, sig_batch,
                      self.use_grad_checkpoint)
        m = self._mix_heads(m, C, head_mix_mat)
        m = self._pool_spatial(m)
        for j, tile in enumerate(m.split(B, dim=0)):
            chan_tiles[grp[j][0]] = tile

    aff = torch.stack(chan_tiles, dim=1)                           # (B, S, C, 1, 1)

    # Raw scores exactly as _transform_weights returns them: raw logits for
    # 'softmax', clamped positive affinities for 'control_sigmoid'; then the
    # model's own per-channel transform-axis normalization (sum_g w_g = 1).
    raw = aff if self.output_activation == "softmax" else aff.clamp_min(1e-8)
    w = self._finalize_weights(raw)                                # (B, S, C, 1, 1)

    # ------------------------------------------------------------------
    # Group-averaged accumulation: U = sum_g w_g P_g x, sum_g w_g = 1 (per
    # channel).
    #   translations -> fused accumulate_uz_scalar with (St, B, C) weights
    #                   (one scalar per transform per channel); the kernel's Z
    #                   output is discarded -- no division.
    #   D4 / other    -> torch per-channel weighted permutation accumulate.
    # ------------------------------------------------------------------
    U = torch.zeros_like(x)

    if St > 0:
        trans_idx = [orig_idx for (orig_idx, _dx, _dy) in trans_meta]
        # w[:, trans_idx] is (B, St, C, 1, 1) -> (St, B, C)
        w_trans = w[:, trans_idx, :, 0, 0].permute(1, 0, 2).contiguous()
        U_t, _Z_t = accumulate_uz_scalar(x.contiguous(), w_trans, trans_shifts)
        U = U + U_t

    for (orig_idx, g) in other_meta:
        U = U + w[:, orig_idx] * g.apply(x)                        # (B,C,1,1)*(B,C,H,W)

    if return_D:
        # W is exactly stochastic (sum_g w_g = 1 per channel), so Z is one.
        return U, torch.ones_like(x)
    return U, None


# ---------------------------------------------------------------------------
# Public installer
# ---------------------------------------------------------------------------

def install_cuda_shift(model_cls):
    """Monkey-patch a GADSDDRUNetAttn class to use neural_shift_cuda. Idempotent.

    The patch calls the AttentionWeightHead, head mixing, spatial pooling and
    weight finalization through the model's own stable API, so internal changes
    to those stay in sync.
    """
    if getattr(model_cls, "_cuda_shift_installed", False):
        return model_cls

    # Fail at install time, not mid-training: if the compiled binary is stale
    # (missing the scalar symbols) or failed to import, the ops layer would
    # otherwise only raise on the first CUDA forward.
    if torch.cuda.is_available():
        from neural_shift_cuda.ops import _require_cuda_ext
        _require_cuda_ext("install_cuda_shift[gadsd]", need_scalar=True)

    # This patch targets the per-channel (C*S weights per image) GADSD.
    for attr in ("_pool_spatial", "_mix_heads", "_head_mix_mat",
                 "_finalize_weights"):
        if not hasattr(model_cls, attr):
            raise AttributeError(
                f"install_cuda_shift[gadsd]: {model_cls.__name__} has no "
                f"`{attr}` -- this patch targets the per-channel GADSD "
                f"architecture (C*S weights per image, pooled over (H, W) with "
                f"`_pool_spatial`).")

    original_forward = model_cls.forward

    # Derive return_D's default from the wrapped forward so the patched
    # signature stays in lockstep with the model instead of hardcoding it, and
    # only forward return_D to the reference path when that forward accepts it.
    _fwd_sig = inspect.signature(original_forward)
    _has_return_D = "return_D" in _fwd_sig.parameters
    _return_D_default = (
        _fwd_sig.parameters["return_D"].default if _has_return_D else False)

    def patched_forward(self, x, guide=None, sig=None,
                        return_D=_return_D_default):
        if not hasattr(self, "use_cuda_shift"):
            self.use_cuda_shift = True
        if getattr(self, "use_cuda_shift", True) and x.is_cuda:
            return _forward_cuda(self, x, guide=guide, sig=sig,
                                 return_D=return_D)
        if _has_return_D:
            return original_forward(self, x, guide=guide, sig=sig,
                                    return_D=return_D)
        return original_forward(self, x, guide=guide, sig=sig)

    model_cls.forward = patched_forward
    model_cls._cuda_shift_installed = True
    return model_cls
