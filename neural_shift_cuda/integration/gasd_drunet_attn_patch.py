"""
Integration patch for GASDDRUNetAttn (GASD_drunet_attn_v2) to use
neural_shift_cuda kernels.

What this is
------------
A drop-in replacement for the `forward` method of `GASDDRUNetAttn` (the class
formerly named `NeKDeDRUNetAttn`; the alias still resolves, so this patch
installs on either name). Same math, same gradients, but:

  * the per-transform feature gather ``torch.cat([g.apply(phi) for g ...])``
    inside ``_transform_weights`` is replaced, for the TRANSLATION transforms,
    by a single ``shift_gather`` CUDA kernel per chunk (no per-shift
    ``torch.roll`` and no ``torch.cat``);
  * the scalar-weighted accumulation loop ``U += w_g * P_g x`` / ``Z += w_g``
    in ``forward`` is replaced, for the TRANSLATION transforms, by a single
    fused ``accumulate_uz_scalar`` call (weights stay (S, B, C) scalars -- they
    are never broadcast to an (S*B, C, H, W) tensor).

D4 transforms (``rot90``/``flip``/``transpose`` ...) are index permutations, not
translations, so they are handled with the model's own ``g.apply`` in a small
torch pass (at most 8 of them). The two partial (U, Z) are summed. Transform
ordering is preserved so ``alpha[:, idx]`` still lines up with
``self.transforms[idx]``.

max_batch_shifts / gradient checkpointing
-----------------------------------------
Both are honoured HERE, exactly as the reference ``_transform_weights`` does:
  * ``self.max_batch_shifts`` chunks the head evaluation (translations AND D4)
    so peak activation memory is bounded by the chunk, not by S. Because GASD
    pools each affinity map to a SCALAR before stacking, the cross-transform
    stack is only (B, S, n_heads, 1, 1) -- there is no (B, S, n_heads, H, W)
    materialization, so chunking actually bounds memory here (unlike the old
    per-pixel softmax path).
  * ``self.use_grad_checkpoint`` wraps each per-chunk head call in
    ``torch.utils.checkpoint`` (``use_reentrant=False``), so head activations
    are recomputed in backward instead of being stored.

What this is NOT
----------------
This does not touch ``AttentionWeightHead``. Its ``logit`` / ``weight`` methods
take ``phi_c`` and ``phi_s`` as SEPARATE tensors, so ``pair_gather`` (channel
concat fusion) does not apply. Pooling (``self._spatial_pool``) and head mixing
(``self._mix_heads``) are called on the model itself, so any change to those
stays in sync automatically.

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
    from GASD_drunet_attn_v2 import GASDDRUNetAttn      # or NeKDeDRUNetAttn alias
    from gasd_drunet_attn_patch import install_cuda_shift
    install_cuda_shift(GASDDRUNetAttn)

After this, every instance routes its forward through the CUDA path when the
input is on a CUDA device. To disable per-instance:

    model.use_cuda_shift = False
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
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


def _head_mix_mat(self) -> Optional[torch.Tensor]:
    """(C, n_heads) head-mixing matrix, using the model's own
    ``head_mix_pos_act`` (defaults to 'softmax')."""
    if self.raw_head_mix is None:
        return None
    act = getattr(self, "head_mix_pos_act", "softmax")
    if act == "softmax":
        return F.softmax(self.raw_head_mix, dim=1)
    if act == "softplus":
        return F.softplus(self.raw_head_mix)
    return F.relu(self.raw_head_mix)


def _normalize_transform_weights(self, aff: torch.Tensor) -> torch.Tensor:
    """Normalize stacked scalars (B, S, n_heads, 1, 1) over the transform axis.
    Mirrors ``GASDDRUNetAttn._transform_weights`` tail exactly."""
    if not self.normalize_transform_weights:
        if self.output_activation != "softmax":
            return aff.clamp_min(1e-8)
    if self.output_activation == "softmax":
        aff = aff - aff.amax(dim=1, keepdim=True)
        return torch.softmax(aff, dim=1)
    eps = 1e-8
    aff = aff.clamp_min(eps)
    return aff / aff.sum(dim=1, keepdim=True).clamp_min(eps)


def _gasd_transform_partition(self):
    """Split self.transforms into translation entries and 'other' (D4) entries,
    preserving original indices. Cached; invalidated if the list identity
    changes (it is built once in __init__)."""
    cache = getattr(self, "_gasd_part_cache", None)
    if cache is not None and cache[0] == id(self.transforms):
        return cache[1], cache[2]
    trans_meta: List[Tuple[int, int, int]] = []   # (orig_idx, dx, dy)
    other_meta: List[Tuple[int, object]] = []      # (orig_idx, transform_obj)
    for i, g in enumerate(self.transforms):
        if hasattr(g, "dx") and hasattr(g, "dy"):
            trans_meta.append((i, int(g.dx), int(g.dy)))
        else:
            other_meta.append((i, g))
    self._gasd_part_cache = (id(self.transforms), trans_meta, other_meta)
    return trans_meta, other_meta


def _gasd_trans_shifts(self, trans_meta, device: torch.device) -> torch.Tensor:
    """Cache the (St, 2) int32 NEGATED shift tensor (gather convention) on
    ``device`` -- see the 'Shift convention' note in the module docstring."""
    cache = getattr(self, "_gasd_shift_cache", None)
    if (cache is not None and cache[0] == id(self.transforms)
            and cache[1] == device):
        return cache[2]
    rows = [(-dx, -dy) for (_, dx, dy) in trans_meta]
    shifts = (torch.tensor(rows, dtype=torch.int32, device=device)
              if rows else torch.empty(0, 2, dtype=torch.int32, device=device))
    self._gasd_shift_cache = (id(self.transforms), device, shifts)
    return shifts


# ---------------------------------------------------------------------------
# Forward replacement
# ---------------------------------------------------------------------------

def _forward_cuda(
    self,
    x: torch.Tensor,
    guide: Optional[torch.Tensor] = None,
    sig: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CUDA-accelerated drop-in for ``GASDDRUNetAttn.forward``.

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

    trans_meta, other_meta = _gasd_transform_partition(self)
    S = len(self.transforms)
    chunk = self.max_batch_shifts if self.max_batch_shifts is not None else max(S, 1)

    head_fn = (self.attn_head.logit
               if self.output_activation == "softmax"
               else self.attn_head.weight)

    # Scalar weight tile (B, n_heads, 1, 1) per transform, placed by orig index.
    alpha_tiles: List[Optional[torch.Tensor]] = [None] * S

    # ---- (a) translation transforms: shift_gather + chunked head ----
    trans_shifts = _gasd_trans_shifts(self, trans_meta, x.device)   # (St, 2) int32
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
        m = self._spatial_pool(m)                                  # (n*B, h, 1, 1)
        for j, tile in enumerate(m.split(B, dim=0)):
            alpha_tiles[trans_meta[start + j][0]] = tile

    # ---- (b) D4 / other transforms: torch permutation + chunked head ----
    for start in range(0, len(other_meta), chunk):
        grp = other_meta[start:start + chunk]
        n = len(grp)
        phi_s_batch = torch.cat([g.apply(phi) for (_, g) in grp], dim=0)
        phi_c_batch = phi.repeat(n, 1, 1, 1)
        sig_batch = sig.repeat(n, 1, 1, 1) if sig is not None else None

        m = _run_head(head_fn, phi_c_batch, phi_s_batch, sig_batch,
                      self.use_grad_checkpoint)
        m = self._spatial_pool(m)
        for j, tile in enumerate(m.split(B, dim=0)):
            alpha_tiles[grp[j][0]] = tile

    aff = torch.stack(alpha_tiles, dim=1)                          # (B, S, h, 1, 1)
    alpha = _normalize_transform_weights(self, aff)                # (B, S, h, 1, 1)

    head_mix_mat = _head_mix_mat(self)

    # ------------------------------------------------------------------
    # Accumulation: U = sum_g w_g P_g x, Z = sum_g w_g (w_g scalar over space).
    #   translations -> fused accumulate_uz_scalar (weights stay (St, B, C))
    #   D4 / other    -> torch scalar-weighted permutation accumulate
    # ------------------------------------------------------------------
    U = torch.zeros_like(x)
    Z = torch.zeros_like(x)

    if St > 0:
        w_list = []
        for (orig_idx, _dx, _dy) in trans_meta:
            w_g = self._mix_heads(alpha[:, orig_idx], C, head_mix_mat)  # (B, C, 1, 1)
            w_list.append(w_g.reshape(B, C))
        w_trans = torch.stack(w_list, dim=0).contiguous()             # (St, B, C)
        U_t, Z_t = accumulate_uz_scalar(x.contiguous(), w_trans, trans_shifts)
        U = U + U_t
        Z = Z + Z_t

    for (orig_idx, g) in other_meta:
        w_g = self._mix_heads(alpha[:, orig_idx], C, head_mix_mat)     # (B, C, 1, 1)
        U = U + w_g * g.apply(x)
        Z = Z + w_g

    U = U / Z.clamp_min(1e-6)
    Z = Z.expand(B, C, H, W)
    return U, Z


# ---------------------------------------------------------------------------
# Public installer
# ---------------------------------------------------------------------------

def install_cuda_shift(model_cls):
    """Monkey-patch a GASDDRUNetAttn class to use neural_shift_cuda. Idempotent.

    The patch calls the AttentionWeightHead, spatial pooling and head mixing
    through the model's own stable API, so internal changes to those stay in
    sync.
    """
    if getattr(model_cls, "_cuda_shift_installed", False):
        return model_cls

    # Fail at install time, not mid-training: if the compiled binary is stale
    # (missing the 0.4.x scalar symbols) or failed to import, the ops layer
    # would otherwise only raise on the first CUDA forward.
    if torch.cuda.is_available():
        from neural_shift_cuda.ops import _require_cuda_ext
        _require_cuda_ext("install_cuda_shift[gasd]", need_scalar=True)

    original_forward = model_cls.forward

    def patched_forward(self, x, guide=None, sig=None):
        if not hasattr(self, "use_cuda_shift"):
            self.use_cuda_shift = True
        if getattr(self, "use_cuda_shift", True) and x.is_cuda:
            return _forward_cuda(self, x, guide=guide, sig=sig)
        return original_forward(self, x, guide=guide, sig=sig)

    model_cls.forward = patched_forward
    model_cls._cuda_shift_installed = True
    return model_cls
