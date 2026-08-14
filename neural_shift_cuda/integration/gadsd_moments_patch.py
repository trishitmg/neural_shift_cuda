"""
Integration patch for GADSDMoments (GADSD_moments) to use
neural_shift_cuda kernels.

Architecture
------------
GADSDMoments (formerly GADSDLightweight in GADSD_lightweight.py; the model
file is now GADSD_moments.py) is the lightweight group-averaged DOUBLY-stochastic
denoiser::

    W x = sum_{g in G} alpha_g P_g x ,   alpha_g > 0, sum_g alpha_g = 1

where alpha_g is a PER-CHANNEL scalar per transform. Unlike the DRUNet-attn
GADSD, the scorer (``TinyPairwiseMomentHead``) consumes the WHOLE transformed
feature stack (B, S, F, H, W) at once -- there is no per-shift ``logit`` /
``weight`` call and no per-pixel weight map at any point: the head pools
pairwise moments over (H, W) internally and emits per-transform scalars
directly. The old ``gadsd_drunet_attn_patch`` therefore does not apply to this
class (no ``_pool_spatial``, no ``attn_head.logit``); this patch targets the
stack-consuming head instead.

Two methods are patched, matching the model's own stage separation:

``_transform_weights``  (scores; also serves ``get_transform_weights``)
    The per-chunk ``torch.stack([g.apply(phi) for g in chunk])`` -- S
    ``torch.roll`` launches plus a cat -- is replaced, for TRANSLATION
    transforms, by ONE ``shift_gather`` kernel per chunk. The gathered
    (n*B, F, H, W) batch is viewed as (n, B, F, H, W) and transposed to the
    (B, n, F, H, W) layout the head expects; the head's (H, W) reductions are
    per-slice contiguous either way, so the pooled moments are bit-identical
    to the stacked reference. Head invocation (with descriptors and
    ``shift_weight_pool``), the grad-checkpoint condition, ``_mix_heads``,
    the control-sigmoid clamp, and chunk sizing (``_choose_chunk_size``) are
    all the model's own -- the loop mirrors the reference chunk-for-chunk, so
    raw scores line up index-for-index.

``forward``  (accumulation)
    The per-transform ``output += weights[:, g] * g.apply(x)`` loop is
    replaced, for TRANSLATION transforms, by a single fused
    ``accumulate_uz_scalar`` call with (St, B, C) weights -- one scalar per
    (transform, image, channel), never materialized per-pixel. The kernel's Z
    output is discarded: ``_finalize_weights`` normalizes over the transform
    axis per channel, so Z is identically one and there is no division.
    ``return_D=True`` returns all-ones, exactly as the reference does.

D4 transforms (``rot90``/``flip``/``transpose`` ...) are index permutations,
not translations, so both stages handle them with the model's own ``g.apply``
(at most 8 of them). Original transform ordering is preserved throughout so
scores, descriptors, and weights stay aligned with ``self.transforms``.

Shift convention
----------------
``shift_gather`` / ``accumulate_uz_scalar`` use GATHER indexing
``out[h,w] = in[(h+dx)%H, (w+dy)%W]``, whereas ``ShiftTransform(dx, dy).apply``
uses ``torch.roll(t, (dx, dy))`` = ``in[(h-dx)%H, (w-dy)%W]``. To reproduce the
model's transform EXACTLY (per index, not just set-equal), we pass
``(-dx, -dy)`` to the kernels. The same negated tensor is used for both the
phi gather and the x accumulation, so each transform index maps to the same
permutation on both sides.

Usage
-----
    from GADSD_moments import GADSDMoments
    from neural_shift_cuda.integration import install_cuda_shift_gadsd_moments
    install_cuda_shift_gadsd_moments(GADSDMoments)

After this, every instance routes through the CUDA path when the input is on
a CUDA device. To disable per-instance:

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

def _normalise_sigma_light(sig, reference: torch.Tensor):
    """Return sigma as (B, 1, 1, 1) on reference device/dtype (mirrors the
    model's `_normalise_sigma` for the shapes its forward accepts)."""
    if sig is None:
        return None
    B = reference.shape[0]
    if not torch.is_tensor(sig):
        return reference.new_full((B, 1, 1, 1), float(sig))
    sig = sig.to(device=reference.device, dtype=reference.dtype)
    if sig.ndim == 0:
        sig = sig.reshape(1, 1, 1, 1)
    elif sig.ndim in (1, 2):
        sig = sig.reshape(-1, 1, 1, 1)
    if sig.shape[0] == 1 and B > 1:
        sig = sig.expand(B, -1, -1, -1)
    return sig


def _moments_partition(self):
    """Split self.transforms into translation entries and 'other' (D4)
    entries, preserving original indices. Cached; keyed on the identity of the
    transforms list (built once in __init__)."""
    cache = getattr(self, "_gadsdm_part_cache", None)
    if cache is not None and cache[0] == id(self.transforms):
        return cache[1], cache[2]
    trans_meta: List[Tuple[int, int, int]] = []   # (orig_idx, dx, dy)
    other_meta: List[Tuple[int, object]] = []      # (orig_idx, transform_obj)
    for i, g in enumerate(self.transforms):
        if hasattr(g, "dx") and hasattr(g, "dy"):
            trans_meta.append((i, int(g.dx), int(g.dy)))
        else:
            other_meta.append((i, g))
    self._gadsdm_part_cache = (id(self.transforms), trans_meta, other_meta)
    return trans_meta, other_meta


def _moments_trans_shifts(self, trans_meta, device: torch.device) -> torch.Tensor:
    """Cache the (St, 2) int64 NEGATED shift tensor (gather convention) on
    ``device`` -- see the 'Shift convention' note in the module docstring."""
    cache = getattr(self, "_gadsdm_shift_cache", None)
    if (cache is not None and cache[0] == id(self.transforms)
            and cache[1] == device):
        return cache[2]
    rows = [(-dx, -dy) for (_, dx, dy) in trans_meta]
    shifts = (torch.tensor(rows, dtype=torch.int64, device=device)
              if rows else torch.empty(0, 2, dtype=torch.int64, device=device))
    self._gadsdm_shift_cache = (id(self.transforms), device, shifts)
    return shifts


def _gather_chunk_stack(self, phi: torch.Tensor, start: int, end: int) -> torch.Tensor:
    """Build the (B, n, F, H, W) transformed-feature stack for transforms
    [start, end), replacing per-shift torch.roll with shift_gather.

    Fast path: an all-translation chunk (the whole run for the pure
    'translations' family) is ONE shift_gather kernel and a stride-permuted
    view -- no stack copy at all. Mixed chunks (translations+d4 boundary)
    gather the translation members in one kernel, apply the <=8 D4 members
    with the model's own transforms, and reassemble in original order.
    """
    members = self.transforms[start:end]
    n = end - start
    is_shift = [hasattr(g, "dx") and hasattr(g, "dy") for g in members]

    if all(is_shift):
        rows = [(-int(g.dx), -int(g.dy)) for g in members]
        shifts = torch.tensor(rows, dtype=torch.int64, device=phi.device)
        gathered, _ = shift_gather(phi, shifts)          # (n*B, F, H, W)
        B = phi.shape[0]
        return gathered.view(n, B, *phi.shape[1:]).transpose(0, 1)

    if not any(is_shift):
        return torch.stack([g.apply(phi) for g in members], dim=1)

    # Mixed chunk: one gather for the shift members, apply() for the rest,
    # reassembled in original order (positions must line up with descriptors).
    B = phi.shape[0]
    rows = [(-int(g.dx), -int(g.dy)) for g, s in zip(members, is_shift) if s]
    shifts = torch.tensor(rows, dtype=torch.int64, device=phi.device)
    gathered, _ = shift_gather(phi, shifts)
    gathered = gathered.view(len(rows), B, *phi.shape[1:])
    tiles: List[torch.Tensor] = []
    k = 0
    for g, s in zip(members, is_shift):
        if s:
            tiles.append(gathered[k])
            k += 1
        else:
            tiles.append(g.apply(phi))
    return torch.stack(tiles, dim=1)


# ---------------------------------------------------------------------------
# _transform_weights replacement
# ---------------------------------------------------------------------------

def _transform_weights_cuda(
    self,
    phi_center: torch.Tensor,
    sigma,
    C: int,
) -> torch.Tensor:
    """CUDA-accelerated drop-in for GADSDMoments._transform_weights.

    Mirrors the reference chunk-for-chunk (same _choose_chunk_size, same
    descriptor slices, same grad-checkpoint condition, same head call, same
    _mix_heads, same control-sigmoid clamp); only the construction of the
    transformed feature stack changes. Numerics: bit-identical in fp64.
    """
    B = phi_center.shape[0]
    sigma_n = _normalise_sigma_light(sigma, phi_center)
    chunk_size = self._choose_chunk_size(phi_center)
    head_mix_mat = self._head_mix_mat()
    chunks: List[torch.Tensor] = []

    for start in range(0, len(self.transforms), chunk_size):
        end = min(start + chunk_size, len(self.transforms))
        transformed = _gather_chunk_stack(self, phi_center, start, end)  # (B,n,F,H,W)
        descriptors = self.transform_descriptors[start:end]

        if self.use_grad_checkpoint and torch.is_grad_enabled() and phi_center.requires_grad:
            sigma_tensor = (
                sigma_n
                if sigma_n is not None
                else phi_center.new_zeros((B, 1, 1, 1))
            )

            def score_fn(center, shifted, sig, desc):
                return self.attn_head(
                    center,
                    shifted,
                    sig,
                    desc,
                    pool=self.shift_weight_pool,
                )

            scores = grad_checkpoint(
                score_fn,
                phi_center,
                transformed,
                sigma_tensor,
                descriptors,
                use_reentrant=False,
            )
        else:
            scores = self.attn_head(
                phi_center,
                transformed,
                sigma_n,
                descriptors,
                pool=self.shift_weight_pool,
            )

        n = end - start
        mixed = self._mix_heads(
            scores.reshape(B * n, self.n_heads, 1, 1),
            C,
            head_mix_mat,
        ).reshape(B, n, C, 1, 1)
        chunks.append(mixed)

    raw = torch.cat(chunks, dim=1)
    if self.output_activation == "control_sigmoid":
        raw = raw.clamp_min(1e-8)
    return raw


# ---------------------------------------------------------------------------
# forward replacement
# ---------------------------------------------------------------------------

def _forward_cuda(
    self,
    x: torch.Tensor,
    guide: Optional[torch.Tensor] = None,
    sig=None,
    return_D: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """CUDA-accelerated drop-in for GADSDMoments.forward.

    Scores come from the (patched) self._transform_weights; normalization is
    the model's own _finalize_weights; only the group-averaged accumulation
    changes: translations go through ONE fused accumulate_uz_scalar call, D4
    members through the model's own g.apply. Numerics: bit-identical in fp64
    up to floating-point reduction order (exact in the pure-translation case,
    where the kernel accumulates in the same transform order).
    """
    B, C, H, W = x.shape
    if C != self.in_channels:
        raise ValueError(f"Model expects {self.in_channels} channels, got {C}.")
    if self._requires_square and H != W:
        raise ValueError(
            f"transform_family={self.transform_family!r} requires square images; "
            f"got H={H}, W={W}."
        )

    guide_input = x if guide is None else guide
    if (
        guide_input.shape[0] != B
        or guide_input.shape[1] != self.in_channels
        or guide_input.shape[-2:] != (H, W)
    ):
        raise ValueError(
            "guide must have the same batch size, channel count, and spatial size as x; "
            f"got x={tuple(x.shape)}, guide={tuple(guide_input.shape)}."
        )

    sigma_n = _normalise_sigma_light(sig, guide_input)
    phi = self.pre_activation(guide_input, sigma=sigma_n)
    raw = self._transform_weights(phi, sigma_n, C)     # patched: shift_gather path
    weights = self._finalize_weights(raw)              # (B, S, C, 1, 1)

    trans_meta, other_meta = _moments_partition(self)
    trans_shifts = _moments_trans_shifts(self, trans_meta, x.device)  # (St, 2)
    St = trans_shifts.size(0)

    output = torch.zeros_like(x)

    if St > 0:
        trans_idx = [orig_idx for (orig_idx, _dx, _dy) in trans_meta]
        # weights[:, trans_idx] is (B, St, C, 1, 1) -> (St, B, C)
        w_trans = weights[:, trans_idx, :, 0, 0].permute(1, 0, 2).contiguous()
        U_t, _Z = accumulate_uz_scalar(x.contiguous(), w_trans, trans_shifts)
        output = output + U_t                          # Z == sum_g w_g == 1; discarded

    for (orig_idx, g) in other_meta:
        output = output + weights[:, orig_idx] * g.apply(x)   # (B,C,1,1)*(B,C,H,W)

    if return_D:
        return output, torch.ones_like(x)
    return output, None


# ---------------------------------------------------------------------------
# Public installer
# ---------------------------------------------------------------------------

def install_cuda_shift(model_cls):
    """Monkey-patch a GADSDMoments class to use neural_shift_cuda.
    Idempotent.

    Patches BOTH `_transform_weights` (so `get_transform_weights` and any
    external scorer callers are accelerated too) and `forward` (fused scalar
    accumulation). All head calls, head mixing, chunk sizing, and weight
    finalization go through the model's own stable API.
    """
    if getattr(model_cls, "_cuda_shift_installed", False):
        return model_cls

    # Fail at install time, not mid-training, on a stale/absent binary.
    if torch.cuda.is_available():
        from neural_shift_cuda.ops import _require_cuda_ext
        _require_cuda_ext("install_cuda_shift[gadsd_moments]",
                          need_scalar=True)

    # This patch targets the lightweight stack-consuming-head GADSD.
    for attr in ("_choose_chunk_size", "_mix_heads", "_head_mix_mat",
                 "_finalize_weights", "pre_activation"):
        if not hasattr(model_cls, attr):
            raise AttributeError(
                f"install_cuda_shift[gadsd_moments]: {model_cls.__name__} "
                f"has no `{attr}` -- this patch targets GADSDMoments "
                f"(stack-consuming TinyPairwiseMomentHead, per-channel scalar "
                f"weights). For the DRUNet-attn GADSD use "
                f"install_cuda_shift_gadsd instead.")
    if hasattr(model_cls, "_pool_spatial"):
        raise AttributeError(
            "install_cuda_shift[gadsd_moments]: this class has "
            "`_pool_spatial`, which marks the per-pixel-head DRUNet-attn "
            "GADSD -- use install_cuda_shift_gadsd for that architecture.")

    original_forward = model_cls.forward
    original_transform_weights = model_cls._transform_weights

    _fwd_sig = inspect.signature(original_forward)
    _has_return_D = "return_D" in _fwd_sig.parameters
    _return_D_default = (
        _fwd_sig.parameters["return_D"].default if _has_return_D else False)

    def patched_transform_weights(self, phi_center, sigma, C):
        if getattr(self, "use_cuda_shift", True) and phi_center.is_cuda:
            return _transform_weights_cuda(self, phi_center, sigma, C)
        return original_transform_weights(self, phi_center, sigma, C)

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

    model_cls._transform_weights = patched_transform_weights
    model_cls.forward = patched_forward
    model_cls._cuda_shift_installed = True
    return model_cls
