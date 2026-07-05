"""
neural_shift_cuda.metropolis
----------------------------
Thin wrapper around the fused CUDA op `metropolis_aggregate` (csrc/) used by
nkd_metropolis_attn_v2 / v5. Exposes a single function

    metropolis_aggregate(w_half, img, shifts, use_box, eps) -> (Wx, d_hat)

that calls the compiled CUDA kernel when available and otherwise falls back to a
pure-PyTorch implementation with identical semantics (so the NKD modules run
unchanged on CPU / when the extension is not built).

Conventions (must match the kernel and the NKD reference forward):
  * w_half : (B, S, C, H, W) -- per-HALF-PLANE forward weights, ALREADY produced
             by the network and head-mixed to C channels. shifts[s] = (dx,dy,inv)
             is the canonical half-plane shift list from `_collect_shifts()`.
  * img    : (B, C, H, W) -- the image x the operator W acts on (unpadded; the
             op uses circular addressing internally).
  * shifts : (S, 3) int64 = (dx, dy, has_inverse).
  * use_box: 1 -> v2 comp_box (wrap-around neighbours masked out);
             0 -> v5 pure circular.
Returns (Wx, d_hat) where Wx = x - d_hat (.) x + K_hat x and d_hat = K_hat e.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .ops import accumulate_uz, shift_gather

try:  # compiled extension (built from csrc/metropolis_cuda.cu + metropolis.cpp)
    from . import _C as _ext  # type: ignore
    _HAS_EXT = hasattr(_ext, "metropolis_aggregate")
except Exception:  # pragma: no cover - extension optional
    _ext = None
    _HAS_EXT = False


def _shifts_to_tensor(shifts: List[Tuple[int, int, bool]],
                      device: torch.device) -> torch.Tensor:
    return torch.tensor(
        [[int(dx), int(dy), int(bool(inv))] for (dx, dy, inv) in shifts],
        dtype=torch.long, device=device)


def metropolis_aggregate_torch(
    w_half: torch.Tensor,                 # (B, S, C, H, W)
    img: torch.Tensor,                    # (B, C, H, W)
    shifts: List[Tuple[int, int, bool]],
    use_box: bool,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reference PyTorch path; identical math to the CUDA kernel."""
    B, S, C, H, W = w_half.shape
    R = max(max(abs(dx), abs(dy)) for (dx, dy, _) in shifts)

    padded_img = F.pad(img, (R, R, R, R), mode="circular")
    if use_box:
        box = F.pad(torch.ones(B, C, H, W, device=img.device, dtype=img.dtype),
                    (R, R, R, R), mode="constant", value=0.0)

    def fwd_mask(dx, dy):
        return box[:, :, R + dx:R + dx + H, R + dy:R + dy + W] if use_box else 1.0

    # ---- pass 1: degree d = K e ----
    d = torch.zeros_like(img)
    w_fwd_list: List[torch.Tensor] = []
    for s, (dx, dy, has_inv) in enumerate(shifts):
        w_fwd = w_half[:, s]                                   # (B, C, H, W)
        if use_box:
            w_fwd = w_fwd * fwd_mask(dx, dy)
        w_fwd_list.append(w_fwd)
        d = d + w_fwd
        if has_inv:
            wi = F.pad(w_fwd, (R, R, R, R), mode="circular")[
                :, :, R - dx:R - dx + H, R - dy:R - dy + W]
            if use_box:
                wi = wi * box[:, :, R - dx:R - dx + H, R - dy:R - dy + W]
            d = d + wi

    padded_d = F.pad(d, (R, R, R, R), mode="circular")

    # ---- pass 2: K_hat x and d_hat ----
    khat_x = torch.zeros_like(img)
    d_hat = torch.zeros_like(img)
    for s, (dx, dy, has_inv) in enumerate(shifts):
        w_fwd = w_fwd_list[s]
        d_nb = padded_d[:, :, R + dx:R + dx + H, R + dy:R + dy + W]
        wk = w_fwd / torch.maximum(d, d_nb).clamp_min(eps)
        v = padded_img[:, :, R + dx:R + dx + H, R + dy:R + dy + W]
        khat_x = khat_x + wk * v
        d_hat = d_hat + wk
        if has_inv:
            wi = F.pad(w_fwd, (R, R, R, R), mode="circular")[
                :, :, R - dx:R - dx + H, R - dy:R - dy + W]
            if use_box:
                wi = wi * box[:, :, R - dx:R - dx + H, R - dy:R - dy + W]
            d_nb_i = padded_d[:, :, R - dx:R - dx + H, R - dy:R - dy + W]
            wk_i = wi / torch.maximum(d, d_nb_i).clamp_min(eps)
            v_i = padded_img[:, :, R - dx:R - dx + H, R - dy:R - dy + W]
            khat_x = khat_x + wk_i * v_i
            d_hat = d_hat + wk_i

    Wx = img - d_hat * img + khat_x
    # d_hat  = K_hat e    (the row-sum the model's forward returns as `degree_hat`)
    return Wx, d_hat


def metropolis_aggregate_fused(
    w_half: torch.Tensor,                 # (B, S, C, H, W)
    img: torch.Tensor,                    # (B, C, H, W)
    shifts: List[Tuple[int, int, bool]],
    use_box: bool,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Differentiable Metropolis aggregation built on the fused parallel ops.

    Same math as ``metropolis_aggregate_torch`` (bit-exact for both ``use_box``
    modes), but the two symmetric accumulations run through the ``accumulate_uz``
    autograd op (one fused CUDA kernel + fused backward each) and the Metropolis
    denominator is gathered with a single ``shift_gather``, so training on CUDA
    uses the same kernels as the NeKDe / GASD paths instead of a per-shift
    Python loop. On CPU (or without the compiled extension) the ops fall back to
    their references, so this is also the CPU reference -- and it stays
    differentiable throughout.

    We expand the canonical half-plane into the FULL symmetric edge set and pass
    it to ``accumulate_uz`` with ``has_inverse=0`` on every edge. This is
    deliberate: ``accumulate_uz`` forms its own inverse edge by a *box-masked*
    circular shift (correct for finite-window v2, wrong for the purely circular
    v5), so instead we build each mirror edge's weight ourselves -- a circular
    shift of the forward weight, masked by the inverse box only when
    ``use_box`` -- exactly matching the reference. K stays symmetric by
    construction because every mirror weight is a shift of its forward partner.
    """
    B, S, C, H, W = w_half.shape
    device, dtype = img.device, img.dtype
    R = max((max(abs(dx), abs(dy)) for (dx, dy, _) in shifts), default=0)

    if use_box:
        box = F.pad(torch.ones(1, 1, H, W, device=device, dtype=dtype),
                    (R, R, R, R), mode="constant", value=0.0)

        def _box_slice(dx, dy):
            return box[:, :, R + dx:R + dx + H, R + dy:R + dy + W]  # (1,1,H,W)

    w_half_s = w_half.permute(1, 0, 2, 3, 4)                    # (S, B, C, H, W)

    edge_w: List[torch.Tensor] = []                            # each (B,C,H,W)
    edge_shifts: List[Tuple[int, int, int]] = []               # (dx,dy,0)
    for s, (dx, dy, has_inv) in enumerate(shifts):
        w_fwd = w_half_s[s]                                     # (B, C, H, W)
        if use_box:
            w_fwd = w_fwd * _box_slice(dx, dy)
        edge_w.append(w_fwd)
        edge_shifts.append((dx, dy, 0))
        if has_inv:
            # Mirror weight b(i) = w_fwd(i - s): a circular roll by +s. Masked by
            # the inverse box (R-dx, R-dy) when finite-window, as in the ref.
            b = torch.roll(w_fwd, shifts=(dx, dy), dims=(-2, -1))
            if use_box:
                b = b * _box_slice(-dx, -dy)
            edge_w.append(b)
            edge_shifts.append((-dx, -dy, 0))

    E = len(edge_shifts)
    w_full = torch.stack(edge_w, dim=0).reshape(E * B, C, H, W).contiguous()
    shifts_full = torch.tensor(edge_shifts, dtype=torch.long, device=device)
    shifts_full_xy = shifts_full[:, :2].contiguous()

    # ---- pass 1: raw degree d = K e (Z output; forward-only, no box inverse) ----
    _, d = accumulate_uz(img, w_full, shifts_full)             # d: (B, C, H, W)

    # ---- Metropolis per-edge normalization: wk = w / max(d, gather(d, edge)) --
    d_nb, _ = shift_gather(d, shifts_full_xy)                  # (E*B, C, H, W)
    d_rep = d.repeat(E, 1, 1, 1)                               # (E*B, C, H, W)
    denom = torch.maximum(d_rep, d_nb).clamp_min(eps)
    wk_full = w_full / denom               # forward/inverse masks inherited

    # ---- pass 2: K_hat x (U) and d_hat (Z) ----
    khat_x, d_hat = accumulate_uz(img, wk_full, shifts_full)

    Wx = img - d_hat * img + khat_x
    return Wx, d_hat


def metropolis_aggregate(
    w_half: torch.Tensor,
    img: torch.Tensor,
    shifts: List[Tuple[int, int, bool]],
    use_box: bool,
    eps: float = 1e-6,
    checkpoint_train: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to the CUDA kernel if built and inputs are on GPU, else PyTorch.

    Returns ``(Wx, d_hat)`` where ``Wx = (1 - d_hat) * x + K_hat x`` and
    ``d_hat = K_hat e`` (the value the model's ``forward`` returns as
    ``degree_hat``). The raw degree ``d = K e`` is used internally to form the
    Metropolis denominators but is not returned.

    The dedicated fused CUDA kernel has no backward, so it is used only for
    inference (grad disabled). Training now runs the fused, differentiable
    ``metropolis_aggregate_fused`` path, which expresses both symmetric
    accumulations through the ``accumulate_uz`` autograd op (one fused kernel +
    fused backward each) and gathers the Metropolis denominator with a single
    ``shift_gather`` -- the same parallel primitives the NeKDe / GASD training
    paths use, replacing the old per-shift Python loop. ``checkpoint_train``
    wraps that path in gradient checkpointing (default) so the ``(S*B,C,H,W)``
    weight tensors are recomputed in backward rather than retained.
    """
    if _HAS_EXT and img.is_cuda and not torch.is_grad_enabled():
        # The fused kernel is for inference / fixed-point iteration (no autograd
        # through the kernel). Training uses the differentiable fused path below.
        shifts_t = _shifts_to_tensor(shifts, img.device)
        Wx, d_hat = _ext.metropolis_aggregate(
            w_half.contiguous(), img.contiguous(), shifts_t,
            int(bool(use_box)), float(eps))
        return Wx, d_hat

    if (checkpoint_train and torch.is_grad_enabled()
            and (w_half.requires_grad or img.requires_grad)):
        # Recompute the fused accumulation in backward instead of retaining the
        # per-shift weight tensors. shifts/use_box/eps are captured by closure so
        # only the tensors (w_half, img) are checkpoint inputs.
        def _run(w_half_, img_):
            return metropolis_aggregate_fused(w_half_, img_, shifts, use_box, eps)
        return checkpoint(_run, w_half, img, use_reentrant=False)

    return metropolis_aggregate_fused(w_half, img, shifts, use_box, eps)
