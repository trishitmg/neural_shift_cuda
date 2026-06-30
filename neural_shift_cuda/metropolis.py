"""
neural_shift_cuda.metropolis
----------------------------
Thin wrapper around the fused CUDA op `metropolis_aggregate` (csrc/) used by
nkd_metropolis_attn_v2 / v5. Exposes a single function

    metropolis_aggregate(w_half, img, shifts, use_box, eps) -> (Wx, d, d_hat)

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
Returns (Wx, d, d_hat) where Wx = x - d_hat (.) x + K_hat x, d = K e, d_hat = K_hat e.
"""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn.functional as F

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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
    # d      = K e        (raw degree, kept for interface compatibility)
    # d_hat  = K_hat e    (the row-sum the model's forward returns as `degree_hat`)
    return Wx, d, d_hat


def metropolis_aggregate(
    w_half: torch.Tensor,
    img: torch.Tensor,
    shifts: List[Tuple[int, int, bool]],
    use_box: bool,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dispatch to the CUDA kernel if built and inputs are on GPU, else PyTorch.

    Returns ``(Wx, d, d_hat)`` where ``Wx = (1 - d_hat) * x + K_hat x``,
    ``d = K e`` (raw degree) and ``d_hat = K_hat e`` (the value the model's
    ``forward`` returns as ``degree_hat``).
    """
    if _HAS_EXT and img.is_cuda and not torch.is_grad_enabled():
        # The fused kernel is for inference / fixed-point iteration (no autograd
        # through the kernel). Training uses the differentiable PyTorch path.
        shifts_t = _shifts_to_tensor(shifts, img.device)
        Wx, d, d_hat = _ext.metropolis_aggregate(
            w_half.contiguous(), img.contiguous(), shifts_t,
            int(bool(use_box)), float(eps))
        return Wx, d, d_hat
    return metropolis_aggregate_torch(w_half, img, shifts, use_box, eps)
