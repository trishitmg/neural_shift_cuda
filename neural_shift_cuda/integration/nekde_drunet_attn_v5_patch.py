"""
Integration shim for NKD_drunet_attn_v5 (NeKDeDRUNetAttn, post-mix-softmax).

WHY v5 NEEDS ITS OWN SHIM (and why install_cuda_shift_attn does NOT work)
-------------------------------------------------------------------------
The v2/v3/v4 shim (`install_cuda_shift_attn`) is built around `accumulate_uz`,
whose entire value is the half-plane + circular-shift symmetry: it reconstructs
each inverse-direction weight as a circular shift of the corresponding forward
weight, so it never materialises the full window. That reconstruction is valid
only when

    weight_inv(i, s) == weight_fwd(i - s, s).

In v2/v3/v4 the per-shift weights are computed INDEPENDENTLY (softmax over the
shift axis happened per-head BEFORE head mixing, or control_sigmoid acted per
shift), so that identity holds.

v5 changed the aggregation: there is now a SINGLE softmax over the shift axis,
taken per pixel and per channel AFTER head mixing. The softmax denominator

    Zsm(i) = sum_t exp( s(i, t) )

is pixel-dependent, and Zsm(i) != Zsm(i - s) in general. Therefore

    weight_inv(i, s) = exp( s_fwd(i - s, s) ) / Zsm(i)
                    != exp( s_fwd(i - s, s) ) / Zsm(i - s) = weight_fwd(i - s, s).

The circular-shift symmetry is broken, so `accumulate_uz` would compute the
WRONG U for v5. It must not be used here. (v5 is also row-stochastic by
construction, so D = ones and there is no degree term for accumulate_uz to
produce anyway.)

There are also two API-level mismatches: v5's attention head exposes `.score(...)`
(not `.logit` / `.weight`), and v5 has no `output_activation` attribute; the head
mix matrix `raw_head_mix` is used UNCONSTRAINED (no softmax over the head axis).

WHAT THIS SHIM DOES
-------------------
The only fusable piece left is the neighbour gather. This shim replaces the
`F.pad(phi, circular)` + Python list-comprehension that builds the shifted guide
features with a single `shift_gather` call, then performs the score computation,
the inverse-score circular shift, the wrap-around masking, the shift-axis
softmax, and the weighted sum EXACTLY as v5 does (in PyTorch). Numerics are
identical to v5 up to floating-point reduction order.

Expectations: v5's cost is dominated by the DRUNet feature extractor + attention
score network and by the shift-axis softmax over a (B, C, S, H, W) tensor.
Neither is fusable here, so the speedup from this shim is modest -- it mainly
removes the padded-feature allocation. If you are choosing models for speed,
v5 does not benefit from this extension the way v2/v3/v4 do.

Usage
-----
    from NKD_drunet_attn_v5 import NeKDeDRUNetAttn
    from neural_shift_cuda.integration import install_cuda_shift_attn_v5
    install_cuda_shift_attn_v5(NeKDeDRUNetAttn)

Disable per-instance: model.use_cuda_shift = False
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from neural_shift_cuda import shift_gather


def _get_shift_tensor(self, device: torch.device) -> torch.Tensor:
    """Cache the (S, 3) int32 half-plane shift tensor on `device`."""
    if (getattr(self, "_cached_shift_tensor", None) is None
            or self._cached_shift_device != device):
        shifts = self._collect_shifts()
        rows = [(int(dx), int(dy), int(bool(hi))) for (dx, dy, hi) in shifts]
        self._cached_shift_tensor = torch.tensor(
            rows, dtype=torch.int32, device=device)
        self._cached_shift_device = device
    return self._cached_shift_tensor


def _forward_cuda_v5(
    self,
    x: torch.Tensor,
    guide: Optional[torch.Tensor] = None,
    sig: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Drop-in replacement for NeKDeDRUNetAttn.forward (v5).

    Identical math to v5; only the shifted-feature gather is replaced by
    `shift_gather`. Returns (U, D=ones), matching v5.
    """
    if not self.training:
        self.max_batch_shifts = 10  # OOM-avoidance at test time, as in v5

    B, C, H, W = x.shape
    R = self.window_rad

    # --- sigma normalisation (verbatim from v5) ---
    if sig is not None and not torch.is_tensor(sig):
        sig = x.new_full((B, 1, 1, 1), float(sig))
    elif sig is not None:
        if sig.dim() == 0:
            sig = sig.view(1, 1, 1, 1).expand(B, 1, 1, 1)
        elif sig.shape[0] == 1 and B > 1:
            sig = sig.expand(B, -1, -1, -1)

    g_input = x if guide is None else guide
    phi = self.pre_activation(g_input, sigma=sig).contiguous()

    shifts = self._collect_shifts()                 # half-plane list
    shifts_t = self._get_shift_tensor(x.device)      # (S_half, 3) int32
    shifts_xy = shifts_t[:, :2].contiguous()
    S_half = shifts_t.shape[0]

    # --- gather shifted guide features (replaces F.pad(phi) + list-comp) ---
    # phi_s_all: (S_half * B, feat_ch, H, W) in shift-major order.
    phi_s_all, _ = shift_gather(phi, shifts_xy)

    chunk = self.max_batch_shifts if self.max_batch_shifts is not None else S_half
    head_mix_mat = self.raw_head_mix  # UNCONSTRAINED (no softmax) -- v5 semantics

    # --- half-plane raw scores, mixed to C channels (mirrors _halfplane_scores) ---
    half_scores: List[torch.Tensor] = []
    for start in range(0, S_half, chunk):
        end = min(start + chunk, S_half)
        n = end - start
        phi_s_batch = phi_s_all[start * B:end * B]           # (n*B, feat_ch, H, W)
        phi_c_batch = phi.repeat(n, 1, 1, 1)
        sig_batch = sig.repeat(n, 1, 1, 1) if sig is not None else None

        if self.use_grad_checkpoint:
            s = grad_checkpoint(
                self.attn_head.score, phi_c_batch, phi_s_batch, sig_batch,
                use_reentrant=False,
            )
        else:
            s = self.attn_head.score(phi_c_batch, phi_s_batch, sig_batch)
        for s_one in s.split(B, dim=0):
            half_scores.append(self._mix_heads(s_one, C, head_mix_mat))

    # --- assemble the full window (forward + inverse via circular shift) ---
    # The image-value gather still uses padded_img; the inverse SCORE is a
    # circular shift of the forward score (NOT a fresh gather), exactly as v5,
    # to keep the numerator symmetric. shift_gather cannot do this step (each
    # score is shifted by its own single offset, not by all offsets), so it
    # stays in PyTorch.
    padded_img = F.pad(x, (R, R, R, R), mode="circular")
    box = F.pad(
        torch.ones(1, 1, H, W, device=x.device, dtype=x.dtype),
        (R, R, R, R), mode="constant", value=0.0,
    )

    score_list: List[torch.Tensor] = []
    valid_list: List[torch.Tensor] = []
    v_list: List[torch.Tensor] = []

    for i, (dx, dy, has_inverse) in enumerate(shifts):
        s_fwd = half_scores[i]
        score_list.append(s_fwd)
        valid_list.append(box[:, :, R + dx:R + dx + H, R + dy:R + dy + W] > 0.5)
        v_list.append(padded_img[:, :, R + dx:R + dx + H, R + dy:R + dy + W])

        if not has_inverse:
            continue

        dx_inv, dy_inv = -dx, -dy
        s_fwd_padded = F.pad(s_fwd, (R, R, R, R), mode="circular")
        s_inv = s_fwd_padded[:, :, R + dx_inv:R + dx_inv + H,
                             R + dy_inv:R + dy_inv + W]
        score_list.append(s_inv)
        valid_list.append(
            box[:, :, R + dx_inv:R + dx_inv + H, R + dy_inv:R + dy_inv + W] > 0.5)
        v_list.append(
            padded_img[:, :, R + dx_inv:R + dx_inv + H, R + dy_inv:R + dy_inv + W])

    scores = torch.stack(score_list, dim=2)          # (B, C, S_full, H, W)
    valid = torch.stack(valid_list, dim=2)            # (1, 1, S_full, H, W) bool

    # --- single softmax over the shift axis (verbatim from v5) ---
    neg_inf = torch.finfo(scores.dtype).min
    masked = torch.where(valid, scores, scores.new_full((), neg_inf))
    masked = masked - masked.amax(dim=2, keepdim=True)
    weights = F.softmax(masked, dim=2)               # (B, C, S_full, H, W)

    v_full = torch.stack(v_list, dim=2)
    U = (weights * v_full).sum(dim=2)
    D = torch.ones_like(x)

    if not self.training:
        self.max_batch_shifts = None
    return U, D


def install_cuda_shift_attn_v5(model_cls):
    """Monkey-patch NeKDeDRUNetAttn (v5) to use shift_gather in its forward.

    Idempotent. Only the gather is accelerated; the post-mix softmax
    aggregation stays in PyTorch (it cannot use accumulate_uz -- see module
    docstring). Falls back to the original forward on CPU or when disabled.
    """
    if getattr(model_cls, "_cuda_shift_v5_installed", False):
        return model_cls

    model_cls._get_shift_tensor = _get_shift_tensor
    original_forward = model_cls.forward

    def patched_forward(self, x, guide=None, sig=None):
        if not hasattr(self, "_cached_shift_tensor"):
            self.use_cuda_shift = True
            self._cached_shift_tensor = None
            self._cached_shift_device = None

        if getattr(self, "use_cuda_shift", True) and x.is_cuda:
            return _forward_cuda_v5(self, x, guide=guide, sig=sig)
        return original_forward(self, x, guide=guide, sig=sig)

    model_cls.forward = patched_forward
    model_cls._cuda_shift_v5_installed = True
    return model_cls
