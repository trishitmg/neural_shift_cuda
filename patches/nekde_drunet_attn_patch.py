"""
Integration patch for NeKDeDRUNetAttn (NKD_drunet_attn_v2 / v3 / v4) to use
neural_shift_cuda kernels.

What this is
------------
A drop-in replacement for the `forward` method of `NeKDeDRUNetAttn`.
Same math, same gradients, but:
  * the F.pad + Python list-comprehension shift gather inside
    `_halfplane_weights` is replaced by `shift_gather` (one CUDA kernel
    per chunk, no padded_phi materialization),
  * the per-shift U/Z accumulation loop in `forward` (including the
    circular-shift inverse-symmetry branch) is replaced by a single
    `accumulate_uz` kernel call.

What this is NOT
----------------
This does not touch the AttentionWeightHead. Its `logit` / `weight`
methods take phi_c and phi_s as SEPARATE tensors (q_proj acts on phi_c,
k_proj on phi_s), so `pair_gather` (channel-concat fusion) is not
applicable here. The old `nekre_patch.py` used `pair_gather` because the
old nekre.compute_weights did `torch.cat((x, x_s), dim=1)` then ran the
result through a single proj stack -- that's the pattern pair_gather
fuses. The new heads don't have that pattern.

Cross-version compatibility
---------------------------
v2, v3, and v4 of NKD_drunet_attn share:
  * the same `_collect_shifts()` (S = 2R^2 + 2R + 1, half-plane + horizontal axis),
  * the same `_halfplane_weights` signature
    (phi_center, padded_phi, sigma, shifts) -> list of (B, n_heads, H, W),
  * the same forward accumulation structure
    (fwd weight -> comp_box -> +U,+Z; F.pad-circular -> inverse weight -> comp_box -> +U,+Z),
  * the same `_mix_heads(w, C, head_mix_mat)` -> (B, C, H, W).
This patch therefore applies to all three identically. Internal changes
to AttentionWeightHead between versions (e.g. layer_norm placement,
n_heads expansion in the control_sigmoid head, etc.) do not affect the
patch because we only call `attn_head.logit(...)` / `attn_head.weight(...)`
through their public signatures.

Usage
-----
    from NKD_drunet_attn_v4 import NeKDeDRUNetAttn       # or v2 / v3
    from nekde_drunet_attn_patch import install_cuda_shift
    install_cuda_shift(NeKDeDRUNetAttn)

After this, every NeKDeDRUNetAttn instance routes its forward through
the CUDA path. To disable per-instance:

    model.use_cuda_shift = False
"""

from __future__ import annotations

from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from neural_shift_cuda import shift_gather, accumulate_uz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_shift_tensor(self, device: torch.device) -> torch.Tensor:
    """Cache the (S, 3) int32 shift tensor on `device`."""
    if (getattr(self, "_cached_shift_tensor", None) is None
            or self._cached_shift_device != device):
        shifts = self._collect_shifts()              # [(dx, dy, has_inverse), ...]
        rows = [(int(dx), int(dy), int(bool(hi))) for (dx, dy, hi) in shifts]
        self._cached_shift_tensor = torch.tensor(
            rows, dtype=torch.int32, device=device)
        self._cached_shift_device = device
    return self._cached_shift_tensor


def _mix_heads_vec(
    w_stack: torch.Tensor,            # (S, B, n_heads, H, W)
    C: int,
    head_mix_mat: Optional[torch.Tensor],
) -> torch.Tensor:
    """Vectorised _mix_heads over the full stack. Output: (S, B, C, H, W).

    Mirrors NeKDeDRUNetAttn._mix_heads case-for-case:
      * n_heads == 1            -> broadcast to C channels
      * head_mix_mat available  -> (C, n_heads) @ stack
      * n_heads == C            -> identity
      * fallback                -> mean across heads, broadcast to C
    The result is bit-identical to looping the per-shift version, just
    fused across the S axis to save S kernel launches.
    """
    n_heads = w_stack.size(2)
    if n_heads == 1:
        return w_stack.expand(-1, -1, C, -1, -1)
    if head_mix_mat is not None:
        # (C, h) x (S, B, h, H, W) -> (S, B, C, H, W)
        return torch.einsum("ch,sbhij->sbcij", head_mix_mat, w_stack).contiguous()
    if n_heads == C:
        return w_stack
    return w_stack.mean(dim=2, keepdim=True).expand(-1, -1, C, -1, -1)


# ---------------------------------------------------------------------------
# Forward replacement
# ---------------------------------------------------------------------------

def _forward_cuda(
    self,
    x: torch.Tensor,
    guide: Optional[torch.Tensor] = None,
    sig: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CUDA-accelerated drop-in replacement for NeKDeDRUNetAttn.forward.

    Numerics: identical to the original up to floating-point reduction
    order (~1e-5 in fp32, exact in fp64).
    """
    if not self.training:
        # OOM-avoidance at test time. Mirrors the original.
        self.max_batch_shifts = 10

    B, C, H, W = x.shape
    R = self.window_rad

    # ---- Normalise sigma to (B, 1, 1, 1) (verbatim from original) ----
    if sig is not None and not torch.is_tensor(sig):
        sig = x.new_full((B, 1, 1, 1), float(sig))
    elif sig is not None:
        if sig.dim() == 0:
            sig = sig.view(1, 1, 1, 1).expand(B, 1, 1, 1)
        elif sig.shape[0] == 1 and B > 1:
            sig = sig.expand(B, -1, -1, -1)

    # ---- Guide features ----
    g_input = x if guide is None else guide
    phi = self.pre_activation(g_input, sigma=sig).contiguous()   # (B, F, H, W)

    # ---- Shift tensor ----
    shifts_t = self._get_shift_tensor(x.device)                  # (S, 3) int32
    shifts_xy = shifts_t[:, :2].contiguous()                     # (S, 2)
    S = shifts_t.shape[0]
    chunk = self.max_batch_shifts if self.max_batch_shifts is not None else S

    # ---- Head-mixing matrix ----
    head_mix_mat = (
        F.softmax(self.raw_head_mix, dim=1)
        if self.raw_head_mix is not None else None
    )

    # ------------------------------------------------------------------
    # Per-chunk weight computation.
    #
    # For each chunk:
    #   1. shift_gather(phi, shifts_chunk) -> (n*B, F, H, W) gathered phi_s,
    #      plus (n*B, 1, H, W) validity mask. This replaces the
    #      [padded_phi[:, :, R+dx:R+dx+H, R+dy:R+dy+W] for ...] +
    #      torch.cat sequence. F.pad is avoided entirely (one less
    #      copy of an (B, F, H+2R, W+2R) tensor).
    #   2. phi_c_batch = phi.repeat(n, 1, 1, 1) (unchanged).
    #   3. attn_head.logit / weight as before.
    # The mask tiles are concatenated outside the loop -- we need the
    # full (S*B, 1, H, W) mask for the accumulator.
    # ------------------------------------------------------------------
    weight_tiles: List[torch.Tensor] = []   # logits or weights depending on path
    mask_tiles: List[torch.Tensor] = []

    for start in range(0, S, chunk):
        end = min(start + chunk, S)
        n = end - start
        shifts_chunk = shifts_xy[start:end]                      # (n, 2) int32

        phi_s_batch, mask_chunk = shift_gather(phi, shifts_chunk)  # (n*B, F, H, W), (n*B, 1, H, W)
        phi_c_batch = phi.repeat(n, 1, 1, 1)
        sig_batch = sig.repeat(n, 1, 1, 1) if sig is not None else None

        if self.output_activation == "softmax":
            if self.use_grad_checkpoint:
                t = grad_checkpoint(
                    self.attn_head.logit, phi_c_batch, phi_s_batch, sig_batch,
                    use_reentrant=False,
                )
            else:
                t = self.attn_head.logit(phi_c_batch, phi_s_batch, sig_batch)
        else:  # control_sigmoid
            if self.use_grad_checkpoint:
                t = grad_checkpoint(
                    self.attn_head.weight, phi_c_batch, phi_s_batch, sig_batch,
                    use_reentrant=False,
                )
            else:
                t = self.attn_head.weight(phi_c_batch, phi_s_batch, sig_batch)

        # t: (n*B, n_heads, H, W). Split into n per-shift (B, n_heads, H, W) tiles.
        weight_tiles.extend(t.split(B, dim=0))
        mask_tiles.append(mask_chunk)        # (n*B, 1, H, W)

    # ---- Joint softmax across the half-plane shifts (softmax path only) ----
    if self.output_activation == "softmax":
        # Stack: (B, S, n_heads, H, W)
        logits = torch.stack(weight_tiles, dim=1)
        logits = logits - logits.amax(dim=1, keepdim=True)
        w_per_shift = F.softmax(logits, dim=1)                   # (B, S, n_heads, H, W)
    else:
        # Already positive; just stack.
        w_per_shift = torch.stack(weight_tiles, dim=1)           # (B, S, n_heads, H, W)

    # Reorder to shift-major to match shift_gather/accumulate_uz convention.
    # (B, S, h, H, W) -> (S, B, h, H, W)
    w_stack = w_per_shift.permute(1, 0, 2, 3, 4).contiguous()

    # Head-mix to per-channel weights: (S, B, h, H, W) -> (S, B, C, H, W)
    w_stack = _mix_heads_vec(w_stack, C, head_mix_mat)

    # Flatten to (S*B, C, H, W) and apply forward validity mask.
    w_all = w_stack.view(S * B, C, H, W)
    mask_all = torch.cat(mask_tiles, dim=0)                      # (S*B, 1, H, W)
    w_all = (w_all * mask_all).contiguous()

    # ---- Single fused U/Z accumulation (forward + inverse symmetry) ----
    U_num, Z = accumulate_uz(x.contiguous(), w_all, shifts_t)
    U = U_num / Z.clamp_min(1e-6)

    if not self.training:
        self.max_batch_shifts = None

    return U, Z


# ---------------------------------------------------------------------------
# Public installer
# ---------------------------------------------------------------------------

def install_cuda_shift(model_cls):
    """Monkey-patch a NeKDeDRUNetAttn class to use neural_shift_cuda.

    Idempotent. Works for v2, v3, and v4 (their `forward` bodies are
    structurally identical and only the AttentionWeightHead internals
    differ -- and we call the head through its stable public API).
    """
    if getattr(model_cls, "_cuda_shift_installed", False):
        return model_cls

    model_cls._get_shift_tensor = _get_shift_tensor
    original_forward = model_cls.forward

    def patched_forward(self, x, guide=None, sig=None):
        # Lazy-init cache fields so we don't need to touch __init__.
        if not hasattr(self, "_cached_shift_tensor"):
            self.use_cuda_shift = True
            self._cached_shift_tensor = None
            self._cached_shift_device = None

        if getattr(self, "use_cuda_shift", True) and x.is_cuda:
            return _forward_cuda(self, x, guide=guide, sig=sig)
        return original_forward(self, x, guide=guide, sig=sig)

    model_cls.forward = patched_forward
    model_cls._cuda_shift_installed = True
    return model_cls


# ---------------------------------------------------------------------------
# Optional: a direct-edit recipe, in case the user prefers editing the
# source file rather than monkey-patching. The diff against any of v2/v3/v4
# is identical.
# ---------------------------------------------------------------------------
#
# In NeKDeDRUNetAttn.__init__, AFTER `self.window_rad = window_rad`:
#
#     # CUDA acceleration cache
#     self.use_cuda_shift = True
#     self._cached_shift_tensor = None
#     self._cached_shift_device = None
#
# Add `_get_shift_tensor` as an instance method (body is the function
# defined at the top of this file).
#
# Replace the body of `forward` with the body of `_forward_cuda` above,
# adjusting `self` references as needed (in-class definition will use
# bare `self`, no first-argument plumbing).
# ---------------------------------------------------------------------------
