"""
Integration patch for GASDDRUNetAttn v2 (GASD_drunet_attn_v2) to use
neural_shift_cuda kernels.

What this is
------------
A drop-in replacement for the `forward` method of v2's `GASDDRUNetAttn`.
Same math, same gradients, but:
  * the F.pad + Python list-comprehension shift gather inside
    `_all_shift_weights` is replaced by `shift_gather` (one CUDA kernel
    per chunk, no padded_phi materialization),
  * the per-shift U/Z accumulation loop in `forward` (including the per-shift
    `comp_box` masking) is replaced by a single `accumulate_uz` kernel call.

GASD vs NeKDe (why this is a separate patch)
--------------------------------------------
GASD is the ROW-stochastic (singly-stochastic) denoiser  W = Z^{-1}U:
  * `_collect_shifts()` returns ALL (2R+1)^2 shifts as (dx, dy) pairs (no
    half-plane, no `has_inverse` flag), and the weight of every shift is
    produced independently by the network;
  * `forward` accumulates each shift exactly ONCE -- there is no circular-shift
    inverse twin, so U is non-symmetric and K = Z^{-1}U is only row-stochastic.
The NeKDe patch, by contrast, feeds a half-plane shift list with `has_inverse=1`
and lets `accumulate_uz` synthesise the symmetric twin. Here we instead build
the (S, 3) shift tensor with the inverse flag set to 0 on EVERY row, so
`accumulate_uz` performs the forward circular gather only -- the exact
full-window, non-symmetric operator GASD's reference forward computes.

comp_box toggle
---------------
The model's `comp_box` flag is honoured at runtime (this replaces the former
separate v5 patch):
  * comp_box=True  -> truncated (non-periodic) NLM. Every shift's weight is
    multiplied by `comp_box` (zeros out the boundary region that circular
    padding wrapped in); the CUDA analog is the `w_all = w_all * mask_all`
    step below, `mask_all` being shift_gather's wrap indicator. Because every
    shift's inverse flag is 0, only the forward mask is applied.
  * comp_box=False -> fully periodic (circulant) NLM; the mask step is skipped.
Instances without the attribute default to True (the historical behaviour).

What this is NOT
----------------
This does not touch the AttentionWeightHead. Its `logit` / `weight` methods
take phi_c and phi_s as SEPARATE tensors (q_proj acts on phi_c, k_proj on
phi_s), so `pair_gather` (channel-concat fusion) is not applicable here.

Head mixing
-----------
GASD keeps the `head_mix_pos_act` knob ('softmax' | 'softplus' | 'relu'), so the
patch builds `head_mix_mat` with the SAME activation the model would use,
rather than hardcoding softmax.

Usage
-----
    from GASD_drunet_attn_v2 import GASDDRUNetAttn
    from gasd_drunet_attn_patch import install_cuda_shift
    install_cuda_shift(GASDDRUNetAttn)

After this, every GASDDRUNetAttn instance routes its forward through the CUDA
path when the input is on a CUDA device. To disable per-instance:

    model.use_cuda_shift = False
"""

from __future__ import annotations

import inspect
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
    """Cache the (S, 3) int32 shift tensor on `device`.

    GASD's `_collect_shifts()` returns (dx, dy) pairs over the FULL (2R+1)^2
    window. We build the (S, 3) tensor with the inverse flag pinned to 0 on
    every row, so `accumulate_uz` does the forward circular gather only (no
    symmetric twin) -- the non-symmetric, row-stochastic operator GASD wants.
    """
    if (getattr(self, "_cached_shift_tensor", None) is None
            or self._cached_shift_device != device):
        shifts = self._collect_shifts()              # [(dx, dy), ...] full window
        rows = [(int(dx), int(dy), 0) for (dx, dy) in shifts]
        self._cached_shift_tensor = torch.tensor(
            rows, dtype=torch.int32, device=device)
        self._cached_shift_device = device
    return self._cached_shift_tensor


def _head_mix_mat(self) -> Optional[torch.Tensor]:
    """Build the (C, n_heads) head-mixing matrix using the model's own
    `head_mix_pos_act`. Matches GASDDRUNetAttn.forward exactly; defaults to
    'softmax' for instances that lack the attribute."""
    if self.raw_head_mix is None:
        return None
    act = getattr(self, "head_mix_pos_act", "softmax")
    if act == "softmax":
        return F.softmax(self.raw_head_mix, dim=1)
    if act == "softplus":
        return F.softplus(self.raw_head_mix)
    return F.relu(self.raw_head_mix)


def _mix_heads_vec(
    w_stack: torch.Tensor,            # (S, B, n_heads, H, W)
    C: int,
    head_mix_mat: Optional[torch.Tensor],
) -> torch.Tensor:
    """Vectorised _mix_heads over the full stack. Output: (S, B, C, H, W).

    Mirrors GASDDRUNetAttn._mix_heads case-for-case:
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
    return_D: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """CUDA-accelerated drop-in replacement for v2 GASDDRUNetAttn.forward.

    Numerics: identical to the v2 reference forward up to floating-point
    reduction order (~1e-5 in fp32, exact in fp64). comp_box boundary masking
    is applied via shift_gather's wrap indicator, matching v2's truncated NLM.
    """
    if not self.training:
        # OOM-avoidance at test time. Mirrors the NeKDe patches.
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

    # ---- Shift tensor (full window, inverse flag = 0 on every row) ----
    shifts_t = self._get_shift_tensor(x.device)                  # (S, 3) int32
    shifts_xy = shifts_t[:, :2].contiguous()                     # (S, 2)
    S = shifts_t.shape[0]
    chunk = self.max_batch_shifts if self.max_batch_shifts is not None else S

    # ---- Head-mixing matrix (respects head_mix_pos_act) ----
    head_mix_mat = _head_mix_mat(self)

    # ---- comp_box toggle (runtime; formerly the separate v5 patch) ----
    # comp_box=True  -> truncated (non-periodic) NLM: mask out wrap-around
    #                   neighbours via shift_gather's validity mask (old v2).
    # comp_box=False -> fully periodic (circulant) NLM: no boundary masking
    #                   (old v5). Default True for instances without the flag.
    use_box = bool(getattr(self, "comp_box", True))

    # ------------------------------------------------------------------
    # Per-chunk weight computation.
    #
    # For each chunk:
    #   1. shift_gather(phi, shifts_chunk) -> (n*B, F, H, W) gathered phi_s,
    #      plus (n*B, 1, H, W) validity mask (the comp_box wrap indicator).
    #   2. phi_c_batch = phi.repeat(n, 1, 1, 1) (unchanged).
    #   3. attn_head.logit / weight as before.
    # The mask tiles are concatenated outside the loop -- we need the
    # full (S*B, 1, H, W) mask to fold comp_box into the weights.
    # ------------------------------------------------------------------
    weight_tiles: List[torch.Tensor] = []   # logits or weights depending on path
    mask_tiles: List[torch.Tensor] = []

    for start in range(0, S, chunk):
        end = min(start + chunk, S)
        n = end - start
        shifts_chunk = shifts_xy[start:end]                      # (n, 2) int32

        phi_s_batch, mask_chunk = shift_gather(
            phi, shifts_chunk)  # (n*B, F, H, W), (n*B, 1, H, W)
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
        if use_box:
            mask_tiles.append(mask_chunk)        # (n*B, 1, H, W)

    # ---- Joint softmax across ALL (2R+1)^2 shifts (softmax path only) ----
    # NOTE: matching v2's reference forward, comp_box is applied AFTER the
    # softmax (masked weights need not re-normalize to sum 1 at the border);
    # Z below is the sum of the masked weights, so W = Z^{-1}U stays row-
    # stochastic wherever Z > 0.
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

    # Flatten to (S*B, C, H, W); apply the forward comp_box mask only when
    # comp_box is on (periodic mode leaves the wrap-around neighbours in).
    w_all = w_stack.view(S * B, C, H, W)
    if use_box:
        mask_all = torch.cat(mask_tiles, dim=0)                  # (S*B, 1, H, W)
        w_all = w_all * mask_all
    w_all = w_all.contiguous()

    # ---- Single fused U/Z accumulation (forward only; inverse flag = 0) ----
    # Every row of shifts_t has has_inverse=0, so accumulate_uz gathers x
    # circularly and places each (comp_box-masked) weight at K[i, i+d] ONLY --
    # the full-window, non-symmetric operator v2's forward computes.
    U_num, Z = accumulate_uz(x.contiguous(), w_all, shifts_t)
    U = U_num / Z.clamp_min(1e-6)

    if not self.training:
        self.max_batch_shifts = None

    if return_D:
        return U, Z
    return U, None


# ---------------------------------------------------------------------------
# Public installer
# ---------------------------------------------------------------------------

def install_cuda_shift(model_cls):
    """Monkey-patch a v2 GASDDRUNetAttn class to use neural_shift_cuda.

    Idempotent. The patch calls the AttentionWeightHead only through its stable
    public API (`logit` / `weight`), so head-internal changes do not affect it.
    """
    if getattr(model_cls, "_cuda_shift_installed", False):
        return model_cls

    model_cls._get_shift_tensor = _get_shift_tensor
    original_forward = model_cls.forward

    # Derive return_D's default from the wrapped forward so the patched
    # signature stays in lockstep with the model instead of hardcoding it, and
    # only forward return_D to the reference path when that forward accepts it.
    _fwd_sig = inspect.signature(original_forward)
    _has_return_D = "return_D" in _fwd_sig.parameters
    _return_D_default = (
        _fwd_sig.parameters["return_D"].default if _has_return_D else True)

    def patched_forward(self, x, guide=None, sig=None,
                        return_D=_return_D_default):
        # Lazy-init cache fields so we don't need to touch __init__.
        if not hasattr(self, "_cached_shift_tensor"):
            self.use_cuda_shift = True
            self._cached_shift_tensor = None
            self._cached_shift_device = None

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
