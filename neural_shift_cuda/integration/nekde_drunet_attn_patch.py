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

comp_box handling
-----------------
The model's `comp_box` flag is honoured on the CUDA path (this replaces the
former separate v5 patch); both values are fused, no reference fallback:
  * comp_box=True  -> truncated NLM. Forward edge masked by shift_gather's
    validity mask; inverse edge masked by accumulate_uz's built-in has_inverse
    boundary check. shifts_t carries has_inverse=1 on the half-plane.
  * comp_box=False -> fully periodic (circulant) NLM. accumulate_uz's
    has_inverse branch always masks the inverse edge, so it cannot express the
    periodic operator directly. Instead we enumerate each half-plane shift AND
    its explicit inverse (-dx,-dy), both with has_inverse=0, feeding the kernel
    the pre-shifted inverse weight w_inv = roll(w_fwd,(dx,dy)). The kernel then
    performs forward-only circular gathers with NO masking, reproducing the
    periodic symmetric operator bit-for-bit -- reusing the already-validated
    accumulate_uz (no kernel change). Costs ~2x shift entries at comp_box=False.
Instances without the attribute default to True (the historical behaviour).

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
Two branches, selected at runtime:
  * NEW files (v2 with attn_impl/_gather_core, GQA n_kv_heads, softmax_impl):
    weight computation is DELEGATED to the model's own `_halfplane_weights`
    (which runs its projection-once gather path), so every weight-path
    feature -- gather/loop, GQA, stacked/streaming softmax -- is inherited
    automatically; only the U/Z accumulation is replaced by accumulate_uz.
  * OLDER files (v3/v4 heads without _gather_core): the original
    shift_gather + per-chunk `attn_head.logit(...)` / `attn_head.weight(...)`
    path, unchanged. These heads are called through their stable public
    signatures, so internal differences between versions do not matter.
All versions share `_collect_shifts()` (S = 2R^2 + 2R + 1), the
`_halfplane_weights` signature, the forward accumulation structure, and
`_mix_heads(w, C, head_mix_mat)`.

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
    """Cache the (S, 3) int32 shift tensor on `device`."""
    if (getattr(self, "_cached_shift_tensor", None) is None
            or self._cached_shift_device != device):
        # [(dx, dy, has_inverse), ...]
        shifts = self._collect_shifts()
        rows = [(int(dx), int(dy), int(bool(hi))) for (dx, dy, hi) in shifts]
        self._cached_shift_tensor = torch.tensor(
            rows, dtype=torch.int32, device=device)
        self._cached_shift_device = device
    return self._cached_shift_tensor


def _head_mix_mat(self) -> Optional[torch.Tensor]:
    """Build the (C, n_heads) head-mixing matrix using the model's own
    `head_mix_pos_act`. Matches NeKDeDRUNetAttn.forward exactly; defaults to
    'softmax' for instances that lack the attribute (e.g. v3/v4)."""
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
    return_D: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
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

    # ---- Head-mixing matrix (respects head_mix_pos_act; defaults to softmax) ----
    head_mix_mat = _head_mix_mat(self)

    # ---- comp_box toggle (runtime; replaces the former separate v5 patch) ----
    # comp_box=True  -> truncated NLM: forward edge masked by shift_gather's
    #                   validity mask, inverse edge masked by accumulate_uz's
    #                   built-in has_inverse boundary check.
    # comp_box=False -> fully periodic (circulant) NLM. accumulate_uz's
    #                   has_inverse branch always masks the inverse edge, so we
    #                   CANNOT use it for the periodic operator. Instead we
    #                   enumerate each half-plane shift AND its explicit inverse
    #                   (-dx,-dy), both with has_inverse=0, and hand the kernel
    #                   the pre-shifted inverse weight w_inv = roll(w_fwd,(dx,dy)).
    #                   The kernel then does pure circular forward gathers with no
    #                   masking -- exactly the periodic symmetric operator, using
    #                   the same validated accumulate_uz (no kernel change).
    use_box = bool(getattr(self, "comp_box", True))

    # ------------------------------------------------------------------
    # Fast path: NEW model API (files with attn_impl/_gather_core, i.e. the
    # projection-once "gather" weight path, GQA via n_kv_heads, and
    # softmax_impl selection). Delegate the weight computation to the
    # model's own `_halfplane_weights`. Rationale:
    #   * the model's gather path already computes q_proj/k_proj ONCE and
    #     slices VIEWS of the padded projection per shift -- there is
    #     nothing left for shift_gather to save on the weight side (a
    #     materialized gather of K would cost n_heads*qk_ch channels per
    #     shift vs feat_ch for phi, i.e. MORE memory, not less);
    #   * re-implementing the weight math here is exactly what breaks when
    #     the head changes. GQA is the concrete case: `_score` views k
    #     with n_heads while a GQA k_proj emits n_kv_heads*qk_ch channels,
    #     so the legacy branch below would crash on a GQA model.
    #     Delegation stays correct for gather/loop, GQA, stacked/streaming
    #     softmax, and any future weight-path change (the model guards its
    #     own illegal combinations).
    # The CUDA value-add of this patch -- fusing the U/Z accumulation and
    # inverse-symmetry loop into one accumulate_uz launch -- is unchanged.
    # softmax_impl='streaming' is honoured inside _halfplane_weights and is
    # numerically identical to 'stacked', so nothing downstream changes.
    # ------------------------------------------------------------------
    if hasattr(self, "_gather_core"):
        shift_list = self._collect_shifts()
        padded_phi = F.pad(phi, (R, R, R, R), mode="circular")
        weights_list = self._halfplane_weights(
            phi, padded_phi, sig, shift_list)
        # per-shift tensors are (B, n_heads, H, W) -> (S, B, h, H, W)
        w_stack = torch.stack(list(weights_list), dim=0)

        # Head-mix to per-channel weights: (S, B, h, H, W) -> (S, B, C, H, W)
        w_stack = _mix_heads_vec(w_stack, C, head_mix_mat)

        if use_box:
            # comp_box mask per shift, built exactly like the reference's
            # zero-padded `ones` box. The mask depends only on the shift, so
            # (S, 1, 1, H, W) broadcasts over B and C -- no (S*B, 1, H, W)
            # materialization needed.
            box = F.pad(
                torch.ones(1, 1, H, W, device=x.device, dtype=x.dtype),
                (R, R, R, R), mode="constant", value=0.0)
            mask_stack = torch.stack(
                [box[:, :, R + dx: R + dx + H, R + dy: R + dy + W]
                 for (dx, dy, _) in shift_list], dim=0)   # (S, 1, 1, H, W)
            w_all = (w_stack * mask_stack).reshape(
                S * B, C, H, W).contiguous()
            U_num, Z = accumulate_uz(x.contiguous(), w_all, shifts_t)
        else:
            # Fully periodic: same explicit-inverse enumeration as the
            # legacy branch (has_inverse=0 rows; kernel does forward-only
            # circular gathers with no masking). w_stack is UNMASKED here.
            entries: List[torch.Tensor] = []
            rows: List[Tuple[int, int, int]] = []
            for i, (dx, dy, has_inv) in enumerate(shift_list):
                w_fwd = w_stack[i]                           # (B, C, H, W)
                entries.append(w_fwd)
                rows.append((int(dx), int(dy), 0))
                if has_inv:
                    w_inv = torch.roll(
                        w_fwd, shifts=(int(dx), int(dy)), dims=(2, 3))
                    entries.append(w_inv)
                    rows.append((-int(dx), -int(dy), 0))
            M = len(entries)
            w_all = torch.stack(entries, dim=0).reshape(
                M * B, C, H, W).contiguous()
            shifts_periodic = torch.tensor(
                rows, dtype=torch.int32, device=x.device)
            U_num, Z = accumulate_uz(x.contiguous(), w_all, shifts_periodic)

        U = U_num / Z.clamp_min(1e-6)
        if not self.training:
            self.max_batch_shifts = None
        if return_D:
            return U, Z
        return U, None

    # ------------------------------------------------------------------
    # Legacy per-chunk weight computation (older model files without the
    # _gather_core API).
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
    # logits or weights depending on path
    weight_tiles: List[torch.Tensor] = []
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

    # ---- Joint softmax across the half-plane shifts (softmax path only) ----
    if self.output_activation == "softmax":
        # Stack: (B, S, n_heads, H, W)
        logits = torch.stack(weight_tiles, dim=1)
        logits = logits - logits.amax(dim=1, keepdim=True)
        # (B, S, n_heads, H, W)
        w_per_shift = F.softmax(logits, dim=1)
    else:
        # Already positive; just stack.
        # (B, S, n_heads, H, W)
        w_per_shift = torch.stack(weight_tiles, dim=1)

    # Reorder to shift-major to match shift_gather/accumulate_uz convention.
    # (B, S, h, H, W) -> (S, B, h, H, W)
    w_stack = w_per_shift.permute(1, 0, 2, 3, 4).contiguous()

    # Head-mix to per-channel weights: (S, B, h, H, W) -> (S, B, C, H, W)
    w_stack = _mix_heads_vec(w_stack, C, head_mix_mat)

    if use_box:
        # ---- comp_box=True: mask forward edge, let accumulate_uz mask the
        #      inverse edge via its has_inverse branch. ----
        w_all = w_stack.view(S * B, C, H, W)
        mask_all = torch.cat(mask_tiles, dim=0)                  # (S*B, 1, H, W)
        w_all = (w_all * mask_all).contiguous()
        U_num, Z = accumulate_uz(x.contiguous(), w_all, shifts_t)
    else:
        # ---- comp_box=False: fully periodic. Enumerate each shift AND its
        #      explicit inverse (-dx,-dy), both has_inverse=0, feeding the
        #      pre-shifted inverse weight. The kernel does forward-only circular
        #      gathers (no masking), reproducing the periodic symmetric operator
        #      bit-for-bit. w_stack is UNMASKED here. ----
        shift_list = self._collect_shifts()                      # [(dx,dy,has_inv)]
        entries: List[torch.Tensor] = []
        rows: List[Tuple[int, int, int]] = []
        for i, (dx, dy, has_inv) in enumerate(shift_list):
            w_fwd = w_stack[i]                                   # (B, C, H, W)
            entries.append(w_fwd)
            rows.append((int(dx), int(dy), 0))
            if has_inv:
                # w_inv[h,w] = w_fwd[(h-dx) % H, (w-dy) % W] == roll by (dx,dy).
                # Matches the reference's F.pad-circular + inverse-slice exactly.
                w_inv = torch.roll(w_fwd, shifts=(int(dx), int(dy)), dims=(2, 3))
                entries.append(w_inv)
                rows.append((-int(dx), -int(dy), 0))
        M = len(entries)
        w_all = torch.stack(entries, dim=0).reshape(M * B, C, H, W).contiguous()
        shifts_periodic = torch.tensor(
            rows, dtype=torch.int32, device=x.device)
        U_num, Z = accumulate_uz(x.contiguous(), w_all, shifts_periodic)

    U = U_num / Z.clamp_min(1e-6)

    if not self.training:
        self.max_batch_shifts = None

    if return_D:
        return U, Z
    return U, None

# ---------------------------------------------------------------------------
# K^T action (NEW). K = K^T for NeKDe (half-plane + circular-shift twin), so
# K^T y is the SAME symmetric accumulation the forward uses -- NO new kernel
# is required. One network pass; apply_Dinv costs one extra accumulate_uz
# launch (weights reused), never a second network pass.
# ---------------------------------------------------------------------------

def _kt_action_cuda(self, y, guide=None, sig=None,
                    return_D=False, apply_Dinv=False):
    """CUDA drop-in for the arch-level `_KT_action`:

        K^T y          (apply_Dinv=False)
        K^T D^{-1} y   (apply_Dinv=True), D = K e = Z from the first launch.

    comp_box=True  -> pre-masked forward weights + (S, 3) has_inverse rows
                      (kernel masks the inverse edge), exactly like forward.
    comp_box=False -> explicit-inverse M-row enumeration with has_inverse=0
                      (same periodic trick as _forward_cuda).
    """
    if not self.training:
        self.max_batch_shifts = 10

    B, C, H, W = y.shape
    R = self.window_rad

    # ---- Normalise sigma to (B, 1, 1, 1) (verbatim from original) ----
    if sig is not None and not torch.is_tensor(sig):
        sig = y.new_full((B, 1, 1, 1), float(sig))
    elif sig is not None:
        if sig.dim() == 0:
            sig = sig.view(1, 1, 1, 1).expand(B, 1, 1, 1)
        elif sig.shape[0] == 1 and B > 1:
            sig = sig.expand(B, -1, -1, -1)

    g_input = y if guide is None else guide
    phi = self.pre_activation(g_input, sigma=sig).contiguous()

    shift_list = self._collect_shifts()
    padded_phi = F.pad(phi, (R, R, R, R), mode="circular")
    weights_list = self._halfplane_weights(
        phi, padded_phi, sig, shift_list)                        # ONE network pass
    w_stack = _mix_heads_vec(
        torch.stack(list(weights_list), dim=0), C, _head_mix_mat(self))

    if bool(getattr(self, "comp_box", True)):
        box = F.pad(
            torch.ones(1, 1, H, W, device=y.device, dtype=y.dtype),
            (R, R, R, R), mode="constant", value=0.0)
        mask_stack = torch.stack(
            [box[:, :, R + dx: R + dx + H, R + dy: R + dy + W]
             for (dx, dy, _) in shift_list], dim=0)              # (S, 1, 1, H, W)
        S = len(shift_list)
        w_all = (w_stack * mask_stack).reshape(S * B, C, H, W).contiguous()
        shifts_used = _get_shift_tensor(self, y.device)          # (S, 3), real flags
    else:
        entries: List[torch.Tensor] = []
        rows: List[Tuple[int, int, int]] = []
        for i, (dx, dy, has_inv) in enumerate(shift_list):
            entries.append(w_stack[i])
            rows.append((int(dx), int(dy), 0))
            if has_inv:
                entries.append(torch.roll(
                    w_stack[i], shifts=(int(dx), int(dy)), dims=(2, 3)))
                rows.append((-int(dx), -int(dy), 0))
        M = len(entries)
        w_all = torch.stack(entries, dim=0).reshape(
            M * B, C, H, W).contiguous()
        shifts_used = torch.tensor(rows, dtype=torch.int32, device=y.device)

    # First launch: (K y, Z = D). D never needs a network pass of its own.
    U, Z = accumulate_uz(y.contiguous(), w_all, shifts_used)
    if apply_Dinv:
        U, _ = accumulate_uz(
            (y / Z.clamp_min(1e-6)).contiguous(), w_all, shifts_used)

    if not self.training:
        self.max_batch_shifts = None
    return (U, Z) if return_D else (U, None)


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

    # Derive return_D's default from the wrapped forward so the patched
    # signature stays in lockstep with the model (v2/v3/v4) instead of
    # hardcoding it, and only forward return_D to the reference path when that
    # forward actually accepts it.
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

    # ---- K^T wiring (arch-level _KT_action / NLM_transpose route here) ----
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
