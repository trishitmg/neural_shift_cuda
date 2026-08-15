"""
Integration patch for GASDMoments (GASD_moments) to use neural_shift_cuda
kernels.

What this is
------------
A drop-in replacement for the `forward` method of `GASDMoments` -- the GASD
port whose weight producer is the GADSD moments pair
(TinyMomentFeatureExtractor / FixedMomentFeatureExtractor +
per-pixel PixelwiseMomentHead) instead of the DRUNet + attention head.
Same math, same gradients, but:
  * the F.pad + Python list-comprehension shift gather inside
    `_all_shift_weights` is replaced by `shift_gather` (one CUDA kernel
    per chunk, no padded_phi materialization),
  * the per-shift U/Z accumulation loop in `forward` (including the per-shift
    `comp_box` masking) is replaced by a single `accumulate_uz` kernel call.

The kernel assembly of GASD_moments is a line-for-line port of
GASD_drunet_attn_v2 (full (2R+1)^2 window as (dx, dy) pairs, no half-plane,
no inverse twin -- K = Z^{-1}U is only ROW-stochastic), so this patch
mirrors `gasd_drunet_attn_patch` structurally: the (S, 3) shift tensor is
built with the inverse flag pinned to 0 on EVERY row, so `accumulate_uz`
performs the forward circular gather only. The ONLY functional difference
from the attn patch is the weight-head API: `PixelwiseMomentHead.logit /
weight` take a FOURTH argument -- the per-shift analytic descriptor -- so
each chunk additionally slices `self.shift_descriptors` and
repeat-interleaves it across the batch axis
(`descriptors[start:end].repeat_interleave(B, dim=0)`), which matches the
shift-major ordering of `shift_gather`'s (n*B, F, H, W) output exactly.

comp_box toggle
---------------
The model's `comp_box` flag is honoured at runtime:
  * comp_box=True  -> truncated (non-periodic) NLM. Every shift's weight is
    multiplied by the comp_box wrap indicator (shift_gather's validity mask).
    Because every shift's inverse flag is 0, only the forward mask applies.
  * comp_box=False -> fully periodic (circulant) NLM; the mask step is skipped.
Instances without the attribute default to True (the model's default).

Usage
-----
    from GASD_moments import GASDMoments
    from neural_shift_cuda.integration import install_cuda_shift_gasd_moments
    install_cuda_shift_gasd_moments(GASDMoments)

After this, every GASDMoments instance routes its forward through the CUDA
path when the input is on a CUDA device. To disable per-instance:

    model.use_cuda_shift = False
"""

from __future__ import annotations

import inspect
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from neural_shift_cuda import (
    shift_gather, accumulate_uz, normalized_accumulate_uz,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_sigma_patch(sig, reference: torch.Tensor):
    """Return sigma as (B, 1, 1, 1) on reference device/dtype (mirrors the
    model's module-level `_normalise_sigma` for the shapes forward accepts)."""
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


def _get_shift_tensor(self, device: torch.device) -> torch.Tensor:
    """Cache the (S, 3) int64 shift tensor on `device`.

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
            rows, dtype=torch.int64, device=device)
        self._cached_shift_device = device
    return self._cached_shift_tensor


def _head_mix_mat(self) -> Optional[torch.Tensor]:
    """Build the (C, n_heads) head-mixing matrix using the model's own
    `head_mix_pos_act`. Matches GASDMoments.forward exactly."""
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

    Mirrors GASDMoments._mix_heads case-for-case:
      * n_heads == 1            -> broadcast to C channels
      * head_mix_mat available  -> (C, n_heads) @ stack
      * n_heads == C            -> identity
      * fallback                -> mean across heads, broadcast to C
    """
    n_heads = w_stack.size(2)
    if n_heads == 1:
        return w_stack.expand(-1, -1, C, -1, -1)
    if head_mix_mat is not None:
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
    return_D: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """CUDA-accelerated drop-in replacement for GASDMoments.forward.

    Numerics: identical to the reference forward up to floating-point
    reduction order (~1e-5 in fp32, exact in fp64).
    """
    if not self.training:
        # OOM-avoidance at test time. Mirrors the attn patches.
        self.max_batch_shifts = 10

    B, C, H, W = x.shape
    R = self.window_rad

    # ---- Normalise sigma to (B, 1, 1, 1) ----
    sig = _normalise_sigma_patch(sig, x)

    # ---- Guide features ----
    g_input = x if guide is None else guide
    phi = self.pre_activation(g_input, sigma=sig).contiguous()   # (B, F, H, W)

    # ---- Shift tensor (full window, inverse flag = 0) + descriptors ----
    # Module-level call (not self._get_shift_tensor) so _forward_cuda also
    # works when invoked directly, before install_cuda_shift attached it.
    shifts_t = _get_shift_tensor(self, x.device)                 # (S, 3) int64
    shifts_xy = shifts_t[:, :2].contiguous()                     # (S, 2)
    S = shifts_t.shape[0]
    chunk = self.max_batch_shifts if self.max_batch_shifts is not None else S
    descriptors = self.shift_descriptors.to(
        device=phi.device, dtype=phi.dtype)                      # (S, D)

    # ---- Head-mixing matrix (respects head_mix_pos_act) ----
    head_mix_mat = _head_mix_mat(self)

    # ---- comp_box toggle (runtime) ----
    use_box = bool(getattr(self, "comp_box", True))

    # ------------------------------------------------------------------
    # Per-chunk weight computation (mirrors _all_shift_weights, with
    # shift_gather instead of the padded slices, plus the descriptor batch
    # in shift-major order to match shift_gather's output ordering).
    # ------------------------------------------------------------------
    weight_tiles: List[torch.Tensor] = []   # logits or weights depending on path
    mask_tiles: List[torch.Tensor] = []

    for start in range(0, S, chunk):
        end = min(start + chunk, S)
        n = end - start
        shifts_chunk = shifts_xy[start:end]                      # (n, 2) int64

        phi_s_batch, mask_chunk = shift_gather(
            phi, shifts_chunk)  # (n*B, F, H, W), (n*B, 1, H, W)
        phi_c_batch = phi.repeat(n, 1, 1, 1)
        sig_batch = sig.repeat(n, 1, 1, 1) if sig is not None else None
        desc_batch = descriptors[start:end].repeat_interleave(B, dim=0)

        if self.output_activation == "softmax":
            if self.use_grad_checkpoint:
                t = grad_checkpoint(
                    self.moment_head.logit,
                    phi_c_batch, phi_s_batch, sig_batch, desc_batch,
                    use_reentrant=False,
                )
            else:
                t = self.moment_head.logit(
                    phi_c_batch, phi_s_batch, sig_batch, desc_batch)
        else:  # control_sigmoid
            if self.use_grad_checkpoint:
                t = grad_checkpoint(
                    self.moment_head.weight,
                    phi_c_batch, phi_s_batch, sig_batch, desc_batch,
                    use_reentrant=False,
                )
            else:
                t = self.moment_head.weight(
                    phi_c_batch, phi_s_batch, sig_batch, desc_batch)

        # t: (n*B, n_heads, H, W). Split into n per-shift (B, n_heads, H, W) tiles.
        weight_tiles.extend(t.split(B, dim=0))
        if use_box:
            mask_tiles.append(mask_chunk)        # (n*B, 1, H, W)

    # ---- Joint softmax across ALL (2R+1)^2 shifts (softmax path only) ----
    # NOTE: matching the reference forward, comp_box is applied AFTER the
    # softmax; Z below is the sum of the masked weights, so W = Z^{-1}U stays
    # row-stochastic wherever Z > 0.
    if self.output_activation == "softmax":
        logits = torch.stack(weight_tiles, dim=1)                # (B, S, h, H, W)
        logits = logits - logits.amax(dim=1, keepdim=True)
        w_per_shift = F.softmax(logits, dim=1)
    else:
        w_per_shift = torch.stack(weight_tiles, dim=1)           # (B, S, h, H, W)

    # Reorder to shift-major to match shift_gather/accumulate_uz convention.
    w_stack = w_per_shift.permute(1, 0, 2, 3, 4).contiguous()    # (S, B, h, H, W)

    # Head-mix to per-channel weights: (S, B, h, H, W) -> (S, B, C, H, W)
    w_stack = _mix_heads_vec(w_stack, C, head_mix_mat)

    # Flatten to (S*B, C, H, W); apply the forward comp_box mask only when
    # comp_box is on (periodic mode leaves the wrap-around neighbours in).
    w_all = w_stack.view(S * B, C, H, W)
    if use_box:
        mask_all = torch.cat(mask_tiles, dim=0)                  # (S*B, 1, H, W)
        w_all = w_all * mask_all
    w_all = w_all.contiguous()

    # ---- Stable normalized tree reduction (forward only; inverse flag = 0) ----
    U, log_Z = normalized_accumulate_uz(
        x.contiguous(), w_all, shifts_t,
        return_log_degree=True, validate=False)
    Z = log_Z.exp() if return_D else None

    if not self.training:
        self.max_batch_shifts = None

    if return_D:
        return U, Z
    return U, None



# ---------------------------------------------------------------------------
# K^T action (NEW). GASD's K is only row-stochastic (K != K^T), so a genuine
# transpose is required (laplacian_grw's reverse factor). The ICC identity
#     pi^{-1}(w (.) y) = pi^{-1}(w) (.) pi^{-1}(y)
# turns K^T y = sum_pi pi^{-1}(w_pi (.) y) into a FORWARD-style accumulation
# with NEGATED shifts and inverse-ROLLED weights -- so the same validated
# accumulate_uz kernel does the work (has_inverse=0 rows; no new CUDA code).
# ---------------------------------------------------------------------------

def _get_kt_shift_tensor(self, device: torch.device) -> torch.Tensor:
    """(S, 3) int64 with every (dx, dy) NEGATED and the inverse flag 0."""
    if (getattr(self, "_cached_kt_shift_tensor", None) is None
            or getattr(self, "_cached_kt_shift_device", None) != device):
        rows = [(-int(dx), -int(dy), 0) for (dx, dy) in self._collect_shifts()]
        self._cached_kt_shift_tensor = torch.tensor(
            rows, dtype=torch.int64, device=device)
        self._cached_kt_shift_device = device
    return self._cached_kt_shift_tensor


def _get_roll_index(self, S: int, H: int, W: int,
                    device: torch.device) -> torch.Tensor:
    """Cached flat gather index for the per-shift inverse roll
    out[s, ..., h, w] = w[s, ..., (h - dx_s) % H, (w - dy_s) % W]."""
    key = (S, H, W, str(device))
    if getattr(self, "_cached_kt_roll_key", None) != key:
        d = torch.tensor(self._collect_shifts(), device=device)   # (S, 2)
        hh = torch.arange(H, device=device).view(1, H, 1)
        ww = torch.arange(W, device=device).view(1, 1, W)
        src_h = (hh - d[:, 0].view(S, 1, 1)) % H
        src_w = (ww - d[:, 1].view(S, 1, 1)) % W
        self._cached_kt_roll_index = (
            src_h * W + src_w).reshape(S, 1, 1, H * W)
        self._cached_kt_roll_key = key
    return self._cached_kt_roll_index


def _roll_stack(self, w_stack: torch.Tensor) -> torch.Tensor:
    """pi_s^{-1} applied to slice s of (S, B, C, H, W) in a single gather
    (the expanded index is stride-0: no (S, B, C, H*W) materialization)."""
    S, B, C, H, W = w_stack.shape
    idx = _get_roll_index(self, S, H, W, w_stack.device)
    return w_stack.reshape(S, B, C, H * W).gather(
        3, idx.expand(S, B, C, H * W)).view_as(w_stack)


def _kt_weight_stack(self, ref: torch.Tensor, guide, sig) -> torch.Tensor:
    """ONE network pass -> comp_box-masked per-shift weights (S, B, C, H, W).
    Same construction as the forward (delegates to the model's own
    _all_shift_weights), so K and K^T are built from identical entries."""
    B, C, H, W = ref.shape
    R = self.window_rad
    g_input = ref if guide is None else guide
    phi = self.pre_activation(g_input, sigma=sig).contiguous()
    padded_phi = F.pad(phi, (R, R, R, R), mode="circular")
    shift_list = self._collect_shifts()
    weights_list = self._all_shift_weights(phi, padded_phi, sig, shift_list)
    w_stack = _mix_heads_vec(
        torch.stack(list(weights_list), dim=0), C, _head_mix_mat(self))
    if bool(getattr(self, "comp_box", True)):
        box = F.pad(
            torch.ones(1, 1, H, W, device=ref.device, dtype=ref.dtype),
            (R, R, R, R), mode="constant", value=0.0)
        mask_stack = torch.stack(
            [box[:, :, R + dx: R + dx + H, R + dy: R + dy + W]
             for (dx, dy) in shift_list], dim=0)                 # (S, 1, 1, H, W)
        w_stack = w_stack * mask_stack
    return w_stack


def _kt_action_cuda(self, y, guide=None, sig=None):
    """CUDA drop-in for the arch-level `_KT_action`: K^T y."""
    if not self.training:
        self.max_batch_shifts = 10
    B, C, H, W = y.shape
    sig = _normalise_sigma_patch(sig, y)

    w_stack = _kt_weight_stack(self, y, guide, sig)
    S = w_stack.shape[0]
    w_t = _roll_stack(self, w_stack).reshape(S * B, C, H, W).contiguous()
    KT_y, _ = accumulate_uz(
        y.contiguous(), w_t, _get_kt_shift_tensor(self, y.device))
    if not self.training:
        self.max_batch_shifts = None
    return KT_y


def _nlm_transpose_cuda(self, x, guide=None, sig=None, return_D=False):
    """W^T x = K^T D^{-1} x in ONE network pass (the arch reference spends
    two: forward for D, then _KT_action). D = row degree = masked-stack
    sum over the shift axis, identical to forward's Z."""
    if not self.training:
        self.max_batch_shifts = 10
    B, C, H, W = x.shape
    sig = _normalise_sigma_patch(sig, x)

    g = x if guide is None else guide
    w_stack = _kt_weight_stack(self, x, g, sig)
    S = w_stack.shape[0]
    D = w_stack.sum(dim=0)                                       # = forward Z
    w_t = _roll_stack(self, w_stack).reshape(S * B, C, H, W).contiguous()
    WT_x, _ = accumulate_uz(
        (x / D.clamp_min(1e-6)).contiguous(), w_t,
        _get_kt_shift_tensor(self, x.device))
    if not self.training:
        self.max_batch_shifts = None
    return (WT_x, D) if return_D else (WT_x, None)


def _laplacian_grw_cuda(self, x, guide, sig=None, eps=1e-10):
    """L_rw^T L_rw x, L_rw = I - D^{-1}K, in ONE network pass: the same weight
    stack feeds the forward accumulate (K) and, rolled, the transpose
    accumulate (K^T). The arch reference spends two network passes."""
    if not self.training:
        self.max_batch_shifts = 10
    B, C, H, W = x.shape
    sig = _normalise_sigma_patch(sig, x)

    g = x if guide is None else guide
    w_stack = _kt_weight_stack(self, x, g, sig)
    S = w_stack.shape[0]
    w_all = w_stack.reshape(S * B, C, H, W).contiguous()
    U_num, Z = accumulate_uz(
        x.contiguous(), w_all, _get_shift_tensor(self, x.device))
    z = x - U_num / Z.clamp_min(1e-6)                            # L_rw x
    w_t = _roll_stack(self, w_stack).reshape(S * B, C, H, W).contiguous()
    KT, _ = accumulate_uz(
        (z / Z.clamp_min(1e-6)).contiguous(), w_t,
        _get_kt_shift_tensor(self, x.device))
    if not self.training:
        self.max_batch_shifts = None
    return z - KT


# ---------------------------------------------------------------------------
# Public installer
# ---------------------------------------------------------------------------

def install_cuda_shift(model_cls):
    """Monkey-patch a GASDMoments class to use neural_shift_cuda.

    Idempotent. The patch calls the PixelwiseMomentHead only through its
    stable public API (`logit` / `weight`, both taking the descriptor as the
    fourth argument), so head-internal changes do not affect it.
    """
    if getattr(model_cls, "_cuda_shift_installed", False):
        return model_cls

    # This patch targets the moments-branch GASD (PixelwiseMomentHead).
    for attr in ("_all_shift_weights", "_mix_heads", "pre_activation",
                 "_collect_shifts"):
        if not hasattr(model_cls, attr):
            raise AttributeError(
                f"install_cuda_shift[gasd_moments]: {model_cls.__name__} has "
                f"no `{attr}` -- this patch targets GASDMoments "
                f"(GASD_moments; PixelwiseMomentHead weight producer). For "
                f"the DRUNet-attn GASD use install_cuda_shift_gasd instead.")

    model_cls._get_shift_tensor = _get_shift_tensor
    original_forward = model_cls.forward

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

        # Instance-level architecture check (moment_head is an instance attr).
        if not hasattr(self, "moment_head"):
            raise AttributeError(
                "install_cuda_shift[gasd_moments]: instance has no "
                "`moment_head` -- this patch targets GASDMoments. For the "
                "DRUNet-attn GASD use install_cuda_shift_gasd instead.")

        if getattr(self, "use_cuda_shift", True) and x.is_cuda:
            return _forward_cuda(self, x, guide=guide, sig=sig,
                                 return_D=return_D)
        if _has_return_D:
            return original_forward(self, x, guide=guide, sig=sig,
                                    return_D=return_D)
        return original_forward(self, x, guide=guide, sig=sig)

    model_cls.forward = patched_forward

    # ---- K^T wiring: _KT_action, plus single-network-pass NLM_transpose and
    # laplacian_grw overrides (their arch references each spend TWO network
    # passes: forward for D/Wx, then _KT_action). CPU falls back to the arch
    # reference methods.
    _orig_kt = getattr(model_cls, "_KT_action", None)
    _orig_nlm_t = getattr(model_cls, "NLM_transpose", None)
    _orig_lap_grw = getattr(model_cls, "laplacian_grw", None)

    def _lazy_init(self):
        if not hasattr(self, "_cached_shift_tensor"):
            self.use_cuda_shift = True
            self._cached_shift_tensor = None
            self._cached_shift_device = None

    def patched_kt_action(self, y, guide=None, sig=None):
        _lazy_init(self)
        if getattr(self, "use_cuda_shift", True) and y.is_cuda:
            return _kt_action_cuda(self, y, guide=guide, sig=sig)
        if _orig_kt is None:
            raise AttributeError(
                f"{type(self).__name__} has no reference _KT_action; add the "
                "arch-level _KT_action/NLM_transpose before installing.")
        return _orig_kt(self, y, guide=guide, sig=sig)

    def patched_nlm_transpose(self, x, guide=None, sig=None, return_D=False):
        _lazy_init(self)
        if getattr(self, "use_cuda_shift", True) and x.is_cuda:
            return _nlm_transpose_cuda(self, x, guide=guide, sig=sig,
                                       return_D=return_D)
        if _orig_nlm_t is None:
            raise AttributeError(
                f"{type(self).__name__} has no reference NLM_transpose; add "
                "the arch-level methods before installing.")
        return _orig_nlm_t(self, x, guide=guide, sig=sig, return_D=return_D)

    def patched_laplacian_grw(self, x, guide, sig=None, eps=1e-10):
        _lazy_init(self)
        if getattr(self, "use_cuda_shift", True) and x.is_cuda:
            return _laplacian_grw_cuda(self, x, guide, sig=sig, eps=eps)
        if _orig_lap_grw is None:
            raise AttributeError(
                f"{type(self).__name__} has no reference laplacian_grw.")
        return _orig_lap_grw(self, x, guide, sig=sig, eps=eps)

    model_cls._KT_action = patched_kt_action
    model_cls.NLM_transpose = patched_nlm_transpose
    model_cls.laplacian_grw = patched_laplacian_grw

    model_cls._cuda_shift_installed = True
    return model_cls
