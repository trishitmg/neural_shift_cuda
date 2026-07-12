"""
Integration patch for NeKDeMoments (NKD_moments) to use neural_shift_cuda
kernels.

What this is
------------
A drop-in replacement for the `forward` method of `NeKDeMoments` -- the
NeKDe port whose weight producer is the GADSD moments pair
(TinyMomentFeatureExtractor / FixedMomentFeatureExtractor +
per-pixel PixelwiseMomentHead) instead of the DRUNet + attention head.
Same math, same gradients, but:
  * the F.pad + Python list-comprehension shift gather inside
    `_halfplane_weights` is replaced by `shift_gather` (one CUDA kernel
    per chunk, no padded_phi materialization),
  * the per-shift U/Z accumulation loop in `forward` (including the
    circular-shift inverse-symmetry branch) is replaced by a single
    `accumulate_uz` kernel call.

The kernel assembly of NKD_moments is a line-for-line port of
NKD_drunet_attn_v2 (same `_collect_shifts`, same symmetric half-plane
accumulation, same `_mix_heads`), so this patch mirrors
`nekde_drunet_attn_patch` structurally. The ONLY functional difference is
the weight-head API: `PixelwiseMomentHead.logit / weight` take a FOURTH
argument -- the per-shift analytic descriptor -- so each chunk additionally
slices `self.shift_descriptors` and repeat-interleaves it across the batch
axis (`descriptors[start:end].repeat_interleave(B, dim=0)`), which matches
the shift-major ordering of `shift_gather`'s (n*B, F, H, W) output exactly.

comp_box handling
-----------------
The model's `comp_box` flag is honoured on the CUDA path; both values are
fused, no reference fallback:
  * comp_box=True  -> truncated NLM. Forward edge masked by shift_gather's
    validity mask; inverse edge masked by accumulate_uz's built-in has_inverse
    boundary check. shifts_t carries has_inverse=1 on the half-plane.
  * comp_box=False -> fully periodic (circulant) NLM. accumulate_uz's
    has_inverse branch always masks the inverse edge, so it cannot express the
    periodic operator directly. Instead we enumerate each half-plane shift AND
    its explicit inverse (-dx,-dy), both with has_inverse=0, feeding the kernel
    the pre-shifted inverse weight w_inv = roll(w_fwd,(dx,dy)). The kernel then
    performs forward-only circular gathers with NO masking, reproducing the
    periodic symmetric operator bit-for-bit. Costs ~2x shift entries.
Instances without the attribute default to True (the model's default).

What this is NOT
----------------
This does not touch the PixelwiseMomentHead. Its `logit` / `weight` methods
take phi_c and phi_s as SEPARATE tensors (the moments |phi_c - phi_s|,
(phi_c - phi_s)^2, phi_c * phi_s are formed inside), so `pair_gather`
(channel-concat fusion) is not applicable here -- same reasoning as the
attn patches.

Usage
-----
    from NKD_moments import NeKDeMoments
    from neural_shift_cuda.integration import install_cuda_shift_nekde_moments
    install_cuda_shift_nekde_moments(NeKDeMoments)

After this, every NeKDeMoments instance routes its forward through the CUDA
path when the input is on a CUDA device. To disable per-instance:

    model.use_cuda_shift = False
"""

from __future__ import annotations

import inspect
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from neural_shift_cuda import shift_gather, accumulate_uz


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
    `head_mix_pos_act`. Matches NeKDeMoments.forward exactly."""
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

    Mirrors NeKDeMoments._mix_heads case-for-case:
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
    return_D: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """CUDA-accelerated drop-in replacement for NeKDeMoments.forward.

    Numerics: identical to the original up to floating-point reduction
    order (~1e-5 in fp32, exact in fp64).
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

    # ---- Shift tensor + analytic descriptors ----
    # Module-level call (not self._get_shift_tensor) so _forward_cuda also
    # works when invoked directly, before install_cuda_shift attached it.
    shifts_t = _get_shift_tensor(self, x.device)                 # (S, 3) int32
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
    # Per-chunk weight computation.
    #
    # For each chunk:
    #   1. shift_gather(phi, shifts_chunk) -> (n*B, F, H, W) gathered phi_s,
    #      plus (n*B, 1, H, W) validity mask. Replaces the
    #      [padded_phi[:, :, R+dx:R+dx+H, R+dy:R+dy+W] for ...] slices.
    #   2. phi_c_batch = phi.repeat(n, 1, 1, 1) (unchanged).
    #   3. desc_batch  = descriptors[start:end].repeat_interleave(B, dim=0)
    #      -- shift-major (n*B, D), aligned with shift_gather's output order.
    #   4. moment_head.logit / weight as in the reference _halfplane_weights.
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

    # ---- Joint softmax across the half-plane shifts (softmax path only) ----
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
        #      pre-shifted inverse weight. ----
        shift_list = self._collect_shifts()                      # [(dx,dy,has_inv)]
        entries: List[torch.Tensor] = []
        rows: List[Tuple[int, int, int]] = []
        for i, (dx, dy, has_inv) in enumerate(shift_list):
            w_fwd = w_stack[i]                                   # (B, C, H, W)
            entries.append(w_fwd)
            rows.append((int(dx), int(dy), 0))
            if has_inv:
                # w_inv[h,w] = w_fwd[(h-dx) % H, (w-dy) % W] == roll by (dx,dy).
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
# Public installer
# ---------------------------------------------------------------------------

def install_cuda_shift(model_cls):
    """Monkey-patch a NeKDeMoments class to use neural_shift_cuda.

    Idempotent. The patch calls the PixelwiseMomentHead only through its
    stable public API (`logit` / `weight`, both taking the descriptor as the
    fourth argument), so head-internal changes do not affect it.
    """
    if getattr(model_cls, "_cuda_shift_installed", False):
        return model_cls

    # This patch targets the moments-branch NeKDe (PixelwiseMomentHead).
    for attr in ("_halfplane_weights", "_mix_heads", "pre_activation",
                 "_collect_shifts"):
        if not hasattr(model_cls, attr):
            raise AttributeError(
                f"install_cuda_shift[nekde_moments]: {model_cls.__name__} has "
                f"no `{attr}` -- this patch targets NeKDeMoments "
                f"(NKD_moments; PixelwiseMomentHead weight producer). For the "
                f"DRUNet-attn NeKDe use install_cuda_shift_attn instead.")

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
                "install_cuda_shift[nekde_moments]: instance has no "
                "`moment_head` -- this patch targets NeKDeMoments. For the "
                "DRUNet-attn NeKDe use install_cuda_shift_attn instead.")

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
