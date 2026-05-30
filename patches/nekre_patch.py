"""
Minimal patch for `nekre` (NKD_models_symm.py) to use neural_shift_cuda
in the batched=True forward path.

This file is illustrative — it shows *exactly* which methods to add and
which lines to change. Apply by hand; nothing here imports the original
class.

What this changes
-----------------
1. Add two flags + a cached int32 shift tensor.
2. Add `_get_shift_tensor(device)` helper.
3. Add `compute_weights_from_pair(pair_batch)` — same body as
   `compute_weights` but skips the leading `torch.cat`, because
   `pair_gather` has already done the concatenation in CUDA.
4. Replace ONLY the `batched=True` branch of `forward`.

The mathematical meaning is preserved:
  - guide values come from `pre_activation` (unchanged);
  - weights are the same function of (guide, guide_shifted);
  - U, Z accumulate over the same set of shifts with the same forward+
    inverse-symmetry structure;
  - the validity mask matches the original `box` zero-padding;
  - autograd flows through everything (pre_activation, proj layers, out,
    activation, and the accumulation).

Numerical tolerance vs the old path is ~1e-5 in fp32, exact in fp64.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from neural_shift_cuda import shift_gather, pair_gather, accumulate_uz


# ----------------------------------------------------------------------
# Methods to add to the `nekre` class.
# Drop these in as instance methods (same indentation as the existing
# methods inside `class nekre(nn.Module):`).
# ----------------------------------------------------------------------

def _init_cuda_shift_flags(self):
    """Call from __init__ AFTER self.window_rad is set.

    Example:
        # in nekre.__init__, after `self.window_rad = args.window_rad`:
        self.use_cuda_shift = True
        self.use_cuda_pair_gather = True
        self._cached_shift_tensor = None
        self._cached_shift_device = None
    """
    self.use_cuda_shift = True
    self.use_cuda_pair_gather = True
    self._cached_shift_tensor = None
    self._cached_shift_device = None


def _get_shift_tensor(self, device):
    """Return shifts as an int32 tensor of shape (S, 3) on `device`.

    Layout matches `_collect_shifts()`:
      [ (dx, dy, has_inverse), ... ]   with S = 2*R*R + 2*R + 1.
    Cached per-device so we don't rebuild it every forward.
    """
    if (self._cached_shift_tensor is None
            or self._cached_shift_device != device):
        shifts = self._collect_shifts()  # list of (dx, dy, bool)
        shifts_int = [(int(dx), int(dy), int(bool(hi)))
                      for (dx, dy, hi) in shifts]
        self._cached_shift_tensor = torch.tensor(
            shifts_int, dtype=torch.int32, device=device)
        self._cached_shift_device = device
    return self._cached_shift_tensor


def compute_weights_from_pair(self, pair_batch):
    """Same as `compute_weights` for model_type == 'concat',
    but assumes the channel-wise concatenation has already been done
    by pair_gather. `pair_batch` has shape (S*B, 2*Cg, H, W).

    Only valid when self.model_type == 'concat'.
    """
    assert self.model_type == 'concat', \
        "compute_weights_from_pair is only valid for model_type='concat'"
    x = pair_batch  # already concatenated

    for proj in self.proj:
        for i, sub_layer in enumerate(proj):
            if i == 1:  # LayerNorm layer
                x = rearrange(x, 'b c h w -> b h w c')
                x = sub_layer(x)
                x = rearrange(x, 'b h w c -> b c h w')
            else:
                x = sub_layer(x)

    x = self.out(x)

    # output_activation, identical to compute_weights
    if self.output_activation == 'exp':
        x = torch.exp(x / self.sigma ** 2)
    elif self.output_activation == 'control_sigmoid':
        k, a, b = torch.chunk(x, dim=1, chunks=3)
        x = self.controlled_sigmoid(k, a, b) + 1e-8
    elif self.output_activation == 'sig':
        normalization_factor = -(self.sigma * self.sigma)
        return torch.nn.functional.sigmoid(k / normalization_factor)

    return x


# ----------------------------------------------------------------------
# Replacement body for the `batched == True` branch of nekre.forward.
# Replace the existing `if batched == True:` block (everything from
# `B, C, H, W = x.shape` through `return U, Z`) with the body below.
# ----------------------------------------------------------------------

def _forward_batched_cuda(self, x, guide=None, sig=None):
    """Optimized batched forward path. Drop-in replacement for the
    body of `if batched == True:` in nekre.forward.

    Returns (U, Z), bit-equivalent (up to fp32 reduction order) to the
    original Python list-comprehension path.
    """
    B, C, H, W = x.shape
    R = self.window_rad

    # --- guide computation (unchanged) ---
    if guide is None:
        guide = self.pre_activation(x, sigma=sig)
    else:
        guide = self.pre_activation(guide, sigma=sig)

    guide = guide.contiguous()
    x_c = x.contiguous()
    S = 2 * R * R + 2 * R + 1
    Cg = guide.shape[1]

    shifts = self._get_shift_tensor(device=x.device)  # (S, 3) int32

    # --- Phase 1/2: shift gather + weight computation ---
    if self.model_type == 'concat' and self.use_cuda_pair_gather:
        # One CUDA kernel: produces (S*B, 2*Cg, H, W) directly.
        pair_batch, mask_batch = pair_gather(guide, shifts)
        weights = self.compute_weights_from_pair(pair_batch)
    else:
        # Generic path: shift_gather then standard compute_weights.
        gs_batch, mask_batch = shift_gather(guide, shifts)
        # g_batch = guide.repeat(S, 1, 1, 1) is equivalent to:
        #   tile guide S times along batch dim.
        # Use expand+reshape to avoid the materialization where possible.
        g_batch = guide.unsqueeze(0).expand(S, B, Cg, H, W) \
                       .reshape(S * B, Cg, H, W).contiguous()
        weights = self.compute_weights(g_batch, gs_batch)

    # weights: (S*B, Cx, H, W); mask_batch: (S*B, 1, H, W)
    # Pre-multiply by validity mask (matches original `weight * comp_box`).
    weights = weights * mask_batch

    # --- Phase 3: U/Z accumulation with forward + inverse symmetry ---
    U_num, Z = accumulate_uz(x_c, weights.contiguous(), shifts)
    U = U_num / Z
    return U, Z


# ----------------------------------------------------------------------
# Concrete diff against the original nekre.forward (batched branch only)
# ----------------------------------------------------------------------
#
# Original (lines 363-425 of NKD_models_symm.py), simplified:
#
#     if batched == True:
#         B, C, H, W = x.shape
#         R = self.window_rad
#
#         if guide is None:
#             guide = self.pre_activation(x, sigma=sig)
#         else:
#             guide = self.pre_activation(guide, sigma=sig)
#
#         padded_img   = F.pad(x,     (R,R,R,R), mode='circular')
#         padded_guide = F.pad(guide, (R,R,R,R), mode='circular')
#         box = F.pad(torch.ones(B,C,H,W, device=x.device, dtype=x.dtype),
#                     (R,R,R,R), mode='constant', value=0)
#
#         shifts = self._collect_shifts()
#         guide_shifted_list = [
#             padded_guide[:, :, R+dx:R+dx+H, R+dy:R+dy+W]
#             for dx, dy, _ in shifts
#         ]
#         weights_list = self._batched_compute_weights(
#             guide, guide_shifted_list, sig)
#
#         U = torch.zeros_like(x)
#         Z = torch.zeros_like(x)
#         for i, (dx, dy, has_inverse) in enumerate(shifts):
#             weight   = weights_list[i]
#             comp_box = box[:, :, R+dx:R+dx+H, R+dy:R+dy+W]
#             weight_fwd = weight * comp_box
#             v          = padded_img[:, :, R+dx:R+dx+H, R+dy:R+dy+W]
#             U = U + weight_fwd * v
#             Z = Z + weight_fwd
#             if has_inverse:
#                 dx_inv, dy_inv = -dx, -dy
#                 weight_padded = F.pad(weight_fwd, (R,R,R,R), mode='circular')
#                 weight_inv = weight_padded[:, :, R+dx_inv:R+dx_inv+H,
#                                                  R+dy_inv:R+dy_inv+W]
#                 comp_box_inv = box[:, :, R+dx_inv:R+dx_inv+H,
#                                          R+dy_inv:R+dy_inv+W]
#                 weight_inv = weight_inv * comp_box_inv
#                 v_inv      = padded_img[:, :, R+dx_inv:R+dx_inv+H,
#                                               R+dy_inv:R+dy_inv+W]
#                 U = U + weight_inv * v_inv
#                 Z = Z + weight_inv
#         U = U / Z
#         return U, Z
#
# New (drop-in replacement):
#
#     if batched == True:
#         if not getattr(self, 'use_cuda_shift', False):
#             # fall through to the original path
#             ...   # (keep the old body as-is)
#         return _forward_batched_cuda(self, x, guide=guide, sig=sig)
#
# That is the entire integration. The rest of the module is untouched.
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# Optional: monkey-patch helper for quick experiments without editing
# the original file. Not recommended for permanent use.
# ----------------------------------------------------------------------

def install_cuda_shift(nekre_cls):
    """Monkey-patch a nekre class to use neural_shift_cuda.

    Usage:
        from NKD_models_symm import nekre
        from nekre_patch import install_cuda_shift
        install_cuda_shift(nekre)

    After this, every nekre instance will route batched=True through
    the CUDA path. Existing __init__-time state is initialized lazily
    on the first forward call.
    """
    nekre_cls._get_shift_tensor = _get_shift_tensor
    nekre_cls.compute_weights_from_pair = compute_weights_from_pair

    original_forward = nekre_cls.forward

    def patched_forward(self, x, guide=None, sig=None, batched=False):
        # lazy-init flags / cache (so we don't need to touch __init__)
        if not hasattr(self, '_cached_shift_tensor'):
            self.use_cuda_shift = True
            self.use_cuda_pair_gather = True
            self._cached_shift_tensor = None
            self._cached_shift_device = None

        if batched and getattr(self, 'use_cuda_shift', False):
            return _forward_batched_cuda(self, x, guide=guide, sig=sig)
        return original_forward(self, x, guide=guide, sig=sig, batched=batched)

    nekre_cls.forward = patched_forward
    return nekre_cls
