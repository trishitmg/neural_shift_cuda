"""Route-1 CUDA patch for the Metropolis NeKDe moments denoiser
(NeKDeMetropolisMoments, NKD_mp_moments).

The model is the Metropolis-symmetrized NKD whose weight producer is the
GADSD moments pair (TinyMomentFeatureExtractor / FixedMomentFeatureExtractor
+ per-pixel PixelwiseMomentHead) instead of the DRUNet + attention head. Its
kernel assembly is a line-for-line port of NKD_mp_drunet_attn_v2, so this
patch mirrors `nkd_metropolis_attn_patch` structurally.

The model's ``comp_box`` flag selects the boundary handling at runtime
(comp_box=True -> finite-window mask; comp_box=False -> purely periodic), so
a single patch covers both.

Replaces the per-shift Python ``kernel_actions`` accumulation by the fused,
symmetric-by-construction ``metropolis_aggregate`` op. The inverse edge is
formed by a circular shift of the *post-activation* forward weight INSIDE
the op, so W is exactly symmetric and nonexpansive regardless of the output
activation (softmax or control-sigmoid).

This patches ``forward`` (not ``kernel_actions``): ``DSG_NLM`` and every
``laplacian_*`` route through ``forward``, so they are all accelerated. The
expensive moment-head evaluation is still produced by the model's own
``_halfplane_weights`` (which batches the head across shifts AND handles the
per-shift analytic descriptors internally -- no descriptor plumbing is
needed in this patch); the op only fuses the cheap symmetric accumulation.
Training uses the differentiable PyTorch path inside ``metropolis_aggregate``;
``no_grad`` inference uses the compiled CUDA kernel when built.

Differences from ``nkd_metropolis_attn_patch``
----------------------------------------------
* sigma normalization: NKD_mp_moments uses a module-level
  ``_normalise_sigma(sigma, reference)`` helper rather than the
  ``self._normalise_sigma(x, sig)`` method of the attn model, so the patch
  carries its own equivalent helper.
* the architecture check requires ``moment_head`` (instance attr) /
  ``_head_mix_matrix`` and points attn models to the attn patch.

Usage
-----
    from NKD_mp_moments import NeKDeMetropolisMoments
    from neural_shift_cuda.integration import install_cuda_shift_metropolis_moments

    install_cuda_shift_metropolis_moments(NeKDeMetropolisMoments)
    model = NeKDeMetropolisMoments(...).cuda()

Set ``model.use_cuda_shift = False`` to force the reference PyTorch path.
"""

from __future__ import annotations

import inspect
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from neural_shift_cuda.metropolis import metropolis_aggregate

# The model's `comp_box` flag selects the boundary handling at runtime:
#   comp_box=True  -> finite-window validity mask (wrap-around neighbours
#                     dropped); the model's default.
#   comp_box=False -> purely periodic (circulant).
# It is read per-forward and forwarded to `metropolis_aggregate(use_box=...)`.
# Instances without the attribute default to True.


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


def _build_w_half(
    self,
    x: torch.Tensor,
    guide: Optional[torch.Tensor],
    sig: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, List[Tuple[int, int, bool]]]:
    """Head-mixed per-half-plane forward weights ``(B, S, C, H, W)``.

    Built with the model's own helpers, so the weights are identical to those
    used by the reference ``kernel_actions`` (same softmax/control-sigmoid
    handling, same per-shift descriptors, same head mixing).
    """
    B, C, H, W = x.shape
    R = self.window_rad
    sig = _normalise_sigma_patch(sig, x)

    z = x if guide is None else guide
    if z.shape != x.shape:
        raise ValueError(
            f"guide must have the same shape as x; got {tuple(z.shape)} "
            f"and {tuple(x.shape)}.")
    if z.device != x.device or z.dtype != x.dtype:
        raise ValueError("guide and x must share device and dtype.")

    phi = self.pre_activation(z, sigma=sig)
    padded_phi = F.pad(phi, (R, R, R, R), mode="circular")

    shifts = self._collect_shifts()
    weights_list = self._halfplane_weights(phi, padded_phi, sig, shifts)
    head_mix_mat = self._head_mix_matrix()

    w_half = torch.stack(
        [self._mix_heads(weights_list[i], C, head_mix_mat)
         for i in range(len(shifts))],
        dim=1,
    ).contiguous()  # (B, S, C, H, W)
    return w_half, shifts


def _forward_cuda(
    self,
    x: torch.Tensor,
    guide: Optional[torch.Tensor] = None,
    sig: Optional[torch.Tensor] = None,
    return_D: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """``forward`` via the fused symmetric Metropolis op.

    Returns ``(W_x, degree_hat)`` exactly as the reference ``forward``.
    """
    if x.dim() != 4:
        raise ValueError(f"x must have shape (B,C,H,W), got {tuple(x.shape)}.")
    w_half, shifts = _build_w_half(self, x, guide, sig)
    eps = max(self.metropolis_eps, torch.finfo(x.dtype).tiny)
    use_box = bool(getattr(self, "comp_box", True))
    Wx, degree_hat = metropolis_aggregate(
        w_half, x, shifts, use_box=use_box, eps=eps)
    if return_D:
        return Wx, degree_hat
    return Wx, None


def install_cuda_shift(model_cls):
    """Patch ``model_cls.forward`` with the fused Metropolis path. Idempotent.

    CPU tensors fall back to the original PyTorch ``forward`` (the numerical
    reference). ``DSG_NLM`` and ``laplacian_*`` call ``forward`` and are
    therefore accelerated without further patching.
    """
    if getattr(model_cls, "_metropolis_cuda_forward_installed", False):
        return model_cls
    if not hasattr(model_cls, "forward"):
        raise TypeError(
            "model_cls must provide forward(x, guide=None, sig=None).")

    # This patch targets the moments-branch Metropolis NKD.
    for attr in ("_halfplane_weights", "_head_mix_matrix", "_mix_heads",
                 "pre_activation", "_collect_shifts"):
        if not hasattr(model_cls, attr):
            raise AttributeError(
                f"install_cuda_shift[nkd_mp_moments]: {model_cls.__name__} "
                f"has no `{attr}` -- this patch targets NeKDeMetropolisMoments "
                f"(NKD_mp_moments; PixelwiseMomentHead weight producer). For "
                f"the DRUNet-attn Metropolis NKD use "
                f"install_cuda_shift_metropolis instead.")

    original_forward = model_cls.forward

    _fwd_sig = inspect.signature(original_forward)
    _has_return_D = "return_D" in _fwd_sig.parameters
    _return_D_default = (
        _fwd_sig.parameters["return_D"].default if _has_return_D else False)

    def patched_forward(self, x, guide=None, sig=None,
                        return_D=_return_D_default):
        if not hasattr(self, "use_cuda_shift"):
            self.use_cuda_shift = True

        # Instance-level architecture check (moment_head is an instance attr).
        if not hasattr(self, "moment_head"):
            raise AttributeError(
                "install_cuda_shift[nkd_mp_moments]: instance has no "
                "`moment_head` -- this patch targets NeKDeMetropolisMoments. "
                "For the DRUNet-attn Metropolis NKD use "
                "install_cuda_shift_metropolis instead.")

        if self.use_cuda_shift and x.is_cuda:
            return _forward_cuda(self, x, guide=guide, sig=sig,
                                 return_D=return_D)
        if _has_return_D:
            return original_forward(self, x, guide=guide, sig=sig,
                                    return_D=return_D)
        return original_forward(self, x, guide=guide, sig=sig)

    model_cls.forward = patched_forward
    model_cls._metropolis_cuda_forward_installed = True
    return model_cls
