"""Route-1 CUDA patch for the Metropolis NeKDe denoiser (v2, finite window).

Replaces the per-shift Python ``kernel_actions`` accumulation by the fused,
symmetric-by-construction ``metropolis_aggregate`` op. Unlike an ``accumulate_uz``
based patch, the inverse edge here is formed by a circular shift of the
*post-activation* forward weight INSIDE the op, so W is exactly symmetric and
nonexpansive regardless of the output activation (softmax or control-sigmoid).

This patches ``forward`` (not ``kernel_actions``): ``DSG_NLM`` and every
``laplacian_*`` route through ``forward``, so they are all accelerated. The
expensive attention-head evaluation is still produced by the model's own
``_halfplane_weights`` (which batches the head across shifts); the op only
fuses the cheap symmetric accumulation. Training uses the differentiable
PyTorch path inside ``metropolis_aggregate``; ``no_grad`` inference uses the
compiled CUDA kernel when built.

Usage
-----
    from NKD_mp_drunet_attn_v2 import NeKDeMetropolisDRUNetAttn
    from nkd_metropolis_attn_v2_patch import install_cuda_shift

    install_cuda_shift(NeKDeMetropolisDRUNetAttn)
    model = NeKDeMetropolisDRUNetAttn(...).cuda()

Set ``model.use_cuda_shift = False`` to force the reference PyTorch path.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from neural_shift_cuda.metropolis import metropolis_aggregate

# v2 uses a finite-window validity mask (wrap neighbours dropped); v5 is purely
# periodic. This is the ONLY line that differs between the two patch files.
_USE_BOX = True


def _build_w_half(
    self,
    x: torch.Tensor,
    guide: Optional[torch.Tensor],
    sig: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, List[Tuple[int, int, bool]]]:
    """Head-mixed per-half-plane forward weights ``(B, S, C, H, W)``.

    Built with the model's own helpers, so the weights are identical to those
    used by the reference ``kernel_actions`` (same softmax/control-sigmoid
    handling, same head mixing).
    """
    B, C, H, W = x.shape
    R = self.window_rad
    sig = self._normalise_sigma(x, sig)

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
) -> Tuple[torch.Tensor, torch.Tensor]:
    """``forward`` via the fused symmetric Metropolis op.

    Returns ``(W_x, degree_hat)`` exactly as the reference ``forward``.
    """
    if x.dim() != 4:
        raise ValueError(f"x must have shape (B,C,H,W), got {tuple(x.shape)}.")
    w_half, shifts = _build_w_half(self, x, guide, sig)
    eps = max(self.metropolis_eps, torch.finfo(x.dtype).tiny)
    Wx, _d, degree_hat = metropolis_aggregate(
        w_half, x, shifts, use_box=_USE_BOX, eps=eps, checkpoint_train=self.use_grad_checkpoint)
    return Wx, degree_hat


def install_cuda_shift(model_cls):
    """Patch ``model_cls.forward`` with the fused Metropolis path. Idempotent.

    CPU tensors fall back to the original PyTorch ``forward`` (the numerical
    reference). ``DSG_NLM`` and ``laplacian_*`` call ``forward`` and are
    therefore accelerated without further patching.
    """
    if getattr(model_cls, "_metropolis_cuda_forward_installed", False):
        return model_cls
    if not hasattr(model_cls, "forward"):
        raise TypeError("model_cls must provide forward(x, guide=None, sig=None).")

    original_forward = model_cls.forward

    def patched_forward(self, x, guide=None, sig=None):
        if not hasattr(self, "use_cuda_shift"):
            self.use_cuda_shift = True
        if self.use_cuda_shift and x.is_cuda:
            return _forward_cuda(self, x, guide=guide, sig=sig)
        return original_forward(self, x, guide=guide, sig=sig)

    model_cls.forward = patched_forward
    model_cls._metropolis_cuda_forward_installed = True
    return model_cls
