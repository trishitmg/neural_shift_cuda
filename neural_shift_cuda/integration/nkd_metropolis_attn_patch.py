"""Route-1 CUDA patch for the Metropolis NeKDe denoiser.

The model's ``comp_box`` flag selects the boundary handling at runtime
(comp_box=True -> finite-window mask, the old v2; comp_box=False -> purely
periodic, the old v5), so a single patch covers both.

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
    from nkd_metropolis_attn_patch import install_cuda_shift

    install_cuda_shift(NeKDeMetropolisDRUNetAttn)
    model = NeKDeMetropolisDRUNetAttn(...).cuda()

Set ``model.use_cuda_shift = False`` to force the reference PyTorch path.
"""

from __future__ import annotations

import inspect
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from neural_shift_cuda import accumulate_uz
from neural_shift_cuda.metropolis import metropolis_aggregate

# The model's `comp_box` flag selects the boundary handling at runtime (this
# replaces the former separate v5 patch):
#   comp_box=True  -> finite-window validity mask (wrap-around neighbours
#                     dropped); the old v2 behaviour.
#   comp_box=False -> purely periodic (circulant); the old v5 behaviour.
# It is read per-forward and forwarded to `metropolis_aggregate(use_box=...)`.
# Instances without the attribute default to True (the historical behaviour).


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


def _cached_metropolis_reduction_inputs(self, x: torch.Tensor, cache):
    """Pack cached Metropolis edges once for the parallel K_hat reduction."""
    if x.dim() != 4:
        raise ValueError(f"x must have shape (B,C,H,W), got {tuple(x.shape)}.")
    if x.size(1) != cache.C:
        raise ValueError(
            f"cached weights have C={cache.C} channels but x has "
            f"{x.size(1)}; rebuild the cache for this channel count.")

    state = getattr(self, "_metropolis_parallel_weight_cache", None)
    shape_key = tuple(x.shape)
    if (state is not None and state[0] is cache and state[1] == shape_key
            and state[2] == x.device and state[3] == x.dtype):
        return state[4], state[5]

    shifts = list(cache.shifts)
    weights = list(cache.w_hat_fwd_list)
    if len(weights) != len(shifts) or not weights:
        raise ValueError(
            "cached w_hat_fwd_list must contain one tensor per cached shift.")
    for weight in weights:
        if tuple(weight.shape) != shape_key:
            raise ValueError(
                "cached weights and x must have identical (B,C,H,W) shapes; "
                f"got {tuple(weight.shape)} and {shape_key}.")
        if weight.device != x.device or weight.dtype != x.dtype:
            raise ValueError("cached weights and x must share device and dtype.")

    w_stack = torch.stack(weights, dim=0)  # (S,B,C,H,W), packed once per guide
    if bool(cache.comp_box):
        rows = [(int(dx), int(dy), int(bool(has_inv)))
                for dx, dy, has_inv in shifts]
        w_all = w_stack.reshape(-1, *x.shape[1:]).contiguous()
    else:
        # Periodic cached weights need an explicit shifted twin because the
        # accumulate_uz inverse flag implements finite-window masking.
        entries: List[torch.Tensor] = []
        rows: List[Tuple[int, int, int]] = []
        for i, (dx, dy, has_inv) in enumerate(shifts):
            w_fwd = w_stack[i]
            entries.append(w_fwd)
            rows.append((int(dx), int(dy), 0))
            if has_inv:
                entries.append(torch.roll(
                    w_fwd, shifts=(int(dx), int(dy)), dims=(-2, -1)))
                rows.append((-int(dx), -int(dy), 0))
        w_all = torch.stack(entries, dim=0).reshape(
            -1, *x.shape[1:]).contiguous()

    shifts_t = torch.tensor(rows, dtype=torch.int64, device=x.device)
    self._metropolis_parallel_weight_cache = (
        cache, shape_key, x.device, x.dtype, w_all, shifts_t)
    return w_all, shifts_t


def _forward_cached_cuda(
    self,
    x: torch.Tensor,
    cache,
    return_D: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Parallel cached Metropolis forward with one fused reduction launch.

    The cache already contains the post-Metropolis edges ``w_hat_fwd`` and
    ``degree_hat``.  Consequently this path performs only K_hat x and the
    inexpensive diagonal/self-loop combination; it never reruns either the
    attention network or the two-pass Metropolis construction.
    """
    w_all, shifts_t = _cached_metropolis_reduction_inputs(self, x, cache)
    K_hat_x, degree_reduced = accumulate_uz(
        x.contiguous(), w_all, shifts_t)
    degree_hat = (
        cache.degree_hat
        if getattr(cache, "degree_hat", None) is not None
        else degree_reduced)
    Wx = x - degree_hat * x + K_hat_x
    return (Wx, degree_hat) if return_D else (Wx, None)


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

    original_forward = model_cls.forward
    original_forward_cached = getattr(model_cls, "forward_cached", None)

    # Derive return_D's default from the wrapped forward so the patched
    # signature stays in lockstep with the model instead of hardcoding it, and
    # only forward return_D to the reference path when that forward accepts it.
    _fwd_sig = inspect.signature(original_forward)
    _has_return_D = "return_D" in _fwd_sig.parameters
    _return_D_default = (
        _fwd_sig.parameters["return_D"].default if _has_return_D else False)

    def patched_forward(self, x, guide=None, sig=None,
                        return_D=_return_D_default):
        if not hasattr(self, "use_cuda_shift"):
            self.use_cuda_shift = True
        if self.use_cuda_shift and x.is_cuda:
            return _forward_cuda(self, x, guide=guide, sig=sig,
                                 return_D=return_D)
        if _has_return_D:
            return original_forward(self, x, guide=guide, sig=sig,
                                    return_D=return_D)
        return original_forward(self, x, guide=guide, sig=sig)

    model_cls.forward = patched_forward

    # Optional cached-forward wiring keeps compatibility with pre-cache model
    # versions while accelerating every cached Laplacian in the supplied file.
    if original_forward_cached is not None:
        _cached_sig = inspect.signature(original_forward_cached)
        _cached_has_return_D = "return_D" in _cached_sig.parameters
        _cached_return_D_default = (
            _cached_sig.parameters["return_D"].default
            if _cached_has_return_D else False)

        def patched_forward_cached(
                self, x, cache, return_D=_cached_return_D_default):
            if not hasattr(self, "use_cuda_shift"):
                self.use_cuda_shift = True
            if getattr(self, "use_cuda_shift", True) and x.is_cuda:
                return _forward_cached_cuda(
                    self, x, cache, return_D=return_D)
            if _cached_has_return_D:
                return original_forward_cached(
                    self, x, cache, return_D=return_D)
            return original_forward_cached(self, x, cache)

        model_cls.forward_cached = patched_forward_cached

    model_cls._metropolis_cuda_forward_installed = True
    return model_cls
