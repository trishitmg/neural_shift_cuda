"""CUDA shift patch for :mod:`nkd_metropolis_attn_v2`.

The base module evaluates the Metropolis operator in two symmetric passes:

    K x, d             = raw symmetric accumulation,
    K_hat x, d_hat     = accumulation after
                         w_ij <- w_ij / max(d_i, d_j),
    W x                = (1 - d_hat) * x + K_hat x.

This patch preserves exactly that mathematics while replacing the expensive
padding/slicing loops by the existing ``neural_shift_cuda`` primitives:

* ``shift_gather`` gathers all shifted guide features and validity masks;
* ``accumulate_uz`` applies the forward half-plane weights together with their
  inverse-consistent reverse copies;
* a second ``shift_gather`` gathers the raw degree at the neighbouring endpoint
  of every forward edge, allowing all Metropolis denominators
  ``max(d_i, d_j)`` to be computed in parallel.

No final row normalization is performed.  The second accumulator returns
``K_hat x`` and ``d_hat = K_hat e``; the model's unchanged ``forward`` method
then performs the diagonal completion that makes W symmetric stochastic.

Usage
-----

    from nkd_metropolis_attn_v2 import NKDMetropolisAttnV2
    from nkd_metropolis_attn_v2_patch import install_cuda_shift

    install_cuda_shift(NKDMetropolisAttnV2)
    model = NKDMetropolisAttnV2(...).cuda()

Set ``model.use_cuda_shift = False`` to force the reference PyTorch path for a
particular instance.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint

from neural_shift_cuda import accumulate_uz, shift_gather


# ---------------------------------------------------------------------------
# Cached shift metadata and vectorized head mixing.
# ---------------------------------------------------------------------------


def _get_shift_tensor(self, device: torch.device) -> torch.Tensor:
    """Return the cached shift table ``(dx, dy, has_inverse)`` on ``device``."""
    cached = getattr(self, "_cached_shift_tensor", None)
    cached_device = getattr(self, "_cached_shift_device", None)
    if cached is None or cached_device != device:
        rows = [
            (int(dx), int(dy), int(bool(has_inverse)))
            for dx, dy, has_inverse in self._collect_shifts()
        ]
        cached = torch.tensor(rows, dtype=torch.int32, device=device)
        self._cached_shift_tensor = cached
        self._cached_shift_device = device
    return cached


def _mix_heads_vec(
    w_stack: torch.Tensor,
    channels: int,
    head_mix_mat: Optional[torch.Tensor],
) -> torch.Tensor:
    """Map ``(S,B,M,H,W)`` head weights to ``(S,B,C,H,W)`` channel weights."""
    n_heads = w_stack.size(2)
    if n_heads == 1:
        return w_stack.expand(-1, -1, channels, -1, -1)
    if head_mix_mat is not None:
        return torch.einsum(
            "cm,sbmij->sbcij", head_mix_mat, w_stack
        ).contiguous()
    if n_heads == channels:
        return w_stack
    return w_stack.mean(dim=2, keepdim=True).expand(
        -1, -1, channels, -1, -1)


# ---------------------------------------------------------------------------
# CUDA implementation of NKDMetropolisAttnV2.kernel_actions.
# ---------------------------------------------------------------------------


def _kernel_actions_cuda(
    self,
    x: torch.Tensor,
    guide: Optional[torch.Tensor] = None,
    sig: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Parallel application of ``K`` and ``K_hat``.

    Returns ``(K_x, degree, K_hat_x, degree_hat)`` with the same shapes and
    semantics as the reference ``kernel_actions`` method.
    """
    if x.dim() != 4:
        raise ValueError(f"x must have shape (B,C,H,W), got {tuple(x.shape)}.")
    if not x.is_cuda:
        raise ValueError("_kernel_actions_cuda requires a CUDA tensor.")

    B, C, H, W = x.shape
    sig = self._normalise_sigma(x, sig)

    z = x if guide is None else guide
    if z.shape != x.shape:
        raise ValueError(
            f"guide must have the same shape as x; got {tuple(z.shape)} "
            f"and {tuple(x.shape)}.")
    if z.device != x.device or z.dtype != x.dtype:
        raise ValueError("guide and x must have the same CUDA device and dtype.")

    phi = self.pre_activation(z, sigma=sig).contiguous()

    shifts_t = self._get_shift_tensor(x.device)      # (S,3), int32
    shifts_xy = shifts_t[:, :2].contiguous()         # (S,2), int32
    n_shifts = shifts_t.size(0)

    configured_chunk = self.max_batch_shifts
    if configured_chunk is None:
        chunk = n_shifts
    else:
        chunk = max(1, min(int(configured_chunk), n_shifts))
    # Evaluation commonly uses larger images.  Limit only the local chunk size;
    # do not mutate self.max_batch_shifts or discard the user's configuration.
    if not self.training:
        chunk = min(chunk, 10)

    head_mix_mat = self._head_mix_matrix()

    # ------------------------------------------------------------------
    # Compute forward-half-plane neural weights.
    # ------------------------------------------------------------------
    weight_tiles: List[torch.Tensor] = []
    mask_tiles: List[torch.Tensor] = []

    for start in range(0, n_shifts, chunk):
        end = min(start + chunk, n_shifts)
        n = end - start
        shift_chunk = shifts_xy[start:end]

        # Shift-major layout: [shift_0 batch, shift_1 batch, ...].
        phi_shift, mask = shift_gather(phi, shift_chunk)
        phi_center = phi.repeat(n, 1, 1, 1)
        sig_batch = sig.repeat(n, 1, 1, 1) if sig is not None else None

        if self.output_activation == "softmax":
            if self.use_grad_checkpoint:
                values = grad_checkpoint(
                    self.attn_head.logit,
                    phi_center,
                    phi_shift,
                    sig_batch,
                    use_reentrant=False,
                )
            else:
                values = self.attn_head.logit(
                    phi_center, phi_shift, sig_batch)
        else:
            if self.use_grad_checkpoint:
                values = grad_checkpoint(
                    self.attn_head.weight,
                    phi_center,
                    phi_shift,
                    sig_batch,
                    use_reentrant=False,
                )
            else:
                values = self.attn_head.weight(
                    phi_center, phi_shift, sig_batch)

        weight_tiles.extend(values.split(B, dim=0))
        mask_tiles.append(mask)

    if self.output_activation == "softmax":
        # The neural softmax is only over the predicted half-plane weights.  It
        # is not the stochastic normalization of W; Metropolis completion below
        # supplies the operator-level stochasticity.
        logits = torch.stack(weight_tiles, dim=1)  # (B,S,M,H,W)
        logits = logits - logits.amax(dim=1, keepdim=True)
        weights = F.softmax(logits, dim=1)
    else:
        weights = torch.stack(weight_tiles, dim=1)

    # (B,S,M,H,W) -> (S,B,M,H,W) -> (S,B,C,H,W).
    weights = weights.permute(1, 0, 2, 3, 4).contiguous()
    weights = _mix_heads_vec(weights, C, head_mix_mat)

    # Flatten in the same shift-major layout used by shift_gather and the CUDA
    # accumulator.  Invalid finite-window edges are set to zero.
    w_raw = weights.reshape(n_shifts * B, C, H, W)
    validity = torch.cat(mask_tiles, dim=0)  # (S*B,1,H,W)
    w_raw = (w_raw * validity).contiguous()

    # ------------------------------------------------------------------
    # Pass 1: symmetric raw kernel K and d = K e.
    # ------------------------------------------------------------------
    K_x, degree = accumulate_uz(x.contiguous(), w_raw, shifts_t)

    # ------------------------------------------------------------------
    # Parallel Metropolis edge scaling.
    #
    # For every forward edge i -> i+delta, gather d(i+delta) and divide the
    # associated raw edge by max(d(i), d(i+delta)).  The accumulator generates
    # the reverse contribution by shifting this already-normalized forward
    # tensor, so both directions use the identical pairwise denominator.
    # ------------------------------------------------------------------
    degree_shift, degree_validity = shift_gather(
        degree.contiguous(), shifts_xy)
    degree_shift = degree_shift.view(n_shifts, B, C, H, W)
    eps = max(self.metropolis_eps, torch.finfo(degree.dtype).tiny)
    denominator = torch.maximum(
        degree.unsqueeze(0), degree_shift
    ).clamp_min(eps)

    w_hat = w_raw.view(n_shifts, B, C, H, W) / denominator
    w_hat = w_hat.reshape(n_shifts * B, C, H, W)
    # ``w_raw`` is already zero outside the graph.  Multiplying by the degree
    # gather mask as well makes the intended finite-window support explicit.
    w_hat = (w_hat * degree_validity).contiguous()

    # Pass 2: K_hat x and d_hat = K_hat e.
    K_hat_x, degree_hat = accumulate_uz(
        x.contiguous(), w_hat, shifts_t)

    return K_x, degree, K_hat_x, degree_hat


# ---------------------------------------------------------------------------
# Installer.
# ---------------------------------------------------------------------------


def install_cuda_shift(model_cls):
    """Patch ``model_cls.kernel_actions`` with the CUDA Metropolis path.

    The installer is idempotent.  CPU tensors continue to use the original
    PyTorch implementation, which also serves as a numerical reference.
    """
    if getattr(model_cls, "_metropolis_cuda_shift_installed", False):
        return model_cls
    if not hasattr(model_cls, "kernel_actions"):
        raise TypeError(
            "model_cls must provide kernel_actions(x, guide=None, sig=None).")

    original_kernel_actions = model_cls.kernel_actions
    model_cls._get_shift_tensor = _get_shift_tensor

    def patched_kernel_actions(self, x, guide=None, sig=None):
        if not hasattr(self, "use_cuda_shift"):
            self.use_cuda_shift = True
            self._cached_shift_tensor = None
            self._cached_shift_device = None

        if self.use_cuda_shift and x.is_cuda:
            return _kernel_actions_cuda(self, x, guide=guide, sig=sig)
        return original_kernel_actions(self, x, guide=guide, sig=sig)

    model_cls.kernel_actions = patched_kernel_actions
    model_cls._metropolis_cuda_shift_installed = True
    return model_cls
