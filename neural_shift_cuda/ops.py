# neural_shift_cuda/ops.py
#
# High-level Python API.
#
#   shift_gather(guide, shifts)         -> (gs_batch, mask_batch)
#   pair_gather(guide, shifts)          -> (pair_batch, mask_batch)
#   accumulate_uz(x, weights, shifts)   -> (U_num, Z)
#
# Each op:
#   * dispatches to the CUDA kernels (extension `_C`) when guide/x are on CUDA;
#   * falls back to a pure-PyTorch reference impl otherwise (and exposes the
#     reference impl as `*_reference` for tests).
#
# All three ops are differentiable via torch.autograd.Function. The reference
# impls are also differentiable because they are pure PyTorch.

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from neural_shift_cuda import _C
    _HAS_CUDA_EXT = True
except Exception:  # pragma: no cover -- CPU-only env / build failure
    _C = None
    _HAS_CUDA_EXT = False


# ---------------------------------------------------------------------------
# Reference (PyTorch) implementations -- used as CPU fallback and as the
# numerical ground truth in the test suite.
# ---------------------------------------------------------------------------

def _wrap_shifts(shifts: torch.Tensor) -> torch.Tensor:
    if shifts.dim() != 2 or shifts.size(1) not in (2, 3):
        raise ValueError(
            f"shifts must be (S, 2) or (S, 3); got {tuple(shifts.shape)}")
    return shifts.to(torch.int64)


def shift_gather_reference(
    guide: torch.Tensor,
    shifts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for `shift_gather`.

    Reproduces exactly what the original `nekre.forward(..., batched=True)`
    code does with F.pad + slice. Returns:

        gs_batch  : (S*B, C, H, W)  -- circular-shifted guide
        mask      : (S*B, 1, H, W)  -- 1 where the unshifted (h, w) is in bounds
    """
    shifts = _wrap_shifts(shifts)
    B, C, H, W = guide.shape
    S = shifts.size(0)
    R = int(shifts[:, :2].abs().max().item())

    if R == 0:
        gs_batch = guide.repeat(S, 1, 1, 1).contiguous()
        mask = torch.ones(
            S * B, 1, H, W, device=guide.device, dtype=guide.dtype)
        return gs_batch, mask

    padded_guide = F.pad(guide, (R, R, R, R), mode='circular')
    box = F.pad(torch.ones(1, 1, H, W, device=guide.device, dtype=guide.dtype),
                (R, R, R, R), mode='constant', value=0)

    gs_list = []
    mask_list = []
    for i in range(S):
        dx = int(shifts[i, 0].item())
        dy = int(shifts[i, 1].item())
        gs_list.append(
            padded_guide[:, :, R + dx:R + dx + H, R + dy:R + dy + W])
        mask_list.append(
            box[:, :, R + dx:R + dx + H, R + dy:R + dy + W].expand(B, 1, H, W)
        )

    gs_batch = torch.cat(gs_list, dim=0).contiguous()
    mask = torch.cat(mask_list, dim=0).contiguous()
    return gs_batch, mask


def pair_gather_reference(
    guide: torch.Tensor,
    shifts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for `pair_gather`.

    Returns (pair_batch, mask) where pair_batch[s*B+b, :, :, :] is
    [guide[b]; shifted_guide_s[b]] along the channel dim.
    """
    gs_batch, mask = shift_gather_reference(guide, shifts)
    B = guide.size(0)
    S = gs_batch.size(0) // B

    center = guide.unsqueeze(0).expand(
        S, B, *guide.shape[1:]).reshape_as(gs_batch)
    pair_batch = torch.cat([center, gs_batch], dim=1).contiguous()
    return pair_batch, mask


def accumulate_uz_reference(
    x: torch.Tensor,
    weights: torch.Tensor,        # (S*B, C, H, W), pre-masked by comp_box
    shifts: torch.Tensor,         # (S, 3)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for `accumulate_uz`.

    Matches the original `nekre` batched accumulation loop semantics
    (including the inverse / symmetric branch and its circular-pad mask).
    """
    shifts = _wrap_shifts(shifts)
    if shifts.size(1) != 3:
        raise ValueError("accumulate_uz requires shifts with shape (S, 3)")
    B, C, H, W = x.shape
    S = shifts.size(0)
    R = int(shifts[:, :2].abs().max().item())

    U = torch.zeros_like(x)
    Z = torch.zeros_like(x)

    padded_img = F.pad(x, (R, R, R, R), mode='circular')
    box = F.pad(torch.ones(1, 1, H, W, device=x.device, dtype=x.dtype),
                (R, R, R, R), mode='constant', value=0)

    w_view = weights.view(S, B, C, H, W)
    for i in range(S):
        dx = int(shifts[i, 0].item())
        dy = int(shifts[i, 1].item())
        has_inv = bool(int(shifts[i, 2].item()))

        weight_fwd = w_view[i]  # already mask-folded
        v = padded_img[:, :, R + dx:R + dx + H, R + dy:R + dy + W]
        U = U + weight_fwd * v
        Z = Z + weight_fwd

        if has_inv:
            dx_inv, dy_inv = -dx, -dy
            wpad = F.pad(weight_fwd, (R, R, R, R), mode='circular')
            w_inv = wpad[:, :, R + dx_inv:R +
                         dx_inv + H, R + dy_inv:R + dy_inv + W]
            cb_inv = box[:, :, R + dx_inv:R +
                         dx_inv + H, R + dy_inv:R + dy_inv + W]
            w_inv = w_inv * cb_inv
            v_inv = padded_img[:, :, R + dx_inv:R +
                               dx_inv + H, R + dy_inv:R + dy_inv + W]
            U = U + w_inv * v_inv
            Z = Z + w_inv

    return U, Z


def accumulate_uz_scalar_reference(
    x: torch.Tensor,              # (B, C, H, W)
    # (S, B, C) -- ONE scalar per (transform, image, channel)
    weights: torch.Tensor,
    shifts: torch.Tensor,         # (S, 2) translations, gather convention
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for `accumulate_uz_scalar` (GASD accumulation).

    Computes, with gather (not roll) shift convention matching `shift_gather`::

        U[b, c, h, w] = sum_s weights[s, b, c] * x[b, c, (h+dx_s) % H, (w+dy_s) % W]
        Z[b, c, h, w] = sum_s weights[s, b, c]                (constant over h, w)

    This is the scalar-per-transform analogue of `accumulate_uz`: the weight is
    constant over space, so it is never materialized as an (S*B, C, H, W)
    tensor. There is no boundary mask (circular) and no inverse-symmetry branch
    -- GASD's transforms are exact permutations and each carries its own
    independent scalar weight.
    """
    if shifts.dim() != 2 or shifts.size(1) < 2:
        raise ValueError(f"shifts must be (S, >=2); got {tuple(shifts.shape)}")
    B, C, H, W = x.shape
    S = shifts.size(0)
    if weights.shape != (S, B, C):
        raise ValueError(
            f"weights must be (S, B, C)={ (S, B, C) }; got {tuple(weights.shape)}")

    U = torch.zeros_like(x)
    Z = x.new_zeros(B, C, 1, 1)
    for s in range(S):
        dx = int(shifts[s, 0].item())
        dy = int(shifts[s, 1].item())
        # gather x at (h+dx, w+dy) circular == roll by (-dx, -dy)
        xs = x if (dx == 0 and dy == 0) else torch.roll(
            x, shifts=(-dx, -dy), dims=(-2, -1))
        w = weights[s].view(B, C, 1, 1)
        U = U + w * xs
        Z = Z + w
    return U, Z.expand(B, C, H, W)


# ---------------------------------------------------------------------------
# torch.autograd.Function wrappers around the CUDA kernels
# ---------------------------------------------------------------------------

def _prep_shifts_for_cuda(shifts: torch.Tensor, device: torch.device) -> torch.Tensor:
    if shifts.dim() != 2 or shifts.size(1) not in (2, 3):
        raise ValueError(
            f"shifts must be (S, 2) or (S, 3); got {tuple(shifts.shape)}")
    return shifts.to(device=device, dtype=torch.int32).contiguous()


class _ShiftGatherFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, guide, shifts):
        # guide: (B, C, H, W) CUDA contiguous float32/float64
        # shifts: int32/int64, will be cast to int32 on device
        shifts_i = _prep_shifts_for_cuda(shifts, guide.device)
        out, mask = _C.shift_gather_forward(guide.contiguous(), shifts_i)
        ctx.save_for_backward(shifts_i)
        ctx.guide_shape = guide.shape  # (B, C, H, W)
        return out, mask

    @staticmethod
    def backward(ctx, grad_out, grad_mask):
        (shifts_i,) = ctx.saved_tensors
        B, C, H, W = ctx.guide_shape
        # mask has no learnable gradient
        grad_guide = _C.shift_gather_backward(
            grad_out.contiguous(), shifts_i, B, C, H, W)
        return grad_guide, None


class _PairGatherFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, guide, shifts):
        shifts_i = _prep_shifts_for_cuda(shifts, guide.device)
        out, mask = _C.pair_gather_forward(guide.contiguous(), shifts_i)
        ctx.save_for_backward(shifts_i)
        ctx.guide_shape = guide.shape
        return out, mask

    @staticmethod
    def backward(ctx, grad_out, grad_mask):
        (shifts_i,) = ctx.saved_tensors
        B, C, H, W = ctx.guide_shape
        grad_guide = _C.pair_gather_backward(
            grad_out.contiguous(), shifts_i, B, C, H, W)
        return grad_guide, None


class _AccumulateUZFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, shifts):
        if shifts.dim() != 2 or shifts.size(1) != 3:
            raise ValueError("accumulate_uz requires shifts of shape (S, 3)")
        shifts_i = _prep_shifts_for_cuda(shifts, x.device)
        U, Z = _C.accumulate_uz_forward(
            x.contiguous(), weights.contiguous(), shifts_i)
        ctx.save_for_backward(x, weights, shifts_i)
        return U, Z

    @staticmethod
    def backward(ctx, grad_U, grad_Z):
        x, weights, shifts_i = ctx.saved_tensors
        grad_x, grad_w = _C.accumulate_uz_backward(
            x.contiguous(), weights.contiguous(),
            grad_U.contiguous(), grad_Z.contiguous(), shifts_i)
        return grad_x, grad_w, None


def _has_scalar_ext() -> bool:
    """The scalar-accumulate CUDA symbols are only present in 0.4.x+ builds of
    the compiled extension."""
    return _HAS_CUDA_EXT and hasattr(_C, "accumulate_uz_scalar_forward")


def _require_cuda_ext(op_name: str, need_scalar: bool = False) -> None:
    # A CUDA tensor reaching a reference impl is a performance trap, not a
    # graceful fallback: the references read (dx, dy) out of a *device* shifts
    # tensor with .item() inside per-shift loops -- hundreds of full-stream
    # synchronizations per forward (~245 for shift_gather + ~242 for
    # accumulate_uz_scalar at R=5), which destroys CPU run-ahead and makes the
    # "CUDA" path slower than the model's own torch.roll forward. Fail loudly
    # instead so a stale binary is caught at the first call, not by a silent
    # 2x training slowdown.
    if not _HAS_CUDA_EXT:
        raise RuntimeError(
            f"neural_shift_cuda.{op_name}: input is on CUDA but the compiled "
            f"extension `_C` failed to import. Rebuild it in THIS environment "
            f"(pip install --no-build-isolation --force-reinstall .) and check "
            f"for a stale in-tree _C*.so shadowing site-packages.")
    if need_scalar and not hasattr(_C, "accumulate_uz_scalar_forward"):
        raise RuntimeError(
            f"neural_shift_cuda.{op_name}: the imported `_C` binary predates "
            f"0.4.0 (no accumulate_uz_scalar symbols) -- a stale build is being "
            f"picked up. `import neural_shift_cuda._C as C; print(C.__file__)` "
            f"to locate it, then force-reinstall from the current source.")


class _AccumulateUZScalarFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, shifts):
        # x: (B, C, H, W); weights: (S, B, C); shifts: (S, 2) gather convention
        if shifts.dim() != 2 or shifts.size(1) < 2:
            raise ValueError(
                "accumulate_uz_scalar requires shifts of shape (S, >=2)")
        shifts_i = shifts[:, :2].to(
            device=x.device, dtype=torch.int32).contiguous()
        U, Z = _C.accumulate_uz_scalar_forward(
            x.contiguous(), weights.contiguous(), shifts_i)
        ctx.save_for_backward(x, weights, shifts_i)
        return U, Z

    @staticmethod
    def backward(ctx, grad_U, grad_Z):
        x, weights, shifts_i = ctx.saved_tensors
        grad_x, grad_w = _C.accumulate_uz_scalar_backward(
            x.contiguous(), weights.contiguous(),
            grad_U.contiguous(), grad_Z.contiguous(), shifts_i)
        return grad_x, grad_w, None


# ---------------------------------------------------------------------------
# Public API: dispatches CUDA vs reference based on device.
# ---------------------------------------------------------------------------

def shift_gather(
    guide: torch.Tensor,
    shifts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Circular shift the guide tensor by every (dx, dy) in `shifts`, returning a
    flattened (S*B, C, H, W) batch plus a (S*B, 1, H, W) binary validity mask.

    Output ordering:
        gs[s*B + b, c, h, w] = guide[b, c, (h+dx_s) mod H, (w+dy_s) mod W]
        mask[s*B + b, 0, h, w] = 1 iff 0 <= h+dx_s < H and 0 <= w+dy_s < W
    """
    if guide.is_cuda:
        _require_cuda_ext("shift_gather")
        return _ShiftGatherFn.apply(guide, shifts)
    return shift_gather_reference(guide, shifts)


def pair_gather(
    guide: torch.Tensor,
    shifts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Same as shift_gather but the returned tensor already has the center guide
    concatenated along the channel dim, saving a guide.repeat + torch.cat in
    `nekre.compute_weights` for `model_type == 'concat'`.

        pair[s*B + b, :C, :, :]  = guide[b]
        pair[s*B + b, C:, :, :]  = shifted_guide_s[b]
    """
    if guide.is_cuda:
        _require_cuda_ext("pair_gather")
        return _PairGatherFn.apply(guide, shifts)
    return pair_gather_reference(guide, shifts)


def accumulate_uz(
    x: torch.Tensor,
    weights: torch.Tensor,
    shifts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Accumulate U_num and Z over all shifts, including the symmetric inverse
    branch. `weights` must already be pre-multiplied by the forward `comp_box`
    mask (matching `weight_fwd = weight * comp_box` in the original code).

    Returns (U_num, Z). User divides: U = U_num / Z.
    """
    if x.is_cuda:
        _require_cuda_ext("accumulate_uz")
        return _AccumulateUZFn.apply(x, weights, shifts)
    return accumulate_uz_reference(x, weights, shifts)


def accumulate_uz_scalar(
    x: torch.Tensor,
    weights: torch.Tensor,
    shifts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scalar-per-transform accumulation used by the GASD forward::

        U[b, c, h, w] = sum_s weights[s, b, c] * x[b, c, (h+dx_s) % H, (w+dy_s) % W]
        Z[b, c, h, w] = sum_s weights[s, b, c]

    `weights` is (S, B, C) -- one positive scalar per (transform, image,
    channel), NOT the (S*B, C, H, W) per-pixel tensor `accumulate_uz` takes.
    `shifts` is (S, 2) in gather convention (same as `shift_gather`).

    Returns (U_num, Z); the caller divides U = U_num / Z. Fully differentiable
    w.r.t. both `x` and `weights`.
    """
    if x.is_cuda:
        _require_cuda_ext("accumulate_uz_scalar", need_scalar=True)
        return _AccumulateUZScalarFn.apply(x, weights, shifts)
    return accumulate_uz_scalar_reference(x, weights, shifts)


__all__ = [
    "shift_gather",
    "pair_gather",
    "accumulate_uz",
    "accumulate_uz_scalar",
    "shift_gather_reference",
    "pair_gather_reference",
    "accumulate_uz_reference",
    "accumulate_uz_scalar_reference",
]
