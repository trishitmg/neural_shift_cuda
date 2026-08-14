# neural_shift_cuda/ops.py
#
# High-level Python API.
#
#   shift_gather(guide, shifts)         -> (gs_batch, mask_batch)
#   pair_gather(guide, shifts)          -> (pair_batch, mask_batch)
#   accumulate_uz(x, weights, shifts)              -> (U_num, Z)
#   normalized_accumulate_uz(x, weights, shifts)   -> D or (D, log_C)
#
# Each op:
#   * dispatches to the CUDA kernels (extension `_C`) when guide/x are on CUDA;
#   * falls back to a pure-PyTorch reference impl otherwise (and exposes the
#     reference impl as `*_reference` for tests).
#
# All ops are differentiable via torch.autograd.Function. The reference
# impls are also differentiable because they are pure PyTorch.

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from neural_shift_cuda import _C
    _HAS_CUDA_EXT = True
    _EXT_IMPORT_ERROR = None
except Exception as _e:  # pragma: no cover -- CPU-only env / build failure
    _C = None
    _HAS_CUDA_EXT = False
    # Keep the real exception (missing .so, torch-ABI undefined-symbol, missing
    # libcudart, ...) so _require_cuda_ext can surface the actual cause instead
    # of a generic "failed to import". A bare except that drops this is what
    # made the fallback silent in the first place.
    _EXT_IMPORT_ERROR = _e


# ---------------------------------------------------------------------------
# Reference (PyTorch) implementations -- used as CPU fallback and as the
# numerical ground truth in the test suite.
# ---------------------------------------------------------------------------

def _wrap_shifts(shifts: torch.Tensor) -> torch.Tensor:
    if shifts.dim() != 2 or shifts.size(1) not in (2, 3):
        raise ValueError(
            f"shifts must be (S, 2) or (S, 3); got {tuple(shifts.shape)}")
    return shifts.to(torch.int64)


def _accumulator_dtype(*tensors: torch.Tensor) -> torch.dtype:
    """Choose a safe accumulation dtype before any products or reductions.

    float16/bfloat16/float32 inputs accumulate in float32; a float64 input
    keeps the whole reduction in float64.  The returned U/Z/D tensors use this
    dtype as well, so a correct float32 sum is never narrowed back to fp16 and
    overflowed at the API boundary.
    """
    for tensor in tensors:
        if not tensor.is_floating_point():
            raise TypeError(
                f"accumulation inputs must be floating point; got {tensor.dtype}")
    return (torch.float64 if any(t.dtype == torch.float64 for t in tensors)
            else torch.float32)


def _promote_accumulation_inputs(
    x: torch.Tensor, weights: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if x.device != weights.device:
        raise ValueError(
            f"x and weights must share a device; got {x.device} and {weights.device}")
    dtype = _accumulator_dtype(x, weights)
    return x.to(dtype=dtype), weights.to(dtype=dtype)


def _tree_sum(values: torch.Tensor) -> torch.Tensor:
    """Pairwise binary-tree reduction over dimension 0.

    This is the PyTorch reference analogue of the CUDA shared-memory tree.
    Each level is a batched tensor add and the dependency depth is O(log S).
    """
    if values.dim() == 0 or values.size(0) == 0:
        raise ValueError("tree reduction requires a non-empty leading dimension")
    level = values
    while level.size(0) > 1:
        paired = level.size(0) // 2
        reduced = level[:2 * paired:2] + level[1:2 * paired:2]
        if level.size(0) & 1:
            reduced = torch.cat((reduced, level[-1:]), dim=0)
        level = reduced
    return level[0]


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
    x, weights = _promote_accumulation_inputs(x, weights)
    shifts = _wrap_shifts(shifts)
    if shifts.size(1) != 3:
        raise ValueError("accumulate_uz requires shifts with shape (S, 3)")
    B, C, H, W = x.shape
    S = shifts.size(0)
    R = int(shifts[:, :2].abs().max().item())

    padded_img = F.pad(x, (R, R, R, R), mode='circular')
    box = F.pad(torch.ones(1, 1, H, W, device=x.device, dtype=x.dtype),
                (R, R, R, R), mode='constant', value=0)

    w_view = weights.reshape(S, B, C, H, W)
    u_terms = []
    z_terms = []
    for i in range(S):
        dx = int(shifts[i, 0].item())
        dy = int(shifts[i, 1].item())
        has_inv = bool(int(shifts[i, 2].item()))

        weight_fwd = w_view[i]  # already mask-folded
        v = padded_img[:, :, R + dx:R + dx + H, R + dy:R + dy + W]
        u_terms.append(weight_fwd * v)
        z_terms.append(weight_fwd)

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
            u_terms.append(w_inv * v_inv)
            z_terms.append(w_inv)

    return _tree_sum(torch.stack(u_terms, dim=0)), _tree_sum(
        torch.stack(z_terms, dim=0))


def _validate_stable_weights(
    weights: torch.Tensor, log_weights: bool, *, asynchronous: bool,
) -> None:
    if log_weights:
        valid = torch.isfinite(weights) | (
            torch.isinf(weights) & (weights < 0))
        message = (
            "log_weights may be finite or -inf (mask), but not NaN or +inf")
    else:
        valid = torch.isfinite(weights) & (weights >= 0)
        message = "positive weights must be finite and nonnegative"
    predicate = valid.all()
    if asynchronous and weights.is_cuda and hasattr(torch, "_assert_async"):
        torch._assert_async(predicate, message)
    elif not bool(predicate.item()):
        raise ValueError(message)


def _as_log_weights(weights: torch.Tensor, log_weights: bool) -> torch.Tensor:
    if log_weights:
        return weights
    neg_inf_value = torch.full_like(weights, -torch.inf)
    return torch.where(weights > 0, weights.log(), neg_inf_value)


def normalized_accumulate_uz_reference(
    x: torch.Tensor,
    weights: torch.Tensor,
    shifts: torch.Tensor,
    *,
    log_weights: bool = False,
    return_log_degree: bool = False,
    validate: bool = True,
):
    """Overflow-safe PyTorch reference for the normalized denoiser.

    The returned image is computed from row-wise max-shifted log-weights, so
    neither the numerator nor degree is ever formed at its original scale.
    With ``log_weights=True``, ``weights`` contains pre-activation log-weights;
    use ``-inf`` for masked entries.  This is the only mode that can also
    prevent overflow *inside* an exponential positive activation.
    """
    x, weights = _promote_accumulation_inputs(x, weights)
    shifts = _wrap_shifts(shifts)
    if shifts.size(1) != 3:
        raise ValueError(
            "normalized_accumulate_uz requires shifts with shape (S, 3)")
    if validate:
        _validate_stable_weights(weights, log_weights, asynchronous=False)

    B, C, H, W = x.shape
    S = shifts.size(0)
    if S == 0:
        raise ValueError("at least one shift is required")
    if weights.shape != (S * B, C, H, W):
        raise ValueError(
            f"weights must be {(S * B, C, H, W)}; got {tuple(weights.shape)}")
    R = int(shifts[:, :2].abs().max().item())
    padded_img = F.pad(x, (R, R, R, R), mode="circular")
    box = F.pad(
        torch.ones(1, 1, H, W, device=x.device, dtype=x.dtype),
        (R, R, R, R), mode="constant", value=0)

    w_view = weights.reshape(S, B, C, H, W)
    log_terms = []
    value_terms = []
    for i in range(S):
        dx = int(shifts[i, 0].item())
        dy = int(shifts[i, 1].item())
        has_inv = bool(int(shifts[i, 2].item()))

        weight_fwd = w_view[i]
        value_fwd = padded_img[
            :, :, R + dx:R + dx + H, R + dy:R + dy + W]
        log_terms.append(_as_log_weights(weight_fwd, log_weights))
        value_terms.append(value_fwd)

        if has_inv:
            wpad = F.pad(weight_fwd, (R, R, R, R), mode="circular")
            weight_inv = wpad[
                :, :, R - dx:R - dx + H, R - dy:R - dy + W]
            cb_inv = box[
                :, :, R - dx:R - dx + H, R - dy:R - dy + W]
            if log_weights:
                weight_inv = torch.where(
                    cb_inv > 0, weight_inv,
                    torch.full_like(weight_inv, -torch.inf))
            else:
                weight_inv = weight_inv * cb_inv
            value_inv = padded_img[
                :, :, R - dx:R - dx + H, R - dy:R - dy + W]
            log_terms.append(_as_log_weights(weight_inv, log_weights))
            value_terms.append(value_inv)

    log_stack = torch.stack(log_terms, dim=0)
    value_stack = torch.stack(value_terms, dim=0)
    row_max = torch.amax(log_stack, dim=0)
    empty = torch.isinf(row_max) & (row_max < 0)
    centered = torch.where(
        empty.unsqueeze(0), torch.full_like(log_stack, -torch.inf),
        log_stack - row_max.unsqueeze(0))
    scaled = centered.exp()
    numerator = _tree_sum(scaled * value_stack)
    degree_scaled = _tree_sum(scaled)
    denoised = torch.where(
        empty, torch.zeros_like(numerator), numerator / degree_scaled)
    log_degree = torch.where(
        empty, torch.full_like(row_max, -torch.inf),
        row_max + degree_scaled.log())
    if return_log_degree:
        return denoised, log_degree
    return denoised


def _broadcast_scalar_weights(
    weights: torch.Tensor, S: int, B: int, C: int,
) -> torch.Tensor:
    """Normalize accumulate_uz_scalar weights to (S, B, C).

    Accepted shapes:
        (S, B, C) -- ONE scalar per (transform, image, channel): the GASD
                     layout, C*S weights per image (the head pools over (H, W)
                     only, keeping the channel axis).
        (S, B)    -- ONE scalar per transform shared across channels; expanded
                     to (S, B, C). Autograd through the expand sums grad over C,
                     the exact gradient of a channel-shared scalar. Kept as a
                     convenience so a channel-tied caller need not tile by hand.
    """
    if weights.dim() == 3:
        if weights.shape != (S, B, C):
            raise ValueError(
                f"weights must be (S, B, C)={(S, B, C)} or (S, B)={(S, B)}; "
                f"got {tuple(weights.shape)}")
        return weights
    if weights.shape != (S, B):
        raise ValueError(
            f"weights must be (S, B, C)={(S, B, C)} or (S, B)={(S, B)}; "
            f"got {tuple(weights.shape)}")
    return weights.unsqueeze(-1).expand(S, B, C)


def accumulate_uz_scalar_reference(
    x: torch.Tensor,              # (B, C, H, W)
    # (S, B, C) -- ONE scalar per (transform, image, channel); (S, B) accepted
    weights: torch.Tensor,
    shifts: torch.Tensor,         # (S, 2) translations, gather convention
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for `accumulate_uz_scalar` (GASD accumulation).

    Computes, with gather (not roll) shift convention matching `shift_gather`::

        U[b, c, h, w] = sum_s weights[s, b, c] * x[b, c, (h+dx_s) % H, (w+dy_s) % W]
        Z[b, c, h, w] = sum_s weights[s, b, c]                (constant over h, w)

    `weights` is (S, B, C) -- one scalar per (transform, image, channel), the
    GASD layout of C*S weights per image -- or (S, B) shared across channels.
    The weight is constant over space either way, so it is never materialized
    as an (S*B, C, H, W) tensor. There is no boundary mask (circular) and no
    inverse-symmetry branch -- GASD's transforms are exact permutations and each
    carries its own independent scalar weight.
    """
    x, weights = _promote_accumulation_inputs(x, weights)
    if shifts.dim() != 2 or shifts.size(1) < 2:
        raise ValueError(f"shifts must be (S, >=2); got {tuple(shifts.shape)}")
    B, C, H, W = x.shape
    S = shifts.size(0)
    weights = _broadcast_scalar_weights(weights, S, B, C)

    u_terms = []
    z_terms = []
    for s in range(S):
        dx = int(shifts[s, 0].item())
        dy = int(shifts[s, 1].item())
        # gather x at (h+dx, w+dy) circular == roll by (-dx, -dy)
        xs = x if (dx == 0 and dy == 0) else torch.roll(
            x, shifts=(-dx, -dy), dims=(-2, -1))
        w = weights[s].reshape(B, C, 1, 1)
        u_terms.append(w * xs)
        z_terms.append(w)
    U = _tree_sum(torch.stack(u_terms, dim=0))
    Z = _tree_sum(torch.stack(z_terms, dim=0))
    return U, Z.expand(B, C, H, W)


# ---------------------------------------------------------------------------
# torch.autograd.Function wrappers around the CUDA kernels
# ---------------------------------------------------------------------------

def _prep_shifts_for_cuda(shifts: torch.Tensor, device: torch.device) -> torch.Tensor:
    if shifts.dim() != 2 or shifts.size(1) not in (2, 3):
        raise ValueError(
            f"shifts must be (S, 2) or (S, 3); got {tuple(shifts.shape)}")
    # Shift values and CUDA pointer arithmetic use int64 end-to-end.  Casting
    # to int32 here used to silently wrap large offsets before the kernel saw
    # them, independently of the flattened-index overflow fixed in CUDA.
    return shifts.to(device=device, dtype=torch.int64).contiguous()


class _ShiftGatherFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, guide, shifts):
        # guide: (B, C, H, W) CUDA contiguous float32/float64
        # shifts: integer tensor, normalized to int64 on device
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


class _NormalizedAccumulateUZFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, shifts, log_weights):
        if shifts.dim() != 2 or shifts.size(1) != 3:
            raise ValueError(
                "normalized_accumulate_uz requires shifts of shape (S, 3)")
        shifts_i = _prep_shifts_for_cuda(shifts, x.device)
        D, log_C = _C.normalized_accumulate_uz_forward(
            x.contiguous(), weights.contiguous(), shifts_i, bool(log_weights))
        ctx.save_for_backward(x, weights, D, log_C, shifts_i)
        ctx.log_weights = bool(log_weights)
        ctx.set_materialize_grads(False)
        return D, log_C

    @staticmethod
    def backward(ctx, grad_D, grad_log_C):
        x, weights, D, log_C, shifts_i = ctx.saved_tensors
        if grad_D is None:
            grad_D = torch.zeros_like(D)
        if grad_log_C is None:
            grad_log_C = torch.zeros_like(log_C)
        grad_x, grad_w = _C.normalized_accumulate_uz_backward(
            x.contiguous(), weights.contiguous(), D.contiguous(),
            log_C.contiguous(), grad_D.contiguous(),
            grad_log_C.contiguous(), shifts_i, ctx.log_weights)
        return grad_x, grad_w, None, None


def _has_scalar_ext() -> bool:
    """The scalar-accumulate CUDA symbols are only present in 0.4.x+ builds of
    the compiled extension."""
    return _HAS_CUDA_EXT and hasattr(_C, "accumulate_uz_scalar_forward")


def _has_int64_index_ext() -> bool:
    """True only for the 0.11.1+ CUDA ABI with 64-bit shifts and indices."""
    if not _HAS_CUDA_EXT or not hasattr(_C, "index_width_bits"):
        return False
    try:
        return int(_C.index_width_bits()) == 64
    except Exception:
        return False


def _require_cuda_ext(
    op_name: str, need_scalar: bool = False, need_normalized: bool = False,
) -> None:
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
            f"for a stale in-tree _C*.so shadowing site-packages.\n"
            f"  Underlying import error: "
            f"{type(_EXT_IMPORT_ERROR).__name__}: {_EXT_IMPORT_ERROR}"
        ) from _EXT_IMPORT_ERROR
    if not _has_int64_index_ext():
        raise RuntimeError(
            f"neural_shift_cuda.{op_name}: the imported `_C` binary predates "
            f"0.11.1 and still uses unsafe 32-bit CUDA shifts/flattened indices. "
            f"Force-reinstall the extension from the current source before "
            f"passing CUDA tensors.")
    if need_scalar and not hasattr(_C, "accumulate_uz_scalar_forward"):
        raise RuntimeError(
            f"neural_shift_cuda.{op_name}: the imported `_C` binary predates "
            f"0.4.0 (no accumulate_uz_scalar symbols) -- a stale build is being "
            f"picked up. `import neural_shift_cuda._C as C; print(C.__file__)` "
            f"to locate it, then force-reinstall from the current source.")
    if need_normalized and not hasattr(_C, "normalized_accumulate_uz_forward"):
        raise RuntimeError(
            f"neural_shift_cuda.{op_name}: the imported `_C` binary predates "
            f"0.11.0 (no overflow-safe normalized reduction symbols). Force-"
            f"reinstall the extension from the current source.")


class _AccumulateUZScalarFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weights, shifts):
        # x: (B, C, H, W); weights: (S, B, C); shifts: (S, 2) gather convention
        if shifts.dim() != 2 or shifts.size(1) < 2:
            raise ValueError(
                "accumulate_uz_scalar requires shifts of shape (S, >=2)")
        shifts_i = shifts[:, :2].to(
            device=x.device, dtype=torch.int64).contiguous()
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

    Returns (U_num, Z). User divides: U = U_num / Z. Inputs are promoted
    before multiplication/reduction: fp16/bfloat16/fp32 -> fp32 and fp64 ->
    fp64. For very large positive weights, prefer ``normalized_accumulate_uz``
    because even a float32 *unscaled* K or C can be unrepresentable.
    """
    x_acc, weights_acc = _promote_accumulation_inputs(x, weights)
    if x_acc.is_cuda:
        _require_cuda_ext("accumulate_uz")
        return _AccumulateUZFn.apply(x_acc, weights_acc, shifts)
    return accumulate_uz_reference(x_acc, weights_acc, shifts)


def normalized_accumulate_uz(
    x: torch.Tensor,
    weights: torch.Tensor,
    shifts: torch.Tensor,
    *,
    log_weights: bool = False,
    return_log_degree: bool = False,
    validate: bool = True,
):
    """Compute the normalized denoiser ``D=K/C`` without overflow.

    The CUDA implementation uses two cooperative O(log S) tree reductions per
    output element: a row maximum followed by a sum of max-shifted terms.  It
    returns ``log(C)`` on request because C itself may be unrepresentable.

    Args:
        x: Image tensor ``(B,C,H,W)``.
        weights: ``(S*B,C,H,W)`` positive weights when ``log_weights=False``;
            zero is a mask.  With ``log_weights=True``, pass log-weights/logits
            directly and use ``-inf`` for masked entries.
        shifts: Integer ``(S,3)`` tensor ``(dx,dy,has_inverse)``.
        log_weights: Use the online-softmax/LSE form.  This is recommended for
            exponential heads because it prevents overflow before activation.
        return_log_degree: Return ``(D, log_C)`` instead of only ``D``.
        validate: Check the positive/log-weight contract.  On CUDA this uses
            an asynchronous assertion when the installed PyTorch supports it.

    All arithmetic is promoted before the reduction: float16/bfloat16/float32
    inputs use float32 accumulators and float64 inputs use float64.
    """
    x_acc, weights_acc = _promote_accumulation_inputs(x, weights)
    if validate:
        _validate_stable_weights(
            weights_acc, log_weights, asynchronous=x_acc.is_cuda)
    if x_acc.is_cuda:
        _require_cuda_ext(
            "normalized_accumulate_uz", need_normalized=True)
        D, log_C = _NormalizedAccumulateUZFn.apply(
            x_acc, weights_acc, shifts, log_weights)
    else:
        D, log_C = normalized_accumulate_uz_reference(
            x_acc, weights_acc, shifts, log_weights=log_weights,
            return_log_degree=True, validate=False)
    if return_log_degree:
        return D, log_C
    return D


def accumulate_uz_scalar(
    x: torch.Tensor,
    weights: torch.Tensor,
    shifts: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Scalar-per-transform accumulation used by the GASD forward::

        U[b, c, h, w] = sum_s w[s, b, c] * x[b, c, (h+dx_s) % H, (w+dy_s) % W]
        Z[b, c, h, w] = sum_s w[s, b, c]

    `weights` is (S, B, C) -- ONE positive scalar per (transform, image,
    channel), the GASD layout of C*S weights per image -- or (S, B) shared
    across channels. Never the (S*B, C, H, W) per-pixel tensor `accumulate_uz`
    takes. `shifts` is (S, 2) in gather convention (same as `shift_gather`).

    Returns (U_num, Z). When the weights are normalized over the transform
    axis (sum_s w_s = 1 per channel, as GASD's `_finalize_weights` guarantees),
    Z is identically one and the caller can ignore it. Fully differentiable
    w.r.t. both `x` and `weights`.

    A (S, B) input is expanded to (S, B, C) IN PYTHON before the autograd
    Function rather than by a dedicated kernel path: the weight tensor is tiny
    (S*B*C floats), the expand's backward is the exact channel-sum gradient of
    a shared scalar, and it reuses the bit-exact validated (S, B, C) kernels
    without a rebuild.
    """
    x_acc, weights_acc = _promote_accumulation_inputs(x, weights)
    if x_acc.is_cuda:
        _require_cuda_ext("accumulate_uz_scalar", need_scalar=True)
        if shifts.dim() != 2 or shifts.size(1) < 2:
            raise ValueError(
                "accumulate_uz_scalar requires shifts of shape (S, >=2)")
        B, C = x_acc.shape[0], x_acc.shape[1]
        weights_acc = _broadcast_scalar_weights(
            weights_acc, shifts.size(0), B, C)
        return _AccumulateUZScalarFn.apply(x_acc, weights_acc, shifts)
    return accumulate_uz_scalar_reference(x_acc, weights_acc, shifts)


__all__ = [
    "shift_gather",
    "pair_gather",
    "accumulate_uz",
    "normalized_accumulate_uz",
    "accumulate_uz_scalar",
    "shift_gather_reference",
    "pair_gather_reference",
    "accumulate_uz_reference",
    "normalized_accumulate_uz_reference",
    "accumulate_uz_scalar_reference",
]
