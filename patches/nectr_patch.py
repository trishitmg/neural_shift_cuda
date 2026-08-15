"""Batched CUDA integration for the official NECTR models and legacy nekre.

Supported forward contracts:

* ``NECTR_denoiser.forward(x, reference=None, sig=None, cache=None,
  store_cache=False)`` from official ``NECTR_models1.py`` and
  ``NECTR_models2.py``.
* ``nekre.forward(x, guide=None, sig=None, batched=...)`` from the earlier
  batched implementation.

Only shift construction, batching, and aggregation are replaced. The two
official neural heads are evaluated with the same layers and activations.
The exponential head is kept in log space until the online-softmax tree so
the positive activation cannot overflow before normalization.
"""

from __future__ import annotations

import functools
import inspect
from typing import Iterable, List, Tuple

import torch

from neural_shift_cuda import (
    accumulate_uz,
    normalized_accumulate_uz,
    pair_gather,
    shift_gather,
)


Shift = Tuple[int, int, bool]
_MISSING = object()


class _NECTRWeightCache(dict):
    """Opaque official-forward cache with its weight representation tagged."""

    def __init__(self, *args, weight_kind: str = "positive", **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_kind = weight_kind


def _lazy_init(self) -> None:
    if not hasattr(self, "use_cuda_shift"):
        self.use_cuda_shift = True
    if not hasattr(self, "use_cuda_pair_gather"):
        self.use_cuda_pair_gather = True
    if not hasattr(self, "_nectr_cached_shift_tensor"):
        self._nectr_cached_shift_tensor = None
        self._nectr_cached_shift_key = None


def _collect_shifts(self) -> List[Shift]:
    """Return the exact half-plane enumeration used by both official files."""
    legacy_collector = getattr(self, "_collect_shifts", None)
    if callable(legacy_collector):
        return [(int(dx), int(dy), bool(inv))
                for dx, dy, inv in legacy_collector()]

    radius = int(self.window_rad)
    shifts: List[Shift] = []
    for dx in range(-radius, radius + 1):
        for dy in range(1, radius + 1):
            shifts.append((dx, dy, True))
    for dx in range(0, radius + 1):
        shifts.append((dx, 0, dx != 0))
    return shifts


def _get_shift_tensor(self, device: torch.device) -> Tuple[List[Shift], torch.Tensor]:
    shifts = _collect_shifts(self)
    key = (device.type, device.index, int(self.window_rad), tuple(shifts))
    if (self._nectr_cached_shift_tensor is None
            or self._nectr_cached_shift_key != key):
        # int64 is deliberate: CUDA offsets and flattened indices are 64-bit
        # end-to-end, avoiding the old int32 overflow boundary.
        rows = [(dx, dy, int(inv)) for dx, dy, inv in shifts]
        self._nectr_cached_shift_tensor = torch.tensor(
            rows, dtype=torch.int64, device=device)
        self._nectr_cached_shift_key = key
    return shifts, self._nectr_cached_shift_tensor


def _finish_projection_head(self, value: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """Apply the official model-1/legacy activation without an early exp."""
    activation = self.output_activation
    if activation == "exp":
        return value / self.sigma ** 2, "log"
    if activation == "control_sigmoid":
        k, a, b = torch.chunk(value, dim=1, chunks=3)
        return self.controlled_sigmoid(k, a, b) + 1e-8, "positive"
    return value, "raw"


def _weights_from_pair(self, pair_batch: torch.Tensor) -> Tuple[torch.Tensor, str]:
    """Run the unchanged neural head after pair_gather's concatenation."""
    if (not (hasattr(self, "unet") and hasattr(self, "layer_norm"))
            and self.output_activation == "sig"):
        # The released model-1/legacy branch references an undefined ``k``.
        # Delegate to it instead of silently changing that model behavior.
        channels = pair_batch.shape[1] // 2
        method = getattr(self, "N_theta", None)
        if method is None:
            method = self.compute_weights
        return method(pair_batch[:, :channels], pair_batch[:, channels:]), "raw"
    if hasattr(self, "unet") and hasattr(self, "layer_norm"):
        # Official NECTR model 2.
        value = self.unet(pair_batch)
        value = value.permute(0, 2, 3, 1)
        value = self.layer_norm(value)
        value = value.permute(0, 3, 1, 2)
        value = self.out(value)
        if self.output_activation == "control_sigmoid":
            k, a, b = torch.chunk(value, dim=1, chunks=3)
            value = self.controlled_sigmoid(k, a, b) + 1e-8
            return value, "positive"
        # Model 2 has no exp/sig branch in its released N_theta.
        return value, "raw"

    # Official NECTR model 1 and the legacy lightweight nekre have the same
    # projection/head contract after concatenation.
    value = pair_batch
    for projection in self.proj:
        for index, layer in enumerate(projection):
            if index == 1:
                value = value.permute(0, 2, 3, 1)
                value = layer(value)
                value = value.permute(0, 3, 1, 2)
            else:
                value = layer(value)
    return _finish_projection_head(self, self.out(value))


def _gather_and_weight(self, guide: torch.Tensor, shifts: torch.Tensor):
    if getattr(self, "use_cuda_pair_gather", True):
        pair_batch, mask_batch = pair_gather(guide.contiguous(), shifts)
    else:
        shifted, mask_batch = shift_gather(guide.contiguous(), shifts)
        shift_count = shifts.size(0)
        batch, channels, height, width = guide.shape
        center = guide.unsqueeze(0).expand(
            shift_count, batch, channels, height, width)
        center = center.reshape(
            shift_count * batch, channels, height, width)
        pair_batch = torch.cat((center, shifted), dim=1)

    values, weight_kind = _weights_from_pair(self, pair_batch)
    if weight_kind == "log":
        values = torch.where(
            mask_batch > 0, values, torch.full_like(values, -torch.inf))
    else:
        values = values * mask_batch
    return values.contiguous(), weight_kind


def _kind_for_plain_cache(self) -> str:
    if hasattr(self, "unet") and hasattr(self, "layer_norm"):
        return "positive" if self.output_activation == "control_sigmoid" else "raw"
    if self.output_activation == "exp":
        return "exp-positive"
    if self.output_activation == "control_sigmoid":
        return "positive"
    return "raw"


def _weights_from_cache(self, cache, shifts: Iterable[Shift]):
    values = torch.cat([cache[(dx, dy)] for dx, dy, _ in shifts], dim=0)
    if isinstance(cache, _NECTRWeightCache):
        return values.contiguous(), cache.weight_kind

    kind = _kind_for_plain_cache(self)
    if kind == "exp-positive":
        # A cache made by the unpatched official exp path contains already
        # exponentiated positive weights. Convert once to the log contract.
        values = torch.where(
            values > 0, values.log(), torch.full_like(values, -torch.inf))
        kind = "log"
    return values.contiguous(), kind


def _store_cache(values: torch.Tensor, shifts: Iterable[Shift], batch: int,
                 weight_kind: str) -> _NECTRWeightCache:
    cache = _NECTRWeightCache(weight_kind=weight_kind)
    for (dx, dy, _), value in zip(shifts, values.split(batch, dim=0)):
        cache[(dx, dy)] = value
    return cache


def _aggregate(self, x: torch.Tensor, values: torch.Tensor,
               shifts: torch.Tensor, weight_kind: str):
    if weight_kind in ("positive", "log"):
        denoised, log_degree = normalized_accumulate_uz(
            x.contiguous(), values, shifts,
            log_weights=(weight_kind == "log"),
            return_log_degree=True,
            validate=True,
        )
        # The released official API returns the ordinary degree, not log(C).
        # Materialize it in float64 so a value representable in double is not
        # narrowed back to float32 and overflowed after the stable reduction.
        degree = log_degree.to(torch.float64).exp()
        return denoised, degree

    # Preserve uncommon non-positive/unspecified official activations. This
    # still uses promoted fp32/fp64 binary-tree accumulation, but LSE scaling
    # is mathematically invalid when weights may be negative.
    numerator, degree = accumulate_uz(
        x.contiguous(), values, shifts)
    return numerator / degree, degree


def _forward_official(self, x, reference=None, sig=None, cache=None,
                      store_cache=False):
    if reference is None:
        guide = self.pre_activation(x, sigma=sig)
    else:
        guide = self.pre_activation(reference, sigma=sig)

    shift_rows, shift_tensor = _get_shift_tensor(self, x.device)
    if cache is None:
        values, weight_kind = _gather_and_weight(self, guide, shift_tensor)
        hold_cache = (_store_cache(values, shift_rows, x.shape[0], weight_kind)
                      if store_cache else None)
    else:
        values, weight_kind = _weights_from_cache(self, cache, shift_rows)
        hold_cache = {} if store_cache else None

    denoised, degree = _aggregate(
        self, x, values, shift_tensor, weight_kind)
    if store_cache:
        return denoised, degree, hold_cache
    return denoised, degree


def _forward_legacy(self, x, guide=None, sig=None):
    if guide is None:
        guide_features = self.pre_activation(x, sigma=sig)
    else:
        guide_features = self.pre_activation(guide, sigma=sig)
    _, shift_tensor = _get_shift_tensor(self, x.device)
    values, weight_kind = _gather_and_weight(
        self, guide_features, shift_tensor)
    return _aggregate(self, x, values, shift_tensor, weight_kind)


def install_cuda_shift(nectr_cls):
    """Patch an official ``NECTR_denoiser`` or legacy batched ``nekre``.

    The installer is idempotent. Set ``instance.use_cuda_shift = False`` to
    route back through the exact released Python forward for comparisons.
    """
    if hasattr(nectr_cls, "_neural_shift_nectr_original_forward"):
        return nectr_cls

    original_forward = nectr_cls.forward
    nectr_cls._neural_shift_nectr_original_forward = original_forward
    parameters = inspect.signature(original_forward).parameters
    is_legacy = "batched" in parameters

    if is_legacy:
        default = parameters["batched"].default
        if default is inspect.Parameter.empty:
            default = False

        @functools.wraps(original_forward)
        def patched_forward(self, x, guide=None, sig=None, batched=_MISSING):
            _lazy_init(self)
            effective_batched = default if batched is _MISSING else batched
            if (bool(effective_batched)
                    and getattr(self, "use_cuda_shift", False)
                    and getattr(self, "model_type", "concat") == "concat"):
                return _forward_legacy(self, x, guide=guide, sig=sig)
            return original_forward(
                self, x, guide=guide, sig=sig,
                batched=effective_batched)
    else:
        required = {"reference", "cache", "store_cache"}
        if not required.issubset(parameters):
            raise TypeError(
                "NECTR patch expected either the legacy batched forward or "
                "the official reference/cache/store_cache forward contract")

        @functools.wraps(original_forward)
        def patched_forward(self, x, reference=None, sig=None, cache=None,
                            store_cache=False):
            _lazy_init(self)
            if (getattr(self, "use_cuda_shift", False)
                    and getattr(self, "model_type", "concat") == "concat"):
                return _forward_official(
                    self, x, reference=reference, sig=sig,
                    cache=cache, store_cache=store_cache)
            return original_forward(
                self, x, reference=reference, sig=sig,
                cache=cache, store_cache=store_cache)

    nectr_cls.forward = patched_forward
    return nectr_cls
