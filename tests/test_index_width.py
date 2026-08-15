import importlib

import pytest
import torch

from neural_shift_cuda.ops import _prep_shifts_for_cuda


def test_shift_preparation_promotes_int32_to_int64():
    shifts = torch.tensor([[1, -2, 1]], dtype=torch.int32)
    prepared = _prep_shifts_for_cuda(shifts, torch.device("cpu"))
    assert prepared.dtype == torch.int64
    assert prepared.is_contiguous()
    assert prepared.tolist() == [[1, -2, 1]]


def test_shift_preparation_preserves_values_outside_int32_range():
    large = 2**31 + 17
    shifts = torch.tensor([[large, -large, 0]], dtype=torch.int64)
    prepared = _prep_shifts_for_cuda(shifts, torch.device("cpu"))
    assert prepared.dtype == torch.int64
    assert prepared.tolist() == [[large, -large, 0]]


def test_compiled_extension_advertises_64_bit_index_abi():
    try:
        ext = importlib.import_module("neural_shift_cuda._C")
    except (ImportError, OSError) as exc:
        pytest.skip(f"CUDA extension is not built in this environment: {exc}")
    assert hasattr(ext, "index_width_bits")
    assert ext.index_width_bits() == 64
