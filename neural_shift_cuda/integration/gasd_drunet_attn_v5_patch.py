"""
Integration patch for GASDDRUNetAttn v5 (GASD_drunet_attn_v5) to use
neural_shift_cuda kernels.

v2 and v5 are the SAME operator
-------------------------------
Both are fully-circular, one-weight-per-transform group-averaged denoisers::

    W x = sum_{g in G} alpha_g P_g x ,   alpha_g > 0, sum_g alpha_g = 1

with the weight pipeline head -> _mix_heads (per-pixel) -> _pool_to_scalar
over (C, H, W) -> _finalize_weights over the transform axis. There is no
boundary/comp_box mask and no degree-map division in either version; the two
forwards are byte-for-byte identical in structure -- only their default
constructor arguments differ (softmax vs control_sigmoid, ReLU vs Softplus,
head-mix defaults). A single implementation therefore covers both.

Consequently this module just re-exports the v2 patch. ``install_cuda_shift``
patches whatever class you hand it (the class is named ``GASDDRUNetAttn`` in
both files).

Usage
-----
    from GASD_drunet_attn_v5 import GASDDRUNetAttn
    from gasd_drunet_attn_v5_patch import install_cuda_shift
    install_cuda_shift(GASDDRUNetAttn)
"""

from __future__ import annotations

from .gasd_drunet_attn_patch import (  # noqa: F401
    install_cuda_shift,
    _forward_cuda,
    _run_head,
    _gasd_transform_partition,
    _gasd_trans_shifts,
)

__all__ = ["install_cuda_shift"]
