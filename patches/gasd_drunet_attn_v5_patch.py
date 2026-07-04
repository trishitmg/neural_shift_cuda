"""
Integration patch for GASDDRUNetAttn v5 (GASD_drunet_attn_v5) to use
neural_shift_cuda kernels.

v2 and v5 are now the SAME operator
-----------------------------------
The GASD refactor made both v2 and v5 fully circular, scalar-per-transform
group-averaged denoisers::

    W x = sum_{g in G} alpha_g P_g x ,   alpha_g > 0, sum_g alpha_g = 1

There is no ``comp_box`` boundary mask in either version any more (v2's mask was
removed in the refactor), so the two forwards are byte-for-byte identical in
structure -- only their default constructor arguments differ. The historical
v5-specific patch (which dropped the v2 ``comp_box`` masking step) is therefore
redundant: the v2 GASD patch already produces the periodic/circular operator v5
wants.

Consequently this module just re-exports the v2 patch. ``install_cuda_shift``
patches whatever class you hand it (the class is named ``GASDDRUNetAttn`` in
both files, with the ``NeKDeDRUNetAttn`` alias kept for back-compat), so a
single implementation covers both.

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
    _head_mix_mat,
    _normalize_transform_weights,
    _gasd_transform_partition,
    _gasd_trans_shifts,
)

__all__ = ["install_cuda_shift"]
