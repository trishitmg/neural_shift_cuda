"""Integration shims that wire neural_shift_cuda into model classes.

These do not change any model math or parameters; they monkey-patch the
`forward` method to route through the CUDA kernels.

v2 / v3 / v4 (per-shift-independent weights, accumulate_uz applies):

    from neural_shift_cuda.integration import install_cuda_shift_attn
    from NKD_drunet_attn_v4 import NeKDeDRUNetAttn
    install_cuda_shift_attn(NeKDeDRUNetAttn)

v5 (post-mix softmax over the shift axis -- accumulate_uz does NOT apply;
only the neighbour gather is accelerated, see nekde_drunet_attn_v5_patch):

    from neural_shift_cuda.integration import install_cuda_shift_attn_v5
    from NKD_drunet_attn_v5 import NeKDeDRUNetAttn
    install_cuda_shift_attn_v5(NeKDeDRUNetAttn)

Classic nekre model:

    from neural_shift_cuda.integration import install_cuda_shift_nekre
    from NKD_lw_conv import nekre
    install_cuda_shift_nekre(nekre)
"""

from .nekde_drunet_attn_patch import install_cuda_shift as install_cuda_shift_attn
from .nekde_drunet_attn_v5_patch import install_cuda_shift_attn_v5
from .nekre_patch import install_cuda_shift as install_cuda_shift_nekre

__all__ = [
    "install_cuda_shift_attn",
    "install_cuda_shift_attn_v5",
    "install_cuda_shift_nekre",
]
