"""Integration shims that wire neural_shift_cuda into model classes.

These do not change any model math or parameters; they monkey-patch the
`forward` method to route through the CUDA kernels.

    from neural_shift_cuda.integration import install_cuda_shift_attn
    from NKD_drunet_attn_v4 import NeKDeDRUNetAttn
    install_cuda_shift_attn(NeKDeDRUNetAttn)

For the classic nekre model:

    from neural_shift_cuda.integration import install_cuda_shift_nekre
    from NKD_models_symm import nekre
    install_cuda_shift_nekre(nekre)
"""

from .nekde_drunet_attn_patch import install_cuda_shift as install_cuda_shift_attn
from .nekde_drunet_attn_v5_patch import install_cuda_shift as install_cuda_shift_attn_v5
from .nekre_patch import install_cuda_shift as install_cuda_shift_nekre

__all__ = [
    "install_cuda_shift_attn",
    "install_cuda_shift_attn_v5",
    "install_cuda_shift_nekre",
]
