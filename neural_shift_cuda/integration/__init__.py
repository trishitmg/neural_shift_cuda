"""Integration shims that wire neural_shift_cuda into model classes.

These do not change any model math or parameters; they monkey-patch the
`forward` method to route through the CUDA kernels.

NeKDe (half-plane, per-pixel weights, comp_box / inverse-symmetry):

    from neural_shift_cuda.integration import install_cuda_shift_attn
    from NKD_drunet_attn_v4 import NeKDeDRUNetAttn      # v2 / v3 / v4
    install_cuda_shift_attn(NeKDeDRUNetAttn)

    from neural_shift_cuda.integration import install_cuda_shift_attn_v5
    from NKD_drunet_attn_v5 import NeKDeDRUNetAttn
    install_cuda_shift_attn_v5(NeKDeDRUNetAttn)

GADSD (fully circular, per-channel group averaging + D4, doubly-stochastic):

    from neural_shift_cuda.integration import install_cuda_shift_gadsd
    from GADSD_drunet_attn import GADSDDRUNetAttn
    install_cuda_shift_gadsd(GADSDDRUNetAttn)

GASD (full (2R+1)^2 window, per-pixel weights, row-stochastic Z^{-1}U):

    from neural_shift_cuda.integration import install_cuda_shift_gasd
    from GASD_drunet_attn_v2 import GASDDRUNetAttn          # v2 (comp_box)
    install_cuda_shift_gasd(GASDDRUNetAttn)

    from neural_shift_cuda.integration import install_cuda_shift_gasd_v5
    from GASD_drunet_attn_v5 import GASDDRUNetAttn          # v5 (fully circular)
    install_cuda_shift_gasd_v5(GASDDRUNetAttn)

Classic nekre model:

    from neural_shift_cuda.integration import install_cuda_shift_nekre
    from NKD_models_symm import nekre
    install_cuda_shift_nekre(nekre)
"""

# NeKDe (original half-plane / per-pixel accumulate_uz path)
from .nekde_drunet_attn_patch import install_cuda_shift as install_cuda_shift_attn
from .nekde_drunet_attn_v5_patch import install_cuda_shift as install_cuda_shift_attn_v5

# GADSD (per-channel accumulate_uz_scalar path, doubly-stochastic)
from .gadsd_drunet_attn_patch import install_cuda_shift as install_cuda_shift_gadsd

# GASD (per-pixel accumulate_uz path, row-stochastic; full (2R+1)^2 window)
from .gasd_drunet_attn_patch import install_cuda_shift as install_cuda_shift_gasd
from .gasd_drunet_attn_v5_patch import install_cuda_shift as install_cuda_shift_gasd_v5

# Other model families
from .nekre_patch import install_cuda_shift as install_cuda_shift_nekre
from .nkd_metropolis_attn_patch import install_cuda_shift as install_cuda_shift_metropolis
from .nkd_metropolis_attn_v5_patch import install_cuda_shift as install_cuda_shift_metropolis_v5

__all__ = [
    "install_cuda_shift_attn",
    "install_cuda_shift_attn_v5",
    "install_cuda_shift_gadsd",
    "install_cuda_shift_gasd",
    "install_cuda_shift_gasd_v5",
    "install_cuda_shift_nekre",
    "install_cuda_shift_metropolis",
    "install_cuda_shift_metropolis_v5",
]
