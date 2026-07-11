"""Integration shims that wire neural_shift_cuda into model classes.

These do not change any model math or parameters; they monkey-patch the
`forward` method to route through the CUDA kernels.

NeKDe (half-plane, per-pixel weights, inverse-symmetry). The model's
``comp_box`` flag is honoured on the CUDA path for both values (True -> masked /
truncated; False -> fully periodic, realised by enumerating each shift's
explicit inverse with has_inverse=0, so accumulate_uz's masked inverse branch is
bypassed). One installer covers both:

    from neural_shift_cuda.integration import install_cuda_shift_attn
    from NKD_drunet_attn_v2 import NeKDeDRUNetAttn      # v2 / v3 / v4
    install_cuda_shift_attn(NeKDeDRUNetAttn)

GADSD (fully circular, per-channel group averaging + D4, doubly-stochastic):

    from neural_shift_cuda.integration import install_cuda_shift_gadsd
    from GADSD_drunet_attn import GADSDDRUNetAttn
    install_cuda_shift_gadsd(GADSDDRUNetAttn)

GADSD lightweight (stack-consuming pairwise-moment head, per-channel scalar
weights; patches _transform_weights AND forward):

    from neural_shift_cuda.integration import install_cuda_shift_gadsd_lightweight
    from GADSD_lightweight import GADSDLightweight
    install_cuda_shift_gadsd_lightweight(GADSDLightweight)

GASD (full (2R+1)^2 window, per-pixel weights, row-stochastic Z^{-1}U). The
model's ``comp_box`` flag toggles boundary handling at runtime (True ->
comp_box / masked; False -> fully circular):

    from neural_shift_cuda.integration import install_cuda_shift_gasd
    from GASD_drunet_attn_v2 import GASDDRUNetAttn
    install_cuda_shift_gasd(GASDDRUNetAttn)

Classic nekre model:

    from neural_shift_cuda.integration import install_cuda_shift_nekre
    from NKD_models_symm import nekre
    install_cuda_shift_nekre(nekre)
"""

# NeKDe (half-plane / per-pixel accumulate_uz path; comp_box toggled at runtime)
from .nekde_drunet_attn_patch import install_cuda_shift as install_cuda_shift_attn

# GADSD (per-channel accumulate_uz_scalar path, doubly-stochastic)
from .gadsd_drunet_attn_patch import install_cuda_shift as install_cuda_shift_gadsd
from .gadsd_lightweight_patch import install_cuda_shift as install_cuda_shift_gadsd_lightweight

# GASD (per-pixel accumulate_uz path, row-stochastic; full (2R+1)^2 window)
from .gasd_drunet_attn_patch import install_cuda_shift as install_cuda_shift_gasd

# Other model families
from .nekre_patch import install_cuda_shift as install_cuda_shift_nekre
from .nkd_metropolis_attn_patch import install_cuda_shift as install_cuda_shift_metropolis

__all__ = [
    "install_cuda_shift_attn",
    "install_cuda_shift_gadsd",
    "install_cuda_shift_gadsd_lightweight",
    "install_cuda_shift_gasd",
    "install_cuda_shift_nekre",
    "install_cuda_shift_metropolis",
]
