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

GADSD moments (stack-consuming pairwise-moment head, per-channel scalar
weights; patches _transform_weights AND forward). NOTE: the model file
GADSD_lightweight.py was renamed to GADSD_moments.py and the class
GADSDLightweight to GADSDMoments; ``install_cuda_shift_gadsd_lightweight``
is kept as a back-compat alias of ``install_cuda_shift_gadsd_moments``:

    from neural_shift_cuda.integration import install_cuda_shift_gadsd_moments
    from GADSD_moments import GADSDMoments
    install_cuda_shift_gadsd_moments(GADSDMoments)

GASD (full (2R+1)^2 window, per-pixel weights, row-stochastic Z^{-1}U). The
model's ``comp_box`` flag toggles boundary handling at runtime (True ->
comp_box / masked; False -> fully circular):

    from neural_shift_cuda.integration import install_cuda_shift_gasd
    from GASD_drunet_attn_v2 import GASDDRUNetAttn
    install_cuda_shift_gasd(GASDDRUNetAttn)

Moments-branch NeKDe / GASD / Metropolis (the *_moments ports whose weight
producer is the GADSD moments pair -- TinyMomentFeatureExtractor +
per-pixel PixelwiseMomentHead -- but whose kernel assembly matches the
respective drunet_attn file). The kernel structure per class is identical
to the corresponding attn patch; only the head API differs (the moment head
takes an extra per-shift analytic descriptor argument):

    from neural_shift_cuda.integration import install_cuda_shift_nekde_moments
    from NKD_moments import NeKDeMoments
    install_cuda_shift_nekde_moments(NeKDeMoments)

    from neural_shift_cuda.integration import install_cuda_shift_gasd_moments
    from GASD_moments import GASDMoments
    install_cuda_shift_gasd_moments(GASDMoments)

    from neural_shift_cuda.integration import install_cuda_shift_metropolis_moments
    from NKD_mp_moments import NeKDeMetropolisMoments
    install_cuda_shift_metropolis_moments(NeKDeMetropolisMoments)

Classic nekre model:

    from neural_shift_cuda.integration import install_cuda_shift_nekre
    from NKD_models_symm import nekre
    install_cuda_shift_nekre(nekre)
"""

# NeKDe (half-plane / per-pixel accumulate_uz path; comp_box toggled at runtime)
from .nekde_drunet_attn_patch import install_cuda_shift as install_cuda_shift_attn

# GADSD (per-channel accumulate_uz_scalar path, doubly-stochastic)
from .gadsd_drunet_attn_patch import install_cuda_shift as install_cuda_shift_gadsd
from .gadsd_moments_patch import install_cuda_shift as install_cuda_shift_gadsd_moments


# GASD (per-pixel accumulate_uz path, row-stochastic; full (2R+1)^2 window)
from .gasd_drunet_attn_patch import install_cuda_shift as install_cuda_shift_gasd

# Moments-branch models (PixelwiseMomentHead weight producer)
from .nekde_moments_patch import install_cuda_shift as install_cuda_shift_nekde_moments
from .gasd_moments_patch import install_cuda_shift as install_cuda_shift_gasd_moments
from .nkd_mp_moments_patch import install_cuda_shift as install_cuda_shift_metropolis_moments

# Other model families
from .nekre_patch import install_cuda_shift as install_cuda_shift_nekre
from .nectr_patch import install_cuda_shift as install_cuda_shift_nectr
from .nkd_metropolis_attn_patch import install_cuda_shift as install_cuda_shift_metropolis

__all__ = [
    "install_cuda_shift_attn",
    "install_cuda_shift_gadsd",
    "install_cuda_shift_gadsd_moments",
    "install_cuda_shift_gasd",
    "install_cuda_shift_nekde_moments",
    "install_cuda_shift_gasd_moments",
    "install_cuda_shift_metropolis_moments",
    "install_cuda_shift_nekre",
    "install_cuda_shift_nectr",
    "install_cuda_shift_metropolis",
]
