from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os
import re
import subprocess

# Resolve sources relative to setup.py (project root).
_HERE = os.path.dirname(os.path.abspath(__file__))


def _src(p):
    return os.path.join(_HERE, p)


def _cuda_toolkit_version():
    """(major, minor) of the nvcc that will actually do the compile, or None.

    We query the *real* nvcc (from CUDA_HOME, else PATH) rather than
    torch.version.cuda, because whether `code=sm_120` is a legal target is
    decided by the local toolkit doing the build, not by the toolkit torch
    happened to be built against. Fall back to torch.version.cuda only if nvcc
    can't be probed.
    """
    from torch.utils.cpp_extension import CUDA_HOME

    nvcc = "nvcc"
    if CUDA_HOME:
        cand = os.path.join(CUDA_HOME, "bin", "nvcc")
        if os.path.exists(cand):
            nvcc = cand
    try:
        out = subprocess.check_output([nvcc, "--version"], text=True)
        m = re.search(r"release (\d+)\.(\d+)", out)
        if m:
            return int(m.group(1)), int(m.group(2))
    except Exception:
        pass

    import torch
    if torch.version.cuda:
        major, minor = torch.version.cuda.split(".")[:2]
        return int(major), int(minor)
    return None


# Base SASS targets built on every toolkit.
# We use atomicAdd on float and double; double atomicAdd requires SM 6.0 or
# newer (Pascal+), satisfied by all of these.
_nvcc_args = [
    "-O3",
    "--use_fast_math",
    "-gencode=arch=compute_86,code=sm_86",
    "-gencode=arch=compute_89,code=sm_89",
    "-gencode=arch=compute_90,code=sm_90",
]

_cuda_ver = _cuda_toolkit_version()
if _cuda_ver is not None and _cuda_ver >= (13, 0):
    # Blackwell SASS + matching PTX. compute_120 is a valid nvcc target only on
    # recent toolkits; per build policy we gate it on CUDA >= 13.0. Note the
    # PTX line stays compute_120 (same major family as sm_120) -- pairing
    # compute_90 PTX with code=sm_120 is an invalid cross-family gencode.
    _nvcc_args += [
        "-gencode=arch=compute_120,code=sm_120",
        "-gencode=arch=compute_120,code=compute_120",
    ]
else:
    # No sm_120 SASS on this toolkit. Emit sm_90 PTX so the driver can JIT it
    # to Blackwell (sm_120) at runtime instead of failing with "no kernel image
    # available". PTX is forward-compatible across arch families (Hopper ->
    # Blackwell); SASS is not, which is why we can't just add code=sm_120 here.
    _nvcc_args += [
        "-gencode=arch=compute_90,code=compute_90",
    ]

setup(
    name="neural_shift_cuda",
    version="0.10.7",
    description="CUDA shift-gather / pair-gather / accumulate (per-pixel and scalar-per-transform) ops for NeKDe / GASD / GADSD denoisers.",
    # picks up neural_shift_cuda/ and neural_shift_cuda/integration/
    packages=find_packages(),
    # ship the kernel sources inside the wheel/sdist so a non-editable
    # install can still rebuild the extension on the target machine.
    package_data={"neural_shift_cuda": ["../csrc/*.cpp", "../csrc/*.cu"]},
    include_package_data=True,
    ext_modules=[
        CUDAExtension(
            name="neural_shift_cuda._C",
            sources=[
                _src("csrc/shift_ops.cpp"),
                _src("csrc/shift_ops_cuda.cu"),
                _src("csrc/metropolis.cpp"),
                _src("csrc/metropolis_cuda.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": _nvcc_args,
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    # torch is deliberately NOT in install_requires. This is a compiled torch
    # extension: torch must already be importable to BUILD it (setup.py imports
    # torch.utils.cpp_extension, and we build with --no-build-isolation against
    # the ambient env). Listing it here does nothing useful on a normal install
    # (the existing torch already satisfies any version spec) but is actively
    # harmful with --force-reinstall, which reinstalls every declared dep and,
    # against an unpinned spec, pulls the latest torch wheel -- overwriting a
    # pinned CUDA build (e.g. 2.5.1+cu124) with a mismatched one. Omitting torch
    # lets you drop the --no-deps guard:
    #   pip install --no-build-isolation --force-reinstall "git+...".
    # einops is only needed by the legacy nekre_patch, so it is an extra, not a
    # core requirement (the GASD / NeKDe paths import only torch).
    install_requires=[],
    extras_require={"nekre": ["einops>=0.6"]},
    python_requires=">=3.8",
    zip_safe=False,
)