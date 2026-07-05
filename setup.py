from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

# Resolve sources relative to setup.py (project root).
_HERE = os.path.dirname(os.path.abspath(__file__))


def _src(p):
    return os.path.join(_HERE, p)


setup(
    name="neural_shift_cuda",
    version="0.4.6",
    description="CUDA shift-gather / pair-gather / accumulate (per-pixel and "
                "scalar-per-transform) ops for NeKDe / GASD denoisers.",
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
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    # We use atomicAdd on float and double; double atomicAdd
                    # requires SM 6.0 or newer (Pascal+). Most modern GPUs.
                    "-gencode=arch=compute_60,code=sm_60",
                    "-gencode=arch=compute_70,code=sm_70",
                    "-gencode=arch=compute_75,code=sm_75",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-gencode=arch=compute_86,code=sm_86",
                    "-gencode=arch=compute_89,code=sm_89",
                    "-gencode=arch=compute_90,code=sm_90",
                ],
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
