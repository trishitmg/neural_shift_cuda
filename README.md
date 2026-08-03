# neural_shift_cuda

Fused CUDA ops for the shift-gather and stochastic-weight accumulation in the
DSG-NLM forward of the NeKDe-family denoisers. Each op has a pure-PyTorch CPU
fallback and full autograd; the compiled kernel is used on CUDA at inference,
the reference path during training.

The kernels are installed as drop-in patches: they monkey-patch a model class's
`forward` to route through the CUDA ops without changing any math, parameters,
or outputs (bit-exact against the reference within fp tolerance).

## How the parallelization works

These denoisers restore an image by, for every pixel, looking at a small window
of neighbouring pixels and replacing the pixel with a *weighted average* of
them — where the weights are produced by a small neural network that judges how
similar each neighbour is. A window of radius `R` has `S = 2*R^2 + 2*R + 1`
distinct neighbour offsets, which we call *shifts*.

The reference implementation walks these shifts one at a time in a Python loop:
for each shift it slides the image by that offset, runs the weight network, and
adds the result into a running numerator `U` and normaliser `Z`. That means `S`
separate passes, each launching many tiny GPU operations with Python overhead in
between, so the GPU spends most of its time waiting between small launches rather
than computing.

`neural_shift_cuda` replaces that loop with a few fused CUDA kernels that handle
all shifts together:

- **Gather once.** `shift_gather` / `pair_gather` collect every shifted neighbour
  window (and the border-validity mask) in a single kernel, instead of padding
  and slicing `S` times.
- **Score once.** The weight network then runs a single time on the whole stack
  of shifts rather than `S` times.
- **Accumulate once, using symmetry.** `accumulate_uz` (or `metropolis_aggregate`
  for the Metropolis variant) sums each shift's weighted contribution into `U`
  and `Z` in one fused kernel. Because the kernel is symmetric — a shift and its
  mirror share the same weight — only about half the shifts need a network
  evaluation; the mirror contribution is recovered by an inverse shift *inside*
  the kernel.

The net effect is a handful of large, GPU-friendly kernel launches instead of
hundreds of small serialized ones, so the GPU stays busy. The math is unchanged
(outputs match the reference to fp tolerance), and because the fused kernels are
differentiable the speed-up carries through the backward pass — which is why it
shows up directly as more training iterations per second.

## Supported model archs

CUDA parallel patches are currently available for the three latest attention
denoisers plus the original model:

| Model arch | Class (source) | Installer | Kernel | Paper Link and Code |
|---|---|---|---|---|
| NeKDe attn | `NeKDeDRUNetAttn` (`NKD_drunet_attn_v2.py`, v2/v3/v4) | `install_cuda_shift_attn` | `accumulate_uz` | _to be updated soon_ |
| GASD attn | `GASDDRUNetAttn` (`GASD_drunet_attn_v2.py`) | `install_cuda_shift_gasd` | `accumulate_uz` | _to be updated soon_ |
| NKD_mp attn | `NeKDeMetropolisDRUNetAttn` (`NKD_mp_drunet_attn_v2.py`) | `install_cuda_shift_metropolis` | `metropolis_aggregate` | _to be updated soon_ |
| NECTR (original) | `nekre` (`NKD_models_symm.py`) | `install_cuda_shift_nekre` | `shift_gather`, `pair_gather`, `accumulate_uz` | [![arXiv](https://img.shields.io/badge/arXiv-2607.23347-b31b1b?logo=arxiv&logoColor=white)](https://arxiv.org/abs/2607.23347) [![Code](https://img.shields.io/badge/Code-181717?logo=github&logoColor=white)](https://github.com/arghyasinha/nectr) |

- **NeKDe attn** — half-plane shifts, per-pixel weights, inverse-symmetry. The
  model's `comp_box` flag is read per-forward (True → finite-window mask; False →
  fully periodic).
- **GASD attn** — full `(2R+1)^2` window, per-pixel weights, row-stochastic
  `Z^{-1}U`. `comp_box` toggles boundary handling at runtime.
- **NKD_mp attn** — Metropolis normalization; the inverse edge is formed by a
  circular shift of the *post-activation* forward weight inside the op, so `W` is
  exactly symmetric and nonexpansive regardless of the output activation.
- **NECTR (original)** — the original model from the paper (`nekre` class). Only
  the `batched=True` branch of `forward` is patched; the non-batched path is
  untouched. `pair_gather` fuses the center/shifted concatenation, then
  `accumulate_uz` does the forward + inverse-symmetric U/Z accumulation with the
  same box validity mask as the reference. Two lazy-init flags gate it:
  `use_cuda_shift` (whole CUDA path) and `use_cuda_pair_gather` (the fused
  gather). Tolerance vs the reference is ~1e-5 in fp32, exact in fp64. This is
  the only arch that needs the `einops` extra (see Install).

Patches for further archs (GADSD, moments-branch) exist internally and will be
listed here as they are released.

## Requirements

- CUDA-capable GPU, SM 6.0 / Pascal or newer (double `atomicAdd`).
- CUDA toolkit (`nvcc`) whose major version matches the CUDA your PyTorch was
  built against. Blackwell `sm_120` SASS is emitted only when the build toolkit
  is CUDA >= 13.0; older toolkits fall back to `sm_90` PTX (driver JIT).
- PyTorch >= 1.12, already installed in the target environment.

The extension compiles **on the installing machine** (no prebuilt wheel), so
`nvcc` and a matching PyTorch must be present at install time.

## Install

`setup.py` imports `torch` at build time, so disable build isolation:

```bash
pip install --no-build-isolation --force-reinstall -v \
    "git+https://github.com/007Trishit/neural_shift_cuda.git@main"
```

The NECTR (`nekre`) patch additionally imports `einops`; install the extra if
you use it:

```bash
pip install --no-build-isolation --force-reinstall -v \
    "git+https://github.com/007Trishit/neural_shift_cuda.git@main#egg=neural_shift_cuda[nekre]"
```

Editable clone (while iterating on kernels):

```bash
git clone https://github.com/007Trishit/neural_shift_cuda.git
cd neural_shift_cuda
pip install --no-build-isolation -v -e .
```

Verify:

```bash
python -c "from neural_shift_cuda import accumulate_uz, metropolis_aggregate; print('ok')"
```

## Use

Call the installer **once**, before constructing any model:

```python
# NeKDe attn
from NKD_drunet_attn_v2 import NeKDeDRUNetAttn
from neural_shift_cuda.integration import install_cuda_shift_attn
install_cuda_shift_attn(NeKDeDRUNetAttn)

# GASD attn
from GASD_drunet_attn_v2 import GASDDRUNetAttn
from neural_shift_cuda.integration import install_cuda_shift_gasd
install_cuda_shift_gasd(GASDDRUNetAttn)

# NKD_mp attn
from NKD_mp_drunet_attn_v2 import NeKDeMetropolisDRUNetAttn
from neural_shift_cuda.integration import install_cuda_shift_metropolis
install_cuda_shift_metropolis(NeKDeMetropolisDRUNetAttn)

# NECTR (original model) — routes only the batched=True forward
from NKD_models_symm import nekre
from neural_shift_cuda.integration import install_cuda_shift_nekre
install_cuda_shift_nekre(nekre)
```

Force the reference PyTorch path per instance: `model.use_cuda_shift = False`.

## Benchmarks

Serial (reference PyTorch per-shift loop) vs parallel (`neural_shift_cuda`) on
the same GPU, measured with `tests/benchmark_archs.py`. **All four models use
identical settings: batch size `B = 8`, image size `64 x 64`, window radius
`R = 5`, `fp32`.** Rows are the stages of a training step; the columns give the
median wall-clock time of each path and their ratio (serial ÷ parallel). Higher
speed-up is better; the **full step (fwd+bwd)** row is what training
iterations/second track.

### NeKDe attn
| Stage | Serial (ms) | Parallel (ms) | Speed-up |
|---|--:|--:|--:|
| Forward (inference) | 19.17 | 12.45 | 1.54× |
| Forward (train) | 24.15 | 12.32 | 1.96× |
| Backward | 116.22 | 50.72 | 2.29× |
| **Full step (fwd+bwd)** | **140.37** | **63.03** | **2.23×** |

### GASD attn
| Stage | Serial (ms) | Parallel (ms) | Speed-up |
|---|--:|--:|--:|
| Forward (inference) | 22.33 | 21.09 | 1.06× |
| Forward (train) | 25.56 | 20.97 | 1.22× |
| Backward | 151.45 | 113.83 | 1.33× |
| **Full step (fwd+bwd)** | **177.02** | **134.81** | **1.31×** |

### NKD_mp attn
| Stage | Serial (ms) | Parallel (ms) | Speed-up |
|---|--:|--:|--:|
| Forward (inference) | 34.32 | 12.73 | 2.70× |
| Forward (train) | 43.94 | 15.41 | 2.85× |
| Backward | 148.99 | 75.61 | 1.97× |
| **Full step (fwd+bwd)** | **192.92** | **91.02** | **2.12×** |

### NECTR (original)
| Stage | Serial (ms) | Parallel (ms) | Speed-up |
|---|--:|--:|--:|
| Forward (inference) | 91.00 | 82.92 | 1.10× |
| Forward (train) | 91.19 | 82.91 | 1.10× |
| Backward | 226.17 | 167.65 | 1.35× |
| **Full step (fwd+bwd)** | **317.36** | **250.56** | **1.27×** |

Reproduce with `python tests/benchmark_archs.py --config tests/configs/<arch>.yaml`.

## Test

```bash
pytest -q tests/test_shift_gather.py tests/test_accumulate_uz.py
pytest -q tests/test_attn_forward_equivalence.py
```