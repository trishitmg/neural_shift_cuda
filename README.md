# neural_shift_cuda

Fused CUDA ops for the shift-gather and stochastic-weight accumulation in the
DSG-NLM forward of the NeKDe-family denoisers. Each op has a pure-PyTorch CPU
fallback and full autograd; the compiled kernel is used on CUDA at inference,
the reference path during training.

- Official model / paper repo: https://github.com/arghyasinha/nectr
- Paper: https://openreview.net/forum?id=Z6j8S5LWmL

The kernels are installed as drop-in patches: they monkey-patch a model class's
`forward` to route through the CUDA ops without changing any math, parameters,
or outputs (bit-exact against the reference within fp tolerance).

## Supported model archs

CUDA parallel patches are currently available for the three latest attention
denoisers:

| Model arch | Class (source) | Installer | Kernel |
|---|---|---|---|
| NeKDe attn | `NeKDeDRUNetAttn` (`NKD_drunet_attn_v2.py`, v2/v3/v4) | `install_cuda_shift_attn` | `accumulate_uz` |
| GASD attn | `GASDDRUNetAttn` (`GASD_drunet_attn_v2.py`) | `install_cuda_shift_gasd` | `accumulate_uz` |
| NKD_mp attn | `NeKDeMetropolisDRUNetAttn` (`NKD_mp_drunet_attn_v2.py`) | `install_cuda_shift_metropolis` | `metropolis_aggregate` |
| NECTR (original) | `nekre` (`NKD_models_symm.py`) | `install_cuda_shift_nekre` | `shift_gather`, `pair_gather`, `accumulate_uz` |

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

## Test

```bash
pytest -q tests/test_shift_gather.py tests/test_accumulate_uz.py
pytest -q tests/test_attn_forward_equivalence.py
```