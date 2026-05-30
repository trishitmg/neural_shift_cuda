# neural_shift_cuda

CUDA kernels that accelerate the shift-gather and U/Z accumulation in the
DSG-NLM forward of the NeKDe / NeKRe denoisers (`nekre`,
`NeKDeDRUNetAttn` v2/v3/v4).

Three fused ops, each with a pure-PyTorch CPU fallback and full autograd:

- `shift_gather(guide, shifts)`   -> circular-shifted guide + validity mask
- `pair_gather(guide, shifts)`    -> shift_gather with center guide concatenated
- `accumulate_uz(x, weights, shifts)` -> fused U/Z accumulation w/ inverse symmetry

## Requirements

- A CUDA-capable GPU (SM 6.0 / Pascal or newer — needed for double `atomicAdd`).
- The CUDA toolkit (`nvcc`) matching the CUDA your PyTorch was built against.
  Check with `python -c "import torch; print(torch.version.cuda)"` and
  `nvcc --version` — the major version must match.
- PyTorch >= 1.12, already installed in the target environment.

The extension is compiled **on the machine that installs it** (it is not a
prebuilt wheel), so `nvcc` and a matching PyTorch must be present at install
time.

## Install

Because `setup.py` imports `torch` at build time, build isolation must be
disabled (so the build sees your already-installed torch):

```bash
pip install --no-build-isolation \
    "git+https://github.com/<you>/neural_shift_cuda.git@main"
```

Pin to a tag or commit for reproducible runs:

```bash
pip install --no-build-isolation \
    "git+https://github.com/<you>/neural_shift_cuda.git@v0.1.0"
```

Editable clone (recommended while iterating on the kernels):

```bash
git clone https://github.com/<you>/neural_shift_cuda.git
cd neural_shift_cuda
pip install --no-build-isolation -v -e .
```

Verify:

```bash
python -c "from neural_shift_cuda import shift_gather, accumulate_uz; print('ok')"
```

## Use

```python
from NKD_drunet_attn_v4 import NeKDeDRUNetAttn          # or v2 / v3
from neural_shift_cuda.integration import install_cuda_shift_attn
install_cuda_shift_attn(NeKDeDRUNetAttn)                 # once, before building any model
```

Classic nekre:

```python
from NKD_models_symm import nekre
from neural_shift_cuda.integration import install_cuda_shift_nekre
install_cuda_shift_nekre(nekre)
```

Disable per-instance: `model.use_cuda_shift = False`.

## Test

```bash
pytest -q tests/test_shift_gather.py tests/test_accumulate_uz.py
NKD_ATTN_PATH=/path/to/NKD_drunet_attn_v4.py pytest -q tests/test_attn_forward_equivalence.py
```
