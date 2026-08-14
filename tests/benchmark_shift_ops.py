# tests/benchmark_shift_ops.py
#
# Run with:
#   python tests/benchmark_shift_ops.py --R 3 --H 128 --W 128 --B 4
#
# Reports:
#   * shift creation time (python list-comp vs CUDA shift_gather/pair_gather)
#   * peak CUDA memory
#   * (optional) full model forward time using either path
#
# Usage requires CUDA. The extension must be installed first
# (pip install -v -e .).

from __future__ import annotations

import argparse
import importlib.util
import os
import time
import types
import statistics as stats

import torch
import torch.nn.functional as F

from neural_shift_cuda import (
    shift_gather, pair_gather, normalized_accumulate_uz,
)


# ---------------------------------------------------------------------------
# Python-side reference for the original list-comprehension approach
# ---------------------------------------------------------------------------

def _all_shifts(R):
    rows = []
    for dx in range(-R, R + 1):
        for dy in range(1, R + 1):
            rows.append((dx, dy, 1))
    for dx in range(0, R + 1):
        rows.append((dx, 0, 0 if dx == 0 else 1))
    return rows


def py_shift_gather(guide, R, shifts_list):
    B, C, H, W = guide.shape
    padded_guide = F.pad(guide, (R, R, R, R), mode='circular')
    box = F.pad(torch.ones(1, 1, H, W, device=guide.device, dtype=guide.dtype),
                (R, R, R, R), mode='constant', value=0)
    gs = [padded_guide[:, :, R + dx:R + dx + H, R + dy:R + dy + W]
          for dx, dy, _ in shifts_list]
    ms = [box[:, :, R + dx:R + dx + H, R + dy:R + dy + W]
          for dx, dy, _ in shifts_list]
    gs_batch = torch.cat(gs, dim=0)
    mask = torch.cat(ms, dim=0).expand(-1, 1, -1, -1)
    return gs_batch, mask


def py_pair_gather(guide, R, shifts_list):
    gs_batch, mask = py_shift_gather(guide, R, shifts_list)
    S = gs_batch.size(0) // guide.size(0)
    g_batch = guide.repeat(S, 1, 1, 1)
    pair = torch.cat([g_batch, gs_batch], dim=1)
    return pair, mask


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------

def bench(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    timings = []
    for _ in range(iters):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        timings.append(start.elapsed_time(end))  # ms
    return min(timings), stats.median(timings), max(timings)


def peak_mem_mb(fn):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


# ---------------------------------------------------------------------------
# Standalone shift / pair gather benchmark
# ---------------------------------------------------------------------------

def run_gather_bench(B, C, H, W, R):
    print(f"\n[gather] B={B} C={C} H={H} W={W} R={R}  "
          f"(S = {2*R*R + 2*R + 1} shifts)")

    guide = torch.randn(B, C, H, W, device='cuda', dtype=torch.float32)
    shifts_list = _all_shifts(R)
    shifts = torch.tensor([(dx, dy, hi) for dx, dy, hi in shifts_list],
                          dtype=torch.int32, device='cuda')

    # --- shift_gather ---
    def f_py():
        gs, m = py_shift_gather(guide, R, shifts_list)
        return gs, m

    def f_cu():
        gs, m = shift_gather(guide, shifts)
        return gs, m

    py_t = bench(f_py)
    cu_t = bench(f_cu)
    py_mem = peak_mem_mb(f_py)
    cu_mem = peak_mem_mb(f_cu)
    print(f"  shift_gather:  python={py_t[1]:7.3f} ms (peak {py_mem:6.1f} MB) | "
          f"cuda={cu_t[1]:7.3f} ms (peak {cu_mem:6.1f} MB) | "
          f"speedup={py_t[1] / cu_t[1]:5.2f}x")

    # --- pair_gather ---
    def f_py_pair():
        p, m = py_pair_gather(guide, R, shifts_list)
        return p, m

    def f_cu_pair():
        p, m = pair_gather(guide, shifts)
        return p, m

    py_t = bench(f_py_pair)
    cu_t = bench(f_cu_pair)
    py_mem = peak_mem_mb(f_py_pair)
    cu_mem = peak_mem_mb(f_cu_pair)
    print(f"  pair_gather:   python={py_t[1]:7.3f} ms (peak {py_mem:6.1f} MB) | "
          f"cuda={cu_t[1]:7.3f} ms (peak {cu_mem:6.1f} MB) | "
          f"speedup={py_t[1] / cu_t[1]:5.2f}x")


# ---------------------------------------------------------------------------
# Optional: full nekre forward benchmark (only if NKD_models_symm.py is found)
# ---------------------------------------------------------------------------

def _load_nekre():
    for p in [
        os.environ.get("NKD_MODELS_PATH", ""),
        "NKD_models_symm.py",
        "NKD_models_symm__2_.py",
        "/mnt/user-data/uploads/NKD_models_symm__2_.py",
    ]:
        if p and os.path.exists(p):
            spec = importlib.util.spec_from_file_location("nkd_models_symm", p)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            return mod
    return None


def _compute_weights_from_pair(model, pair_batch):
    from einops import rearrange
    x = pair_batch
    for proj in model.proj:
        for i, sub_layer in enumerate(proj):
            if i == 1:
                x = rearrange(x, 'b c h w -> b h w c')
                x = sub_layer(x)
                x = rearrange(x, 'b h w c -> b c h w')
            else:
                x = sub_layer(x)
    x = model.out(x)
    if model.output_activation == 'exp':
        x = torch.exp(x / model.sigma ** 2)
    elif model.output_activation == 'control_sigmoid':
        k, a, b = torch.chunk(x, dim=1, chunks=3)
        x = model.controlled_sigmoid(k, a, b) + 1e-8
    return x


def _shift_tensor(model, device):
    rows = [(dx, dy, int(bool(hi))) for dx, dy, hi in model._collect_shifts()]
    return torch.tensor(rows, dtype=torch.int32, device=device)


def run_model_bench(B, C_img, H, W, R):
    nkd = _load_nekre()
    if nkd is None:
        print("\n[model] NKD_models_symm.py not found -- skipping full-model benchmark.")
        return

    import argparse as _ap
    args = _ap.Namespace(
        model_type='concat',
        residual_depth=2,
        proj_depth=1,
        latent_dim=16,
        window_rad=R,
        in_channel=C_img,
        patch_rad=1,
        output_activation='exp',
        blind=True,
    )
    model = nkd.nekre(args, device='cuda').to('cuda').eval()
    model.compute_weights_from_pair = types.MethodType(_compute_weights_from_pair, model)
    x = torch.randn(B, C_img, H, W, device='cuda', dtype=torch.float32)
    shifts = _shift_tensor(model, 'cuda')

    def f_old():
        with torch.no_grad():
            return model.forward(x, batched=True)

    def f_new():
        with torch.no_grad():
            guide = model.pre_activation(x, sigma=None)
            pair_batch, mask = pair_gather(guide.contiguous(), shifts)
            weights = model.compute_weights_from_pair(pair_batch)
            weights = weights * mask
            U, log_Z = normalized_accumulate_uz(
                x.contiguous(), weights.contiguous(), shifts,
                return_log_degree=True, validate=False)
            return U, log_Z.exp()

    print(f"\n[model] B={B} C_img={C_img} H={H} W={W} R={R}")
    old_t = bench(f_old, warmup=3, iters=10)
    new_t = bench(f_new, warmup=3, iters=10)
    old_mem = peak_mem_mb(f_old)
    new_mem = peak_mem_mb(f_new)
    print(f"  old forward:   {old_t[1]:8.2f} ms (peak {old_mem:7.1f} MB)")
    print(f"  new forward:   {new_t[1]:8.2f} ms (peak {new_mem:7.1f} MB)")
    print(f"  speedup:       {old_t[1] / new_t[1]:5.2f}x   "
          f"mem ratio (new/old): {new_mem / old_mem:.2f}")


# ---------------------------------------------------------------------------

def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the benchmark.")

    p = argparse.ArgumentParser()
    p.add_argument("--B", type=int, default=4)
    p.add_argument("--C", type=int, default=32, help="guide channels for gather bench")
    p.add_argument("--C_img", type=int, default=3, help="image channels for model bench")
    p.add_argument("--H", type=int, default=128)
    p.add_argument("--W", type=int, default=128)
    p.add_argument("--R", type=int, default=3)
    p.add_argument("--model", action="store_true",
                   help="also run the end-to-end nekre forward benchmark.")
    a = p.parse_args()

    run_gather_bench(a.B, a.C, a.H, a.W, a.R)

    if a.model:
        run_model_bench(a.B, a.C_img, a.H, a.W, a.R)


if __name__ == "__main__":
    main()
