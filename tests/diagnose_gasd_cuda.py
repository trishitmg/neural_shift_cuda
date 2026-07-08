# Diagnose why the GASD CUDA path is slower than the reference forward.
# Run on the training machine (needs a GPU):
#
#   python tests/diagnose_gasd_cuda.py --model-file GASD_drunet_attn.py
#
# Checks, in order:
#   [1] which _C binary is imported and whether the 0.4.x scalar symbols exist
#       (a stale binary silently routed accumulate_uz_scalar to the PyTorch
#        reference, whose per-shift .item() calls = ~242 stream syncs/forward);
#   [2] sync-stall count per forward+backward for both paths (torch.profiler,
#       cudaStreamSynchronize / cudaDeviceSynchronize events);
#   [3] phase timing (gather / head / accumulate / backward) for both paths.

import argparse
import importlib.util
import sys
import time

import torch


def load_model_class(path):
    spec = importlib.util.spec_from_file_location("gasd_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.GASDDRUNetAttn


def check_extension():
    print("=" * 70)
    print("[1] extension binary")
    import neural_shift_cuda
    print("    neural_shift_cuda", neural_shift_cuda.__version__,
          "from", neural_shift_cuda.__file__)
    from neural_shift_cuda import ops
    print("    _HAS_CUDA_EXT      =", ops._HAS_CUDA_EXT)
    if ops._HAS_CUDA_EXT:
        from neural_shift_cuda import _C
        print("    _C binary          =", _C.__file__)
        has_scalar = hasattr(_C, "accumulate_uz_scalar_forward")
        print("    scalar symbols     =", has_scalar)
        if not has_scalar:
            print("    >>> STALE BINARY: accumulate_uz_scalar was falling back"
                  " to the .item()-per-shift reference. Rebuild:\n"
                  "        pip install --no-build-isolation --force-reinstall .")
            return False
    else:
        print("    >>> _C FAILED TO IMPORT: every op was falling back to the"
              " reference (with per-shift .item() syncs on device tensors).")
        return False
    return True


def one_step(model, x, sig):
    u, _ = model(x, sig=sig)
    loss = u.sum()
    loss.backward()
    x.grad = None
    for p in model.parameters():
        p.grad = None


def timed(model, x, sig, n=10, warmup=3):
    for _ in range(warmup):
        one_step(model, x, sig)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        one_step(model, x, sig)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1e3


def count_syncs(model, x, sig):
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        one_step(model, x, sig)
    sync_evts = [e for e in prof.key_averages()
                 if "Synchronize" in e.key or "aten::item" in e.key
                 or "aten::_local_scalar_dense" in e.key]
    total = sum(e.count for e in sync_evts)
    for e in sync_evts:
        print(f"      {e.key:45s} count={e.count:5d} "
              f"cpu_total={e.cpu_time_total/1e3:8.2f} ms")
    return total, prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-file", default="GASD_drunet_attn.py")
    ap.add_argument("--patch-file", default=None,
                    help="path to gasd_drunet_attn_patch.py; default: installed package")
    ap.add_argument("--B", type=int, default=8)
    ap.add_argument("--H", type=int, default=96)
    args = ap.parse_args()

    ok = check_extension()

    GASD = load_model_class(args.model_file)
    model = GASD(
        in_channels=3, window_rad=5, feat_ch=128,
        drunet_nc=[64, 128, 256], drunet_nb=2, noise_map=True,
        qk_ch=32, n_heads=4, init_tau=1.0, activation="ReLU",
        max_batch_shifts=43, use_grad_checkpoint=True,
        transform_family="translations+d4",
        output_activation="control_sigmoid", tau_mode="fixed_sigma2",
        head_mix="concat", use_dist=False, layer_norm=True,
    ).cuda().train()

    if args.patch_file:
        spec = importlib.util.spec_from_file_location("gasd_patch", args.patch_file)
        patch = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(patch)
        patch.install_cuda_shift(type(model))
    else:
        from neural_shift_cuda.integration import install_cuda_shift_gasd
        install_cuda_shift_gasd(type(model))

    x = torch.randn(args.B, 3, args.H, args.H, device="cuda", requires_grad=True)
    sig = torch.full((args.B, 1, 1, 1), 25 / 255, device="cuda")

    print("=" * 70)
    print("[2] sync stalls per fwd+bwd step")
    for use in (False, True):
        model.use_cuda_shift = use
        one_step(model, x, sig)  # graph warmup
        print(f"    use_cuda_shift={use}:")
        total, _ = count_syncs(model, x, sig)
        print(f"      -> total sync-ish events: {total}")
        if use and total > 20:
            print("      >>> the 'CUDA' path is sync-bound: it is NOT running"
                  " the compiled kernels end to end.")

    print("=" * 70)
    print("[3] wall clock (fwd+bwd, mean of 10)")
    for use in (False, True):
        model.use_cuda_shift = use
        ms = timed(model, x, sig)
        print(f"    use_cuda_shift={use}: {ms:8.1f} ms/step")

    print("=" * 70)
    print("[4] top CUDA kernels, use_cuda_shift=True")
    model.use_cuda_shift = True
    _, prof = count_syncs(model, x, sig)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))

    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
