"""Benchmark the serial shift loop against the CUDA tree reductions.

Run after installing the extension in editable mode:

    python tests/benchmark_parallel_reduction.py --B 4 --H 128 --W 128 --R 5
"""

from __future__ import annotations

import argparse
import statistics

import torch

from neural_shift_cuda import accumulate_uz, normalized_accumulate_uz


def _periodic_shifts(radius: int):
    rows = [
        (dx, dy, 0)
        for dx in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
    ]
    return rows, torch.tensor(rows, dtype=torch.int32, device="cuda")


def _serial_normalized(x, weights, rows):
    S = len(rows)
    B, C, H, W = x.shape
    w = weights.reshape(S, B, C, H, W)
    numerator = torch.zeros_like(x, dtype=torch.float32)
    degree = torch.zeros_like(x, dtype=torch.float32)
    x32 = x.float()
    w32 = w.float()
    for s, (dx, dy, _) in enumerate(rows):
        shifted = torch.roll(x32, shifts=(-dx, -dy), dims=(-2, -1))
        numerator = numerator + w32[s] * shifted
        degree = degree + w32[s]
    return numerator / degree


def _median_ms(fn, warmup: int, iterations: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        stop = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        stop.record()
        stop.synchronize()
        samples.append(start.elapsed_time(stop))
    return statistics.median(samples)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--B", type=int, default=4)
    parser.add_argument("--C", type=int, default=3)
    parser.add_argument("--H", type=int, default=128)
    parser.add_argument("--W", type=int, default=128)
    parser.add_argument("--R", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")

    rows, shifts = _periodic_shifts(args.R)
    S = len(rows)
    if S > 1024:
        raise SystemExit(f"S={S} exceeds the CUDA tree limit of 1024")
    x = torch.rand(args.B, args.C, args.H, args.W, device="cuda")
    weights = torch.rand(
        S * args.B, args.C, args.H, args.W, device="cuda") + 0.01

    serial = lambda: _serial_normalized(x, weights, rows)

    def exact_tree():
        numerator, degree = accumulate_uz(x, weights, shifts)
        return numerator / degree

    stable_tree = lambda: normalized_accumulate_uz(
        x, weights, shifts, validate=False)

    d_serial = serial()
    d_exact = exact_tree()
    d_stable = stable_tree()
    torch.testing.assert_close(d_exact, d_serial, rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(d_stable, d_serial, rtol=2e-5, atol=2e-5)

    serial_ms = _median_ms(serial, args.warmup, args.iterations)
    exact_ms = _median_ms(exact_tree, args.warmup, args.iterations)
    stable_ms = _median_ms(stable_tree, args.warmup, args.iterations)
    print(f"B={args.B} C={args.C} H={args.H} W={args.W} S={S}")
    print(f"serial PyTorch loop : {serial_ms:9.3f} ms")
    print(f"exact CUDA tree     : {exact_ms:9.3f} ms  ({serial_ms/exact_ms:6.2f}x)")
    print(f"stable CUDA tree    : {stable_ms:9.3f} ms  ({serial_ms/stable_ms:6.2f}x)")

    huge = torch.full_like(weights, 1.0e38)
    _, overflowing_degree = accumulate_uz(x, huge, shifts)
    stable = normalized_accumulate_uz(x, huge, shifts, validate=False)
    print("overflow stress     :",
          f"exact degree finite={bool(torch.isfinite(overflowing_degree).all())},",
          f"stable D finite={bool(torch.isfinite(stable).all())}")


if __name__ == "__main__":
    main()
