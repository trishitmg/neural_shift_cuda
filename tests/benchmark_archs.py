# tests/benchmark_archs.py
#
# Serial (reference PyTorch) vs parallel (neural_shift_cuda) timing for the four
# supported archs, driven by a config file. Reports forward and backprop times.
#
#   python tests/benchmark_archs.py --config tests/configs/nekde.yaml
#   python tests/benchmark_archs.py --config tests/configs/gasd.json --H 256 --W 256
#
# Serial vs parallel is toggled via each patch's `model.use_cuda_shift` flag on a
# single model instance, so both paths run identical weights on identical input.
# Requires CUDA and the extension installed (pip install -v -e .).

from __future__ import annotations

import argparse
import importlib.util
import inspect
import json
import os
import statistics as stats
import sys
import types
from pathlib import Path

import torch

# arch -> (installer name in neural_shift_cuda.integration, default class name,
#          extra forward kwargs). The constructor calling convention is detected
#          per class (see _construct), not hard-coded, because nekre exists in
#          both a **kwargs flavor and an older (args, device=...) flavor.
_ARCHS = {
    "nekde":  ("install_cuda_shift_attn",       "NeKDeDRUNetAttn",           {}),
    "gasd":   ("install_cuda_shift_gasd",       "GASDDRUNetAttn",            {}),
    "nkd_mp": ("install_cuda_shift_metropolis", "NeKDeMetropolisDRUNetAttn", {}),
    "neckre":  ("install_cuda_shift_nekre", "nekre",  {"batched": True}),
    "nectr":  ("install_cuda_shift_nectr", "nectr",  {}),
}


# ---------------------------------------------------------------------------
# Config loading (YAML or JSON)
# ---------------------------------------------------------------------------

def _load_config(path: str) -> dict:
    text = Path(path).read_text()
    if path.endswith((".yaml", ".yml")):
        try:
            import yaml
        except ImportError as e:
            raise SystemExit(
                "PyYAML is needed for YAML configs; use a .json config or "
                "`pip install pyyaml`.") from e
        return yaml.safe_load(text)
    return json.loads(text)


def _load_model_module(model_path: str):
    model_path = os.path.abspath(model_path)
    # Make the model file's siblings importable (drunet, spline_module, ...).
    parent = os.path.dirname(model_path)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location("_arch_under_test", model_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_arch_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


def _construct(cls, init_kwargs: dict, device: str):
    """Build the model regardless of constructor convention.

    Handles three shapes seen across these codebases:
      * ``__init__(self, **kwargs)`` or named kwargs   -> pass init_kwargs
      * ``__init__(self, args, device=...)``           -> wrap in a Namespace
      * either of the above additionally taking device -> device forwarded
    Detection is by signature so a config never has to declare the style.
    """
    params = list(inspect.signature(cls.__init__).parameters.values())[1:]  # drop self
    by_name = {p.name: p for p in params}
    has_var_kw = any(p.kind is p.VAR_KEYWORD for p in params)
    accepts_device = "device" in by_name
    # kwargs style if the class absorbs **kwargs or names any of our keys.
    kwargs_style = has_var_kw or any(k in by_name for k in init_kwargs)

    if kwargs_style:
        kw = dict(init_kwargs)
        if accepts_device:
            kw.setdefault("device", device)
        return cls(**kw), "kwargs"

    # Namespace style: single positional `args` object.
    args = argparse.Namespace(**init_kwargs)
    if accepts_device:
        return cls(args, device=device), "namespace(args, device)"
    return cls(args), "namespace(args)"


def _build_model(cfg: dict, device: str, dtype: torch.dtype):
    arch = cfg["arch"]
    if arch not in _ARCHS:
        raise SystemExit(f"unknown arch '{arch}'; expected one of {list(_ARCHS)}")
    _, default_cls, _ = _ARCHS[arch]

    mod = _load_model_module(cfg["model_path"])
    cls = getattr(mod, cfg.get("class_name", default_cls))
    init_kwargs = dict(cfg.get("init_kwargs", {}))

    model, style = _construct(cls, init_kwargs, device)
    print(f"  built {cls.__name__} via {style} constructor")
    return model.to(device).to(dtype)


def _install_patch(cfg: dict, model):
    installer_name = _ARCHS[cfg["arch"]][0]
    from neural_shift_cuda import integration
    getattr(integration, installer_name)(type(model))


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _ev():
    return torch.cuda.Event(enable_timing=True)


def _scalar_loss(out):
    if isinstance(out, (tuple, list)):
        ts = [o for o in out if torch.is_tensor(o) and o.is_floating_point()]
    elif torch.is_tensor(out):
        ts = [out]
    else:
        ts = []
    if not ts:
        raise RuntimeError("forward produced no floating tensor to backprop")
    return sum(t.float().square().mean() for t in ts)


def _summ(ms):
    return min(ms), stats.median(ms), max(ms)


def bench_infer(model, x, call_kwargs, warmup, iters):
    model.eval()
    ts = []
    with torch.no_grad():
        for _ in range(warmup):
            model(x, **call_kwargs)
        torch.cuda.synchronize()
        for _ in range(iters):
            e0, e1 = _ev(), _ev()
            e0.record(); model(x, **call_kwargs); e1.record()
            torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1))
    return _summ(ts)


def bench_train(model, x, call_kwargs, warmup, iters):
    """Time the (grad-enabled) forward and the backward separately."""
    model.train()
    f_ts, b_ts = [], []
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        _scalar_loss(model(x, **call_kwargs)).backward()
    torch.cuda.synchronize()
    for _ in range(iters):
        model.zero_grad(set_to_none=True)
        ef0, ef1 = _ev(), _ev()
        ef0.record(); out = model(x, **call_kwargs); ef1.record()
        torch.cuda.synchronize()
        f_ts.append(ef0.elapsed_time(ef1))

        loss = _scalar_loss(out)
        eb0, eb1 = _ev(), _ev()
        eb0.record(); loss.backward(); eb1.record()
        torch.cuda.synchronize()
        b_ts.append(eb0.elapsed_time(eb1))
    return _summ(f_ts), _summ(b_ts)


def peak_train_mem_mb(model, x, call_kwargs):
    model.train()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model.zero_grad(set_to_none=True)
    _scalar_loss(model(x, **call_kwargs)).backward()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / 1024 / 1024


def _run_variant(model, x, call_kwargs, use_cuda, warmup, iters, do_backward):
    # IMPORTANT: several patches lazily initialize on the FIRST forward and,
    # because their guard is on the cache attribute rather than the flag,
    # unconditionally set use_cuda_shift=True there -- silently clobbering a
    # pre-set False and making the "serial" variant run the CUDA path
    # (identical times, 1.00x speedup). Run one throwaway priming forward so
    # the lazy-init has already fired, THEN set the flag.
    model.eval()
    with torch.no_grad():
        model(x, **call_kwargs)
    torch.cuda.synchronize()

    model.use_cuda_shift = use_cuda
    if hasattr(model, "use_cuda_pair_gather"):      # nekre only
        model.use_cuda_pair_gather = use_cuda

    res = {"infer": bench_infer(model, x, call_kwargs, warmup, iters)}
    if do_backward:
        res["fwd"], res["bwd"] = bench_train(model, x, call_kwargs, warmup, iters)
        res["mem"] = peak_train_mem_mb(model, x, call_kwargs)

    # Catch any clobber that happened during timing: if this fires, the numbers
    # above did not measure the path they claim to.
    if bool(getattr(model, "use_cuda_shift")) != use_cuda:
        raise RuntimeError(
            f"use_cuda_shift was overwritten during benching "
            f"(wanted {use_cuda}, found {model.use_cuda_shift}); "
            f"results are invalid for this variant.")
    return res


# ---------------------------------------------------------------------------

def _fmt_row(name, serial, parallel):
    sp = serial / parallel if parallel > 0 else float("nan")
    return f"  {name:<20} {serial:10.3f}   {parallel:10.3f}   {sp:6.2f}x"


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this benchmark.")

    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="YAML or JSON arch config")
    p.add_argument("--B", type=int, default=None)
    p.add_argument("--H", type=int, default=None)
    p.add_argument("--W", type=int, default=None)
    p.add_argument("--dtype", choices=["float32", "float64"], default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--no-backward", action="store_true",
                   help="skip forward(grad)+backward timing (inference only)")
    a = p.parse_args()

    cfg = _load_config(a.config)
    inp = dict(cfg.get("input", {}))
    ben = dict(cfg.get("bench", {}))

    B = a.B or inp.get("B", 2)
    C = inp.get("C", cfg.get("init_kwargs", {}).get("in_channels", 1))
    H = a.H or inp.get("H", 128)
    W = a.W or inp.get("W", 128)
    sigma = inp.get("sigma", None)
    dtype = torch.float64 if (a.dtype or ben.get("dtype")) == "float64" else torch.float32
    warmup = a.warmup if a.warmup is not None else ben.get("warmup", 5)
    iters = a.iters if a.iters is not None else ben.get("iters", 20)
    do_backward = not a.no_backward and ben.get("backward", True)

    device = "cuda"
    model = _build_model(cfg, device, dtype)
    _install_patch(cfg, model)

    torch.manual_seed(ben.get("seed", 0))
    x = torch.randn(B, C, H, W, device=device, dtype=dtype)

    call_kwargs = dict(_ARCHS[cfg["arch"]][2])  # e.g. batched=True for nectr
    if sigma is not None:
        call_kwargs["sig"] = torch.full((B, 1, 1, 1), float(sigma),
                                        device=device, dtype=dtype)

    print(f"\narch={cfg['arch']}  class={type(model).__name__}  "
          f"B={B} C={C} H={H} W={W}  "
          f"window_rad={cfg.get('init_kwargs', {}).get('window_rad', '?')}  "
          f"dtype={str(dtype).split('.')[-1]}  warmup={warmup} iters={iters}")

    serial = _run_variant(model, x, call_kwargs, False, warmup, iters, do_backward)
    parallel = _run_variant(model, x, call_kwargs, True, warmup, iters, do_backward)

    print(f"  {'(median ms)':<20} {'serial':>10}   {'parallel':>10}   speedup")
    print(_fmt_row("forward (no_grad)", serial["infer"][1], parallel["infer"][1]))
    if do_backward:
        print(_fmt_row("forward (grad)", serial["fwd"][1], parallel["fwd"][1]))
        print(_fmt_row("backward", serial["bwd"][1], parallel["bwd"][1]))
        print(_fmt_row("fwd+bwd",
                       serial["fwd"][1] + serial["bwd"][1],
                       parallel["fwd"][1] + parallel["bwd"][1]))
        print(f"  {'peak mem (MB)':<20} {serial['mem']:10.1f}   "
              f"{parallel['mem']:10.1f}   "
              f"{parallel['mem'] / serial['mem']:6.2f}x")


if __name__ == "__main__":
    main()