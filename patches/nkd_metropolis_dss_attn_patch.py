"""CUDA patch for the doubly sub-stochastic NKD (NKD_mp_dss_* model files).

The model builds, per guide, J = 2R^2 + 2R + 1 head-mixed maps on the
half-plane G+ = {id} u I+ and turns them into the kernel K (`kernel_mode`):
  'symmetric'   K = D_id + sum_{pi in I+} (D_pi P_pi + P_{pi^{-1}} D_pi)
  'asymmetric'  K = D_id + sum_{pi in I+} D_pi (P_pi + P_{pi^{-1}})
then normalises it with r = K e, c = K^T e:
    W_{i,i+d} = K_{i,i+d} / max(r_i, (P_d c)_i),
and, with `diag_correction`, resets W_ii = 1 - max(R_i, C_i) (off-diagonal
row / column sums of W).

This patch replaces the per-shift Python loops by the package's fused,
differentiable primitives; the attention head is still the model's own
`_halfplane_weights`:
  * symmetric -- K = K^T, so c = r and W = W^T. Everything stays on the J
    half-plane rows with has_inverse=1: `accumulate_uz` synthesises the box-
    masked mirror edge P_{pi^{-1}} (w) exactly as the model does, r and
    W_hat e come out as its Z output, and max(r_i, r_{i+d}) is one
    `shift_gather`. W^T y = W y.
  * asymmetric -- the mirror edge keeps the centre-pixel weight, which is not
    what has_inverse=1 synthesises, so the full (2R+1)^2 edge stack is built
    (with shift_mode='legacy' the head already scores all (2R+1)^2 shifts and
    the stack is just the masked head maps)
    (has_inverse=0 rows). K^T e and C are one batched roll (gather) + sum over
    the stack; W^T y is `accumulate_uz` with negated shifts on the rolled stack
    (pi^{-1}(w (.) y) = pi^{-1}(w) (.) pi^{-1}(y)).
In both, the diagonal of W is folded into the identity row, so one
`accumulate_uz` applies W and its Z output is W e.

Patched: forward, adjoint, laplacian_grw, forward_cached, adjoint_cached.
DSG_NLM, laplacian_{un,rw,norm} and the *_cached Laplacians route through
these. Training wraps the operator build + apply in gradient checkpointing
(as `metropolis_aggregate` does) so the (S*B,C,H,W) edge stacks are recomputed
in backward instead of retained. The DSS route fixes comp_box=True, so edges
leaving the image are always masked.

Usage
-----
    from NKD_mp_dss_drunet_attn_v2 import NeKDeDSSDRUNetAttn
    from neural_shift_cuda.integration import install_cuda_shift_mp_dss
    install_cuda_shift_mp_dss(NeKDeDSSDRUNetAttn)

Set ``model.use_cuda_shift = False`` to force the reference PyTorch path.
"""

from __future__ import annotations

import inspect
from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from neural_shift_cuda import accumulate_uz, shift_gather


# ---------------------------------------------------------------------------
# Static per-(shape, device, dtype) tables: masks, shift rows, roll indices.
# ---------------------------------------------------------------------------

def _static(self, H: int, W: int, device: torch.device, dtype: torch.dtype) -> dict:
    pair = getattr(self, "shift_mode", "inverse_pair") != "legacy"
    key = (H, W, device, dtype, self.window_rad, self.kernel_mode, pair)
    state = getattr(self, "_dss_static", None)
    if state is not None and state[0] == key:
        return state[1]

    R = self.window_rad
    scored = [(int(dx), int(dy)) for dx, dy in self._scored_shifts()]
    full = [(int(dx), int(dy)) for dx, dy in self._collect_shifts()]
    box = F.pad(torch.ones(1, 1, H, W, device=device, dtype=dtype), (R, R, R, R))

    def masks(shifts):  # (S, 1, 1, H, W) validity of i + d
        return torch.stack([box[:, :, R + dx:R + dx + H, R + dy:R + dy + W]
                            for dx, dy in shifts], dim=0)

    def rows(shifts, flags):
        return torch.tensor([(dx, dy, f) for (dx, dy), f in zip(shifts, flags)],
                            dtype=torch.int64, device=device)

    nonid = [k for k, s in enumerate(scored) if s != (0, 0)]
    st = dict(
        pair=pair,
        i0=scored.index((0, 0)),
        mask_fwd=masks(scored),
        # symmetric: half-plane rows, mirror edge synthesised by accumulate_uz
        rows_half=rows(scored, [int(s != (0, 0)) for s in scored]),
        xy_half=torch.tensor(scored, dtype=torch.int64, device=device),
    )
    if self.kernel_mode == "asymmetric":
        # full edge list in the model's _collect_shifts order (scored first,
        # then pi^{-1} for pi in I+; legacy: the scored full window only), so
        # the identity row index is the same.
        hh = torch.arange(H, device=device).view(1, H, 1)
        ww = torch.arange(W, device=device).view(1, 1, W)
        d = torch.tensor(full, dtype=torch.int64, device=device)
        S = d.size(0)
        src = ((hh - d[:, 0].view(S, 1, 1)) % H) * W + (ww - d[:, 1].view(S, 1, 1)) % W
        st.update(
            nonid=torch.tensor(nonid, dtype=torch.int64, device=device),
            mask_rev=masks([(-scored[k][0], -scored[k][1]) for k in nonid]) if pair else None,
            rows_full=rows(full, [0] * S),
            rows_full_T=rows([(-dx, -dy) for dx, dy in full], [0] * S),
            xy_full=d,
            roll=src.reshape(S, 1, 1, H * W),  # out = P_d^{-1} t, i.e. t(i - d)
        )
    self._dss_static = (key, st)
    return st


def _unshift(stack: torch.Tensor, roll: torch.Tensor) -> torch.Tensor:
    """Row s of (S, B, C, H, W) -> P_{d_s}^{-1} row s, in one gather."""
    S, B, C, H, W = stack.shape
    return stack.reshape(S, B, C, H * W).gather(
        3, roll.expand(S, B, C, H * W)).view(S, B, C, H, W)


def _mix_heads_vec(a: torch.Tensor, C: int, mix: Optional[torch.Tensor]) -> torch.Tensor:
    """(J, B, h, H, W) -> (J, B, C, H, W); mirrors the model's _mix_heads."""
    if mix is None:
        return a.expand(-1, -1, C, -1, -1)
    return torch.einsum("ch,sbhij->sbcij", mix, a).contiguous()


def _mixed_half(self, x: torch.Tensor, guide, sig) -> torch.Tensor:
    """Head-mixed J half-plane maps (J, B, C, H, W) from the model's own head."""
    z = x if guide is None else guide
    if z.shape != x.shape:
        raise ValueError(f"guide shape {tuple(z.shape)} must match x shape {tuple(x.shape)}.")
    if z.device != x.device or z.dtype != x.dtype:
        raise ValueError("guide and x must share device and dtype.")
    s = self._normalise_sigma(x, sig)
    phi = self.pre_activation(z, s)
    R = self.window_rad
    a = torch.stack(list(self._halfplane_weights(
        phi, F.pad(phi, (R, R, R, R), mode="circular"), s)), dim=0)
    return _mix_heads_vec(a, x.size(1), self._head_mix_matrix())


# ---------------------------------------------------------------------------
# W as an accumulate_uz operand: W v = accumulate_uz(v, w, rows)[0], and the
# Z output of the same call is W e.
# ---------------------------------------------------------------------------

def _operator(m: torch.Tensor, st: dict, sym: bool, corr: bool, eps: float):
    J, B, C, H, W = m.shape
    i0 = st["i0"]
    e = m * st["mask_fwd"]                                      # forward edges

    if sym:
        rows = st["rows_half"]
        ones = torch.ones(B, C, H, W, device=m.device, dtype=m.dtype)
        _, r = accumulate_uz(ones, e.reshape(-1, C, H, W), rows)  # r = K e = K^T e
        r = r.to(m.dtype)
        Pr, _ = shift_gather(r, st["xy_half"])                  # r_{i+d}
        w = e / torch.maximum(r.unsqueeze(0), Pr.view(J, B, C, H, W)).clamp_min(eps)
        if corr:
            # d_hat = W_hat e incl. the identity, so 1 - (d_hat - w_id) = 1 - R.
            _, d_hat = accumulate_uz(ones, w.reshape(-1, C, H, W), rows)
            w = torch.cat([w[:i0], (1.0 - (d_hat.to(m.dtype) - w[i0])).unsqueeze(0), w[i0 + 1:]])
        return w.reshape(-1, C, H, W), rows

    if st["pair"]:  # inverse_pair: pi^{-1} reuses pi's map; legacy: e is already full
        e = torch.cat([e, m.index_select(0, st["nonid"]) * st["mask_rev"]], dim=0)  # (S,...)
    S = e.size(0)
    r = e.sum(0)
    c = _unshift(e, st["roll"]).sum(0)                          # K^T e
    Pc, _ = shift_gather(c, st["xy_full"])                      # c_{i+d}
    w = e / torch.maximum(r.unsqueeze(0), Pc.view(S, B, C, H, W)).clamp_min(eps)
    if corr:
        # The identity row is unaffected by the roll, so subtracting it from the
        # full sums leaves the off-diagonal row / column sums R, C.
        R_ = w.sum(0) - w[i0]
        C_ = _unshift(w, st["roll"]).sum(0) - w[i0]
        w = torch.cat([w[:i0], (1.0 - torch.maximum(R_, C_)).unsqueeze(0), w[i0 + 1:]])
    return w.reshape(-1, C, H, W), st["rows_full"]


def _apply_T(v: torch.Tensor, w: torch.Tensor, rows: torch.Tensor, st: dict, sym: bool):
    """W^T v for the operand returned by _operator."""
    if sym:
        return accumulate_uz(v, w, rows)[0]
    S = rows.size(0)
    w_t = _unshift(w.view(S, -1, *w.shape[1:]), st["roll"]).reshape(w.shape).contiguous()
    return accumulate_uz(v, w_t, st["rows_full_T"])[0]


def _run(self, fn, m: torch.Tensor, x: torch.Tensor):
    """Checkpoint the operator build + apply while training (weights recomputed
    in backward); plain call otherwise."""
    if torch.is_grad_enabled() and (m.requires_grad or x.requires_grad):
        return checkpoint(fn, m, x, use_reentrant=False)
    return fn(m, x)


def _eps(self, dtype):
    return max(self.metropolis_eps, torch.finfo(dtype).tiny)


# ---------------------------------------------------------------------------
# Non-cached operators.
# ---------------------------------------------------------------------------

def _forward_cuda(self, x, guide=None, sig=None, return_D: bool = False):
    """(W x, W e if return_D)."""
    m = _mixed_half(self, x, guide, sig)
    st = _static(self, x.size(2), x.size(3), x.device, m.dtype)
    sym, corr, eps = self.kernel_mode == "symmetric", bool(self.diag_correction), _eps(self, m.dtype)

    def fn(m_, x_):
        w, rows = _operator(m_, st, sym, corr, eps)
        return accumulate_uz(x_, w, rows)

    Wx, We = _run(self, fn, m, x.contiguous())
    return Wx, (We if return_D else None)


def _adjoint_cuda(self, y, guide=None, sig=None):
    """W^T y."""
    m = _mixed_half(self, y, guide, sig)
    st = _static(self, y.size(2), y.size(3), y.device, m.dtype)
    sym, corr, eps = self.kernel_mode == "symmetric", bool(self.diag_correction), _eps(self, m.dtype)

    def fn(m_, y_):
        w, rows = _operator(m_, st, sym, corr, eps)
        return _apply_T(y_, w, rows, st, sym)

    return _run(self, fn, m, y.contiguous())


def _laplacian_grw_cuda(self, x, guide=None, sig=None, eps=1e-10):
    """(I - W)^T (I - W) x from one network pass and one operator build."""
    m = _mixed_half(self, x, guide, sig)
    st = _static(self, x.size(2), x.size(3), x.device, m.dtype)
    sym, corr, e = self.kernel_mode == "symmetric", bool(self.diag_correction), _eps(self, m.dtype)

    def fn(m_, x_):
        w, rows = _operator(m_, st, sym, corr, e)
        z = x_ - accumulate_uz(x_, w, rows)[0]
        return z - _apply_T(z.contiguous(), w, rows, st, sym)

    return _run(self, fn, m, x.contiguous())


# ---------------------------------------------------------------------------
# Fixed-guide cache: pack the model's DSSWeightCache (diag + off-diagonal
# edges, full shift list in both modes) once, reuse for every CG matvec.
# ---------------------------------------------------------------------------

def _cached_pack(self, x: torch.Tensor, cache, transpose: bool):
    if x.dim() != 4:
        raise ValueError(f"x must have shape (B,C,H,W), got {tuple(x.shape)}.")
    if x.size(1) != cache.C:
        raise ValueError(f"cache has C={cache.C} channels but x has {x.size(1)}.")
    key = (tuple(x.shape), x.device, x.dtype, transpose)
    state = getattr(self, "_dss_cached_pack", None)
    if state is not None and state[0] is cache and state[1] == key:
        return state[2], state[3]

    shifts = [(0, 0)] + [(int(dx), int(dy)) for dx, dy in cache.shifts]
    weights = [cache.diag] + list(cache.w_hat_list)
    for w in weights:
        if tuple(w.shape) != tuple(x.shape) or w.device != x.device or w.dtype != x.dtype:
            raise ValueError("cached weights and x must share shape, device and dtype.")
    stack = torch.stack(weights, dim=0)                         # (S, B, C, H, W)
    S, (B, C, H, W) = len(shifts), x.shape
    if transpose:
        d = torch.tensor(shifts, dtype=torch.int64, device=x.device)
        hh = torch.arange(H, device=x.device).view(1, H, 1)
        ww = torch.arange(W, device=x.device).view(1, 1, W)
        roll = (((hh - d[:, 0].view(S, 1, 1)) % H) * W
                + (ww - d[:, 1].view(S, 1, 1)) % W).reshape(S, 1, 1, H * W)
        stack = _unshift(stack, roll)
        shifts = [(-dx, -dy) for dx, dy in shifts]
    w_all = stack.reshape(S * B, C, H, W).contiguous()
    rows = torch.tensor([(dx, dy, 0) for dx, dy in shifts], dtype=torch.int64, device=x.device)
    self._dss_cached_pack = (cache, key, w_all, rows)
    return w_all, rows


def _forward_cached_cuda(self, x, cache, return_D: bool = False):
    w_all, rows = _cached_pack(self, x, cache, transpose=False)
    Wx, We = accumulate_uz(x.contiguous(), w_all, rows)
    return Wx, (We if return_D else None)


def _adjoint_cached_cuda(self, y, cache):
    w_t, rows = _cached_pack(self, y, cache, transpose=True)
    return accumulate_uz(y.contiguous(), w_t, rows)[0]


# ---------------------------------------------------------------------------
# Public installer
# ---------------------------------------------------------------------------

def install_cuda_shift(model_cls):
    """Patch a NKD_mp_dss model class. Idempotent; CPU tensors and
    ``use_cuda_shift = False`` keep the original PyTorch methods."""
    if getattr(model_cls, "_dss_cuda_installed", False):
        return model_cls
    for name in ("forward", "adjoint", "laplacian_grw", "_scored_shifts",
                 "_collect_shifts", "_halfplane_weights"):
        if not hasattr(model_cls, name):
            raise TypeError(f"{model_cls.__name__} has no {name}; not a NKD_mp_dss model.")

    def _on(self, t):
        if not hasattr(self, "use_cuda_shift"):
            self.use_cuda_shift = True
        return self.use_cuda_shift and t.is_cuda

    orig_forward = model_cls.forward
    orig_adjoint = model_cls.adjoint
    orig_grw = model_cls.laplacian_grw
    rd = inspect.signature(orig_forward).parameters["return_D"].default

    def forward(self, x, guide=None, sig=None, return_D=rd):
        if _on(self, x):
            return _forward_cuda(self, x, guide, sig, return_D=return_D)
        return orig_forward(self, x, guide, sig=sig, return_D=return_D)

    def adjoint(self, y, guide=None, sig=None):
        if _on(self, y):
            return _adjoint_cuda(self, y, guide, sig)
        return orig_adjoint(self, y, guide, sig=sig)

    def laplacian_grw(self, x, guide=None, sig=None, eps=1e-10):
        if _on(self, x):
            return _laplacian_grw_cuda(self, x, guide, sig, eps=eps)
        return orig_grw(self, x, guide, sig=sig, eps=eps)

    model_cls.forward, model_cls.adjoint, model_cls.laplacian_grw = forward, adjoint, laplacian_grw

    orig_fc = getattr(model_cls, "forward_cached", None)
    orig_ac = getattr(model_cls, "adjoint_cached", None)
    if orig_fc is not None:
        rdc = inspect.signature(orig_fc).parameters["return_D"].default

        def forward_cached(self, x, cache, return_D=rdc):
            if _on(self, x):
                return _forward_cached_cuda(self, x, cache, return_D=return_D)
            return orig_fc(self, x, cache, return_D=return_D)

        model_cls.forward_cached = forward_cached
    if orig_ac is not None:
        def adjoint_cached(self, y, cache):
            if _on(self, y):
                return _adjoint_cached_cuda(self, y, cache)
            return orig_ac(self, y, cache)

        model_cls.adjoint_cached = adjoint_cached

    model_cls._dss_cuda_installed = True
    return model_cls
