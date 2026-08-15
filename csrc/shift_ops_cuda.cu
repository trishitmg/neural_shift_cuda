// shift_ops_cuda.cu
// CUDA kernels for shift_gather / pair_gather / accumulate_uz used by nekre.
//
// Layout assumptions (HARD):
//   - All input/output tensors are contiguous NCHW (or (S*B, C, H, W) folded).
//   - Stride for an (N, C, H, W) tensor is (C*H*W, H*W, W, 1).
//   - Tensor sizes, flattened offsets, and shifts use signed int64.  CUDA grid
//     dimensions remain int-sized, but every grid-stride expression is widened
//     before multiplication.
//
// Shifts:
//   - "shifts" is an int64 tensor (S, 2) or (S, 3); we always read 3 values per
//     row when accumulate kernels are used. shift_stride is supplied by host.
//   - The "has_inverse" flag (col 2) is used only by accumulate_uz kernels.
//   - We assume |dx| < H and |dy| < W, which always holds for R << min(H, W).
//     A single +H / -H correction is enough to bring an index back into
//     [0, H); same for W.
//
// Modulo:
//   - "Circular" semantics matches F.pad(..., mode='circular') + slice.
//   - "Mask" semantics matches F.pad(ones, ..., value=0) + slice:
//        mask(s, h, w) = 1 iff (0 <= h+dx < H) && (0 <= w+dy < W).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

namespace {

constexpr int THREADS = 256;

inline int n_blocks(int64_t n, int threads = THREADS) {
    // Cap blocks to a sane value; CUDA_KERNEL_LOOP handles overflow.
    if (n <= 0) return 0;
    // Avoid the otherwise-overflowing n + threads - 1 ceil-division form.
    const int64_t b = 1 + (n - 1) / threads;
    return static_cast<int>(b < 65535 ? b : 65535);
}

inline int reduction_threads(int64_t n) {
    // One lane owns one shift.  A power-of-two block makes the reduction a
    // strict binary tree with ceil(log2(S)) synchronization rounds.
    int t = 1;
    while (t < n && t < 1024) t <<= 1;
    return t;
}

#define CUDA_KERNEL_LOOP(i, n)                                            \
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +      \
                     threadIdx.x;                                         \
         i < (n);                                                         \
         i += static_cast<int64_t>(blockDim.x) * gridDim.x)

// ---------------------------------------------------------------------------
// Forward gather kernels
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void shift_gather_forward_kernel(
    const scalar_t* __restrict__ guide,   // (B, C, H, W)
    const int64_t*  __restrict__ shifts,  // (S, shift_stride)
    scalar_t*       __restrict__ out,     // (S*B, C, H, W)
    int64_t S, int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t shift_stride)
{
    const int64_t N = S * B * C * H * W;
    CUDA_KERNEL_LOOP(idx, N) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;          t /= H;
        int64_t c = t % C;          t /= C;
        int64_t b = t % B;
        int64_t s = t / B;

        const int64_t dx = shifts[s * shift_stride + 0];
        const int64_t dy = shifts[s * shift_stride + 1];

        int64_t sh = h + dx;
        if (sh < 0)        sh += H;
        else if (sh >= H)  sh -= H;
        int64_t sw = w + dy;
        if (sw < 0)        sw += W;
        else if (sw >= W)  sw -= W;

        const int64_t gidx = ((b * C + c) * H + sh) * W + sw;
        out[idx] = guide[gidx];
    }
}

template <typename scalar_t>
__global__ void pair_gather_forward_kernel(
    const scalar_t* __restrict__ guide,   // (B, C, H, W)
    const int64_t*  __restrict__ shifts,  // (S, shift_stride)
    scalar_t*       __restrict__ out,     // (S*B, 2C, H, W)
    int64_t S, int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t shift_stride)
{
    const int64_t Cout = 2 * C;
    const int64_t N = S * B * Cout * H * W;
    CUDA_KERNEL_LOOP(idx, N) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;            t /= H;
        int64_t cp = t % Cout;        t /= Cout;
        int64_t b = t % B;
        int64_t s = t / B;

        if (cp < C) {
            // center: guide[b, cp, h, w]
            const int64_t gidx = ((b * C + cp) * H + h) * W + w;
            out[idx] = guide[gidx];
        } else {
            const int64_t c = cp - C;
            const int64_t dx = shifts[s * shift_stride + 0];
            const int64_t dy = shifts[s * shift_stride + 1];

            int64_t sh = h + dx;
            if (sh < 0)        sh += H;
            else if (sh >= H)  sh -= H;
            int64_t sw = w + dy;
            if (sw < 0)        sw += W;
            else if (sw >= W)  sw -= W;

            const int64_t gidx = ((b * C + c) * H + sh) * W + sw;
            out[idx] = guide[gidx];
        }
    }
}

template <typename scalar_t>
__global__ void shift_mask_kernel(
    const int64_t* __restrict__ shifts,  // (S, shift_stride)
    scalar_t*  __restrict__ mask,    // (S*B, 1, H, W)
    int64_t S, int64_t B, int64_t H, int64_t W,
    int64_t shift_stride)
{
    const int64_t N = S * B * H * W;
    CUDA_KERNEL_LOOP(idx, N) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;            t /= H;
        // b dimension is collapsed (mask is the same for every b)
        int64_t sb = t;
        int64_t s = sb / B;

        const int64_t dx = shifts[s * shift_stride + 0];
        const int64_t dy = shifts[s * shift_stride + 1];

        const int64_t hh = h + dx;
        const int64_t ww = w + dy;
        const bool inb = (hh >= 0 && hh < H && ww >= 0 && ww < W);
        mask[idx] = inb ? scalar_t(1) : scalar_t(0);
    }
}

// ---------------------------------------------------------------------------
// Backward kernels for shift_gather / pair_gather
// (scatter-add into grad_guide, atomicAdd because many (s, h, w) map to the
//  same (src_h, src_w) in the source guide.)
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void shift_gather_backward_kernel(
    const scalar_t* __restrict__ grad_out,   // (S*B, C, H, W)
    const int64_t*  __restrict__ shifts,     // (S, shift_stride)
    scalar_t*       __restrict__ grad_guide, // (B, C, H, W)
    int64_t S, int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t shift_stride)
{
    // Gather formulation (atomic-free, deterministic): each circular shift is
    // a bijection, so guide[b,c,gh,gw] was read by exactly one output pixel
    // per shift s, namely out[s,b,c,h,w] with h=(gh-dx) mod H, w=(gw-dy) mod W.
    // One thread OWNS one grad_guide element and sums its S contributions.
    // This replaces the old scatter version whose S*B*C*H*W atomicAdds
    // contended S-to-1 on grad_guide and were nondeterministic.
    const int64_t N = B * C * H * W;
    CUDA_KERNEL_LOOP(idx, N) {
        int64_t gw = idx % W;
        int64_t t = idx / W;
        int64_t gh = t % H;           t /= H;
        int64_t c = t % C;
        int64_t b = t / C;

        scalar_t acc = scalar_t(0);
        for (int64_t s = 0; s < S; ++s) {
            const int64_t dx = shifts[s * shift_stride + 0];
            const int64_t dy = shifts[s * shift_stride + 1];

            int64_t h = gh - dx;                  // |dx| < H: one correction
            if (h < 0)        h += H;
            else if (h >= H)  h -= H;
            int64_t w = gw - dy;
            if (w < 0)        w += W;
            else if (w >= W)  w -= W;

            acc += grad_out[(((s * B + b) * C + c) * H + h) * W + w];
        }
        grad_guide[idx] = acc;
    }
}

template <typename scalar_t>
__global__ void pair_gather_backward_kernel(
    const scalar_t* __restrict__ grad_out,    // (S*B, 2C, H, W)
    const int64_t*  __restrict__ shifts,      // (S, shift_stride)
    scalar_t*       __restrict__ grad_guide,  // (B, C, H, W)
    int64_t S, int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t shift_stride)
{
    const int64_t Cout = 2 * C;
    const int64_t N = S * B * Cout * H * W;
    CUDA_KERNEL_LOOP(idx, N) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;            t /= H;
        int64_t cp = t % Cout;        t /= Cout;
        int64_t b = t % B;
        int64_t s = t / B;

        const scalar_t go = grad_out[idx];

        if (cp < C) {
            // center half: identity scatter into (b, cp, h, w).
            // Multiple s map here, so we need atomicAdd.
            const int64_t gidx = ((b * C + cp) * H + h) * W + w;
            atomicAdd(grad_guide + gidx, go);
        } else {
            const int64_t c = cp - C;
            const int64_t dx = shifts[s * shift_stride + 0];
            const int64_t dy = shifts[s * shift_stride + 1];

            int64_t sh = h + dx;
            if (sh < 0)        sh += H;
            else if (sh >= H)  sh -= H;
            int64_t sw = w + dy;
            if (sw < 0)        sw += W;
            else if (sw >= W)  sw -= W;

            const int64_t gidx = ((b * C + c) * H + sh) * W + sw;
            atomicAdd(grad_guide + gidx, go);
        }
    }
}

// ---------------------------------------------------------------------------
// accumulate_uz forward / backward
//   weights is (S*B, C, H, W) and ALREADY multiplied by the forward mask
//   (i.e. weight_fwd = weight_raw * comp_box), exactly as in the original
//   PyTorch code. So we don't reapply the forward mask here.
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void accumulate_uz_forward_tree_kernel(
    const scalar_t* __restrict__ x,        // (B, C, H, W)
    const scalar_t* __restrict__ weights,  // (S*B, C, H, W), pre-masked
    const int64_t*  __restrict__ shifts,   // (S, 3): (dx, dy, has_inv)
    scalar_t*       __restrict__ U,        // (B, C, H, W)
    scalar_t*       __restrict__ Z,        // (B, C, H, W)
    int64_t S, int64_t B, int64_t C, int64_t H, int64_t W)
{
    // One cooperative block per output element (grid-strided if N > 65535).
    // Lane s forms the contribution of shift s (forward + optional inverse),
    // then shared memory performs a pairwise binary-tree reduction.  This
    // removes the old O(S) dependency chain inside every CUDA thread.
    const int64_t N = B * C * H * W;
    extern __shared__ unsigned char smem_raw[];
    scalar_t* u_smem = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* z_smem = u_smem + blockDim.x;

    for (int64_t idx = blockIdx.x; idx < N; idx += gridDim.x) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;            t /= H;
        int64_t c = t % C;
        int64_t b = t / C;

        scalar_t u_val = scalar_t(0);
        scalar_t z_val = scalar_t(0);

        const int64_t x_base = (b * C + c) * H * W;

        const int s = threadIdx.x;
        if (s < S) {
            const int64_t dx = shifts[s * 3 + 0];
            const int64_t dy = shifts[s * 3 + 1];
            const int64_t has_inv = shifts[s * 3 + 2];

            // ---- forward direction ----
            // x is read at circular shift (h+dx) mod H, (w+dy) mod W.
            // weight is read at the natural (h, w). Mask is already folded in.
            int64_t sh = h + dx;
            if (sh < 0)        sh += H;
            else if (sh >= H)  sh -= H;
            int64_t sw = w + dy;
            if (sw < 0)        sw += W;
            else if (sw >= W)  sw -= W;

            const int64_t w_idx = (((s * B + b) * C + c) * H + h) * W + w;
            const int64_t x_idx_fwd = x_base + sh * W + sw;

            const scalar_t wv = weights[w_idx];
            u_val += wv * x[x_idx_fwd];
            z_val += wv;

            // ---- inverse direction (symmetry) ----
            if (has_inv) {
                // mask(-s, h, w) = 1 iff 0 <= h-dx < H && 0 <= w-dy < W.
                const int64_t hi = h - dx;
                const int64_t wi = w - dy;
                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                    const int64_t w_idx_inv = (((s * B + b) * C + c) * H + hi) * W + wi;
                    const int64_t x_idx_inv = x_base + hi * W + wi;
                    const scalar_t wv_inv = weights[w_idx_inv];
                    u_val += wv_inv * x[x_idx_inv];
                    z_val += wv_inv;
                }
            }
        }

        u_smem[threadIdx.x] = u_val;
        z_smem[threadIdx.x] = z_val;
        __syncthreads();

        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                u_smem[threadIdx.x] += u_smem[threadIdx.x + stride];
                z_smem[threadIdx.x] += z_smem[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            U[idx] = u_smem[0];
            Z[idx] = z_smem[0];
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void accumulate_uz_backward_x_tree_kernel(
    const scalar_t* __restrict__ weights,  // (S*B, C, H, W)
    const scalar_t* __restrict__ grad_U,   // (B, C, H, W)
    const int64_t*  __restrict__ shifts,   // (S, 3)
    scalar_t*       __restrict__ grad_x,   // (B, C, H, W)
    int64_t S, int64_t B, int64_t C, int64_t H, int64_t W)
{
    // grad_x receives no contribution from grad_Z (Z does not depend on x).
    const int64_t N = B * C * H * W;
    extern __shared__ unsigned char smem_raw[];
    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);

    for (int64_t idx = blockIdx.x; idx < N; idx += gridDim.x) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;            t /= H;
        int64_t c = t % C;
        int64_t b = t / C;

        scalar_t gx = scalar_t(0);

        const int s = threadIdx.x;
        if (s < S) {
            const int64_t dx = shifts[s * 3 + 0];
            const int64_t dy = shifts[s * 3 + 1];
            const int64_t has_inv = shifts[s * 3 + 2];

            // (A) forward case: the forward reads x circularly.  The caller's
            // comp_box multiplication, when enabled, zeros the corresponding
            // weight; the primitive itself must retain circular derivatives.
            int64_t h_src = h - dx;
            if (h_src < 0)       h_src += H;
            else if (h_src >= H) h_src -= H;
            int64_t w_src = w - dy;
            if (w_src < 0)       w_src += W;
            else if (w_src >= W) w_src -= W;
            const int64_t w_idx = (((s * B + b) * C + c) * H + h_src) * W + w_src;
            const int64_t gu_idx = ((b * C + c) * H + h_src) * W + w_src;
            gx += weights[w_idx] * grad_U[gu_idx];

            // (B) inverse case: x[b,c,h,w] is read from output position
            // (h_dst=h+dx, w_dst=w+dy) when has_inv and 0 <= h+dx < H, 0 <= w+dy < W.
            // In that branch weight[s,b,c,h,w] is the source weight.
            if (has_inv) {
                const int64_t h_dst = h + dx;
                const int64_t w_dst = w + dy;
                if (h_dst >= 0 && h_dst < H && w_dst >= 0 && w_dst < W) {
                    const int64_t w_idx = (((s * B + b) * C + c) * H + h) * W + w;
                    const int64_t gu_idx = ((b * C + c) * H + h_dst) * W + w_dst;
                    gx += weights[w_idx] * grad_U[gu_idx];
                }
            }
        }

        smem[threadIdx.x] = gx;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                smem[threadIdx.x] += smem[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            grad_x[idx] = smem[0];
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void accumulate_uz_backward_w_kernel(
    const scalar_t* __restrict__ x,        // (B, C, H, W)
    const scalar_t* __restrict__ grad_U,   // (B, C, H, W)
    const scalar_t* __restrict__ grad_Z,   // (B, C, H, W)
    const int64_t*  __restrict__ shifts,   // (S, 3)
    scalar_t*       __restrict__ grad_w,   // (S*B, C, H, W)
    int64_t S, int64_t B, int64_t C, int64_t H, int64_t W)
{
    const int64_t N = S * B * C * H * W;
    CUDA_KERNEL_LOOP(idx, N) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;            t /= H;
        int64_t c = t % C;            t /= C;
        int64_t b = t % B;
        int64_t s = t / B;

        const int64_t dx = shifts[s * 3 + 0];
        const int64_t dy = shifts[s * 3 + 1];
        const int64_t has_inv = shifts[s * 3 + 2];

        scalar_t gw = scalar_t(0);
        int64_t sh_circ = h + dx;
        if (sh_circ < 0)       sh_circ += H;
        else if (sh_circ >= H) sh_circ -= H;
        int64_t sw_circ = w + dy;
        if (sw_circ < 0)       sw_circ += W;
        else if (sw_circ >= W) sw_circ -= W;

        // Forward contribution is always circular.  If comp_box was used,
        // its multiply sits outside this op and masks this gradient upstream.
        const int64_t x_idx_fwd = ((b * C + c) * H + sh_circ) * W + sw_circ;
        const int64_t gu_idx_fwd = ((b * C + c) * H + h) * W + w;
        gw += x[x_idx_fwd] * grad_U[gu_idx_fwd] + grad_Z[gu_idx_fwd];

        // The synthesized inverse branch is deliberately non-periodic.
        const int64_t sh = h + dx;
        const int64_t sw = w + dy;
        if (has_inv && sh >= 0 && sh < H && sw >= 0 && sw < W) {
            const int64_t x_idx_self = ((b * C + c) * H + h) * W + w;
            const int64_t gu_idx_inv = ((b * C + c) * H + sh) * W + sw;
            gw += x[x_idx_self] * grad_U[gu_idx_inv] + grad_Z[gu_idx_inv];
        }

        grad_w[idx] = gw;
    }
}

// ---------------------------------------------------------------------------
// normalized_accumulate_uz forward / backward
//
// This path computes D = K/C directly and never materializes K or C in their
// potentially overflowing scale.  Each row is reduced in two parallel tree
// passes. For ordinary positive weights, m=max_i(w_i) and a_i=w_i/m. For
// log-weights, m=max_i(log w_i) and a_i=exp(log w_i-m). In both cases
// D=sum_i(a_i*x_i)/sum_i(a_i). The log-weight form should be used for an
// exponential head so the activation itself can never overflow. Zero positive
// weights and -inf log-weights are structural masks. Negative weights, NaNs,
// and +inf deliberately propagate NaN instead of silently producing a
// plausible but wrong normalization.
// ---------------------------------------------------------------------------

template <typename scalar_t>
__device__ inline scalar_t neg_inf() {
    return -std::numeric_limits<scalar_t>::infinity();
}

template <typename scalar_t>
__device__ inline scalar_t quiet_nan() {
    return std::numeric_limits<scalar_t>::quiet_NaN();
}

template <typename scalar_t>
__device__ inline scalar_t max_propagate_nan(scalar_t a, scalar_t b) {
    if (::isnan(a) || ::isnan(b)) return quiet_nan<scalar_t>();
    return a > b ? a : b;
}

template <typename scalar_t>
__device__ inline scalar_t as_log_weight(scalar_t value, bool log_weights) {
    if (log_weights) return value;
    if (value > scalar_t(0)) return ::log(value);
    if (value == scalar_t(0)) return neg_inf<scalar_t>();
    return quiet_nan<scalar_t>();
}

template <typename scalar_t>
__device__ inline scalar_t reduction_key(scalar_t value, bool log_weights) {
    if (log_weights) return value;
    if (value >= scalar_t(0)) return value;
    return quiet_nan<scalar_t>();
}

template <typename scalar_t>
__device__ inline scalar_t scaled_weight(
    scalar_t value, scalar_t row_max, bool log_weights)
{
    if (log_weights) {
        if (value == neg_inf<scalar_t>()) return scalar_t(0);
        return ::exp(value - row_max);
    }
    if (value == scalar_t(0)) return scalar_t(0);
    return value / row_max;
}

template <typename scalar_t>
__device__ inline scalar_t normalized_fraction(
    scalar_t value, scalar_t log_degree, bool log_weights)
{
    const scalar_t lw = as_log_weight(value, log_weights);
    if (lw == neg_inf<scalar_t>() || log_degree == neg_inf<scalar_t>())
        return scalar_t(0);
    return ::exp(lw - log_degree);
}

template <typename scalar_t>
__global__ void normalized_accumulate_uz_forward_tree_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weights,
    const int64_t*  __restrict__ shifts,
    scalar_t*       __restrict__ D,
    scalar_t*       __restrict__ log_C,
    int64_t S, int64_t B, int64_t C, int64_t H, int64_t W,
    bool log_weights)
{
    const int64_t N = B * C * H * W;
    extern __shared__ unsigned char smem_raw[];
    scalar_t* m_smem = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* u_smem = m_smem + blockDim.x;
    scalar_t* z_smem = u_smem + blockDim.x;

    for (int64_t idx = blockIdx.x; idx < N; idx += gridDim.x) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;            t /= H;
        int64_t c = t % C;
        int64_t b = t / C;
        const int64_t x_base = (b * C + c) * H * W;

        const int s = threadIdx.x;
        scalar_t local_m = neg_inf<scalar_t>();
        if (s < S) {
            const int64_t dx = shifts[s * 3 + 0];
            const int64_t dy = shifts[s * 3 + 1];
            const int64_t has_inv = shifts[s * 3 + 2];
            const int64_t w_idx = (((s * B + b) * C + c) * H + h) * W + w;
            local_m = reduction_key(weights[w_idx], log_weights);

            if (has_inv) {
                const int64_t hi = h - dx;
                const int64_t wi = w - dy;
                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                    const int64_t w_idx_inv =
                        (((s * B + b) * C + c) * H + hi) * W + wi;
                    local_m = max_propagate_nan(
                        local_m, reduction_key(weights[w_idx_inv], log_weights));
                }
            }
        }

        m_smem[threadIdx.x] = local_m;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                m_smem[threadIdx.x] = max_propagate_nan(
                    m_smem[threadIdx.x], m_smem[threadIdx.x + stride]);
            __syncthreads();
        }
        const scalar_t row_m = m_smem[0];

        const bool empty_row = log_weights
            ? (row_m == neg_inf<scalar_t>())
            : (row_m == scalar_t(0));
        scalar_t local_u = scalar_t(0);
        scalar_t local_z = scalar_t(0);
        if (s < S && !empty_row) {
            const int64_t dx = shifts[s * 3 + 0];
            const int64_t dy = shifts[s * 3 + 1];
            const int64_t has_inv = shifts[s * 3 + 2];

            int64_t sh = h + dx;
            if (sh < 0)       sh += H;
            else if (sh >= H) sh -= H;
            int64_t sw = w + dy;
            if (sw < 0)       sw += W;
            else if (sw >= W) sw -= W;

            const int64_t w_idx = (((s * B + b) * C + c) * H + h) * W + w;
            const scalar_t value = weights[w_idx];
            if ((log_weights && value != neg_inf<scalar_t>()) ||
                (!log_weights && value != scalar_t(0))) {
                const scalar_t a = scaled_weight(value, row_m, log_weights);
                local_u += a * x[x_base + sh * W + sw];
                local_z += a;
            }

            if (has_inv) {
                const int64_t hi = h - dx;
                const int64_t wi = w - dy;
                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                    const int64_t w_idx_inv =
                        (((s * B + b) * C + c) * H + hi) * W + wi;
                    const scalar_t value_inv = weights[w_idx_inv];
                    if ((log_weights && value_inv != neg_inf<scalar_t>()) ||
                        (!log_weights && value_inv != scalar_t(0))) {
                        const scalar_t a =
                            scaled_weight(value_inv, row_m, log_weights);
                        local_u += a * x[x_base + hi * W + wi];
                        local_z += a;
                    }
                }
            }
        }

        u_smem[threadIdx.x] = local_u;
        z_smem[threadIdx.x] = local_z;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                u_smem[threadIdx.x] += u_smem[threadIdx.x + stride];
                z_smem[threadIdx.x] += z_smem[threadIdx.x + stride];
            }
            __syncthreads();
        }

        if (threadIdx.x == 0) {
            if (empty_row && z_smem[0] == scalar_t(0)) {
                D[idx] = scalar_t(0);
                log_C[idx] = neg_inf<scalar_t>();
            } else if (::isfinite(row_m) && z_smem[0] > scalar_t(0) &&
                       ::isfinite(z_smem[0])) {
                D[idx] = u_smem[0] / z_smem[0];
                log_C[idx] = (log_weights ? row_m : ::log(row_m)) +
                             ::log(z_smem[0]);
            } else {
                D[idx] = quiet_nan<scalar_t>();
                log_C[idx] = quiet_nan<scalar_t>();
            }
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void normalized_accumulate_uz_backward_x_tree_kernel(
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ log_C,
    const scalar_t* __restrict__ grad_D,
    const int64_t*  __restrict__ shifts,
    scalar_t*       __restrict__ grad_x,
    int64_t S, int64_t B, int64_t C, int64_t H, int64_t W,
    bool log_weights)
{
    const int64_t N = B * C * H * W;
    extern __shared__ unsigned char smem_raw[];
    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);

    for (int64_t idx = blockIdx.x; idx < N; idx += gridDim.x) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;            t /= H;
        int64_t c = t % C;
        int64_t b = t / C;

        scalar_t gx = scalar_t(0);
        const int s = threadIdx.x;
        if (s < S) {
            const int64_t dx = shifts[s * 3 + 0];
            const int64_t dy = shifts[s * 3 + 1];
            const int64_t has_inv = shifts[s * 3 + 2];

            // Forward use of x[h,w] comes from the circular output h-dx,w-dy.
            int64_t oh = h - dx;
            if (oh < 0)       oh += H;
            else if (oh >= H) oh -= H;
            int64_t ow = w - dy;
            if (ow < 0)       ow += W;
            else if (ow >= W) ow -= W;
            const int64_t out_fwd = ((b * C + c) * H + oh) * W + ow;
            const int64_t w_idx_fwd =
                (((s * B + b) * C + c) * H + oh) * W + ow;
            gx += normalized_fraction(
                      weights[w_idx_fwd], log_C[out_fwd], log_weights) *
                  grad_D[out_fwd];

            // Synthesized inverse use of x[h,w] appears at h+dx,w+dy.
            if (has_inv) {
                const int64_t ih = h + dx;
                const int64_t iw = w + dy;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    const int64_t out_inv = ((b * C + c) * H + ih) * W + iw;
                    const int64_t w_idx_inv =
                        (((s * B + b) * C + c) * H + h) * W + w;
                    gx += normalized_fraction(
                              weights[w_idx_inv], log_C[out_inv], log_weights) *
                          grad_D[out_inv];
                }
            }
        }

        smem[threadIdx.x] = gx;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                smem[threadIdx.x] += smem[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0) grad_x[idx] = smem[0];
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void normalized_accumulate_uz_backward_w_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weights,
    const scalar_t* __restrict__ D,
    const scalar_t* __restrict__ log_C,
    const scalar_t* __restrict__ grad_D,
    const scalar_t* __restrict__ grad_log_C,
    const int64_t*  __restrict__ shifts,
    scalar_t*       __restrict__ grad_w,
    int64_t S, int64_t B, int64_t C, int64_t H, int64_t W,
    bool log_weights)
{
    const int64_t N = S * B * C * H * W;
    CUDA_KERNEL_LOOP(idx, N) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;            t /= H;
        int64_t c = t % C;            t /= C;
        int64_t b = t % B;
        int64_t s = t / B;

        const scalar_t value = weights[idx];
        const scalar_t lw = as_log_weight(value, log_weights);
        if (lw == neg_inf<scalar_t>()) {
            grad_w[idx] = scalar_t(0);
            continue;
        }

        const int64_t dx = shifts[s * 3 + 0];
        const int64_t dy = shifts[s * 3 + 1];
        const int64_t has_inv = shifts[s * 3 + 2];
        const int64_t base = (b * C + c) * H * W;

        int64_t sh_circ = h + dx;
        if (sh_circ < 0)       sh_circ += H;
        else if (sh_circ >= H) sh_circ -= H;
        int64_t sw_circ = w + dy;
        if (sw_circ < 0)       sw_circ += W;
        else if (sw_circ >= W) sw_circ -= W;

        const int64_t out_fwd = base + h * W + w;
        const scalar_t score_fwd =
            grad_D[out_fwd] * (x[base + sh_circ * W + sw_circ] - D[out_fwd]) +
            grad_log_C[out_fwd];
        scalar_t gw;
        if (log_weights) {
            gw = normalized_fraction(value, log_C[out_fwd], true) * score_fwd;
        } else {
            gw = ::exp(-log_C[out_fwd]) * score_fwd;
        }

        const int64_t ih = h + dx;
        const int64_t iw = w + dy;
        if (has_inv && ih >= 0 && ih < H && iw >= 0 && iw < W) {
            const int64_t out_inv = base + ih * W + iw;
            const scalar_t score_inv =
                grad_D[out_inv] * (x[base + h * W + w] - D[out_inv]) +
                grad_log_C[out_inv];
            if (log_weights) {
                gw += normalized_fraction(value, log_C[out_inv], true) * score_inv;
            } else {
                gw += ::exp(-log_C[out_inv]) * score_inv;
            }
        }
        grad_w[idx] = gw;
    }
}

// ---------------------------------------------------------------------------
// accumulate_uz_scalar forward / backward  (GASD scalar-per-transform)
//   weights is (S, B, C): ONE scalar per (transform, image, channel), constant
//   over space, so it is never materialized as (S*B, C, H, W). Gather
//   convention (matches shift_gather):
//     U[b,c,h,w] = sum_s weights[s,b,c] * x[b,c,(h+dx_s)%H,(w+dy_s)%W]
//     Z[b,c,h,w] = sum_s weights[s,b,c]
//   No boundary mask (circular) and no inverse-symmetry branch: GASD's
//   transforms are exact permutations with independent scalar weights.
//   All three kernels are atomic-free (each output element is owned by exactly
//   one thread), so results are deterministic.
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void accumulate_uz_scalar_forward_tree_kernel(
    const scalar_t* __restrict__ x,        // (B, C, H, W)
    const scalar_t* __restrict__ weights,  // (S, B, C)
    const int64_t*  __restrict__ shifts,   // (S, shift_stride)
    scalar_t*       __restrict__ U,        // (B, C, H, W)
    scalar_t*       __restrict__ Z,        // (B, C, H, W)
    int64_t S, int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t shift_stride)
{
    const int64_t N = B * C * H * W;
    extern __shared__ unsigned char smem_raw[];
    scalar_t* u_smem = reinterpret_cast<scalar_t*>(smem_raw);
    scalar_t* z_smem = u_smem + blockDim.x;

    for (int64_t idx = blockIdx.x; idx < N; idx += gridDim.x) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;            t /= H;
        int64_t c = t % C;
        int64_t b = t / C;

        const int64_t x_base = (b * C + c) * H * W;

        scalar_t u_val = scalar_t(0);
        scalar_t z_val = scalar_t(0);

        const int s = threadIdx.x;
        if (s < S) {
            const int64_t dx = shifts[s * shift_stride + 0];
            const int64_t dy = shifts[s * shift_stride + 1];

            int64_t sh = h + dx;
            if (sh < 0)        sh += H;
            else if (sh >= H)  sh -= H;
            int64_t sw = w + dy;
            if (sw < 0)        sw += W;
            else if (sw >= W)  sw -= W;

            const scalar_t wv = weights[(s * B + b) * C + c];
            u_val += wv * x[x_base + sh * W + sw];
            z_val += wv;
        }

        u_smem[threadIdx.x] = u_val;
        z_smem[threadIdx.x] = z_val;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride) {
                u_smem[threadIdx.x] += u_smem[threadIdx.x + stride];
                z_smem[threadIdx.x] += z_smem[threadIdx.x + stride];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            U[idx] = u_smem[0];
            Z[idx] = z_smem[0];
        }
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void accumulate_uz_scalar_backward_x_tree_kernel(
    const scalar_t* __restrict__ weights,  // (S, B, C)
    const scalar_t* __restrict__ grad_U,   // (B, C, H, W)
    const int64_t*  __restrict__ shifts,   // (S, shift_stride)
    scalar_t*       __restrict__ grad_x,   // (B, C, H, W)
    int64_t S, int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t shift_stride)
{
    // Z is independent of x, so grad_Z contributes nothing to grad_x.
    // x[b,c,h,w] is read by output position (h-dx, w-dy) (circular) for shift s.
    const int64_t N = B * C * H * W;
    extern __shared__ unsigned char smem_raw[];
    scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);

    for (int64_t idx = blockIdx.x; idx < N; idx += gridDim.x) {
        int64_t w = idx % W;
        int64_t t = idx / W;
        int64_t h = t % H;            t /= H;
        int64_t c = t % C;
        int64_t b = t / C;

        const int64_t gu_base = (b * C + c) * H * W;

        scalar_t gx = scalar_t(0);
        const int s = threadIdx.x;
        if (s < S) {
            const int64_t dx = shifts[s * shift_stride + 0];
            const int64_t dy = shifts[s * shift_stride + 1];

            int64_t oh = h - dx;
            if (oh < 0)        oh += H;
            else if (oh >= H)  oh -= H;
            int64_t ow = w - dy;
            if (ow < 0)        ow += W;
            else if (ow >= W)  ow -= W;

            const scalar_t wv = weights[(s * B + b) * C + c];
            gx += wv * grad_U[gu_base + oh * W + ow];
        }

        smem[threadIdx.x] = gx;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                smem[threadIdx.x] += smem[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            grad_x[idx] = smem[0];
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void accumulate_uz_scalar_backward_w_kernel(
    const scalar_t* __restrict__ x,        // (B, C, H, W)
    const scalar_t* __restrict__ grad_U,   // (B, C, H, W)
    const scalar_t* __restrict__ grad_Z,   // (B, C, H, W)
    const int64_t*  __restrict__ shifts,   // (S, shift_stride)
    scalar_t*       __restrict__ grad_w,   // (S, B, C)
    int64_t S, int64_t B, int64_t C, int64_t H, int64_t W,
    int64_t shift_stride)
{
    // One BLOCK per (s, b, c); the block's threads cooperatively reduce the
    // H*W grid via shared memory. This keeps occupancy high (S*B*C blocks *
    // blockDim.x threads) instead of the old one-thread-per-(s,b,c) version,
    // whose S*B*C threads each ran a serial H*W loop and left the GPU idle.
    // Deterministic (tree reduction, no atomics). blockDim.x is a power of two.
    //   grad_w[s,b,c] = sum_{h,w} grad_U[b,c,h,w] * x[b,c,(h+dx)%H,(w+dy)%W]
    //                 + sum_{h,w} grad_Z[b,c,h,w]
    const int64_t total_sbc = S * B * C;
    for (int64_t sbc = blockIdx.x; sbc < total_sbc; sbc += gridDim.x) {
        const int64_t c = sbc % C;
        int64_t t = sbc / C;
        const int64_t b = t % B;
        const int64_t s = t / B;

        const int64_t dx = shifts[s * shift_stride + 0];
        const int64_t dy = shifts[s * shift_stride + 1];
        const int64_t base = (b * C + c) * H * W;
        const int64_t HW = H * W;

        scalar_t local = scalar_t(0);
        for (int64_t p = threadIdx.x; p < HW; p += blockDim.x) {
            const int64_t h = p / W;
            const int64_t w = p - h * W;
            int64_t sh = h + dx;
            if (sh < 0)        sh += H;
            else if (sh >= H)  sh -= H;
            int64_t sw = w + dy;
            if (sw < 0)        sw += W;
            else if (sw >= W)  sw -= W;

            const int64_t o_idx = base + p;
            const int64_t x_idx = base + sh * W + sw;
            local += grad_U[o_idx] * x[x_idx] + grad_Z[o_idx];
        }

        extern __shared__ unsigned char smem_raw[];
        scalar_t* smem = reinterpret_cast<scalar_t*>(smem_raw);
        smem[threadIdx.x] = local;
        __syncthreads();
        for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
            if (threadIdx.x < stride)
                smem[threadIdx.x] += smem[threadIdx.x + stride];
            __syncthreads();
        }
        if (threadIdx.x == 0)
            grad_w[sbc] = smem[0];
        __syncthreads();
    }
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// C++ launchers (called from shift_ops.cpp)
// ---------------------------------------------------------------------------

void launch_shift_gather_forward(
    const torch::Tensor& guide,
    const torch::Tensor& shifts,
    torch::Tensor& out,
    torch::Tensor& mask)
{
    const int64_t B = guide.size(0);
    const int64_t C = guide.size(1);
    const int64_t H = guide.size(2);
    const int64_t W = guide.size(3);
    const int64_t S = shifts.size(0);
    const int64_t shift_stride = shifts.size(1);

    auto stream = at::cuda::getCurrentCUDAStream();

    // gather
    const int64_t N = S * B * C * H * W;
    AT_DISPATCH_FLOATING_TYPES(guide.scalar_type(), "shift_gather_fwd", [&] {
        shift_gather_forward_kernel<scalar_t><<<n_blocks(N), THREADS, 0, stream>>>(
            guide.data_ptr<scalar_t>(),
            shifts.data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(),
            S, B, C, H, W, shift_stride);
    });

    // mask
    const int64_t Nm = S * B * H * W;
    AT_DISPATCH_FLOATING_TYPES(guide.scalar_type(), "shift_mask", [&] {
        shift_mask_kernel<scalar_t><<<n_blocks(Nm), THREADS, 0, stream>>>(
            shifts.data_ptr<int64_t>(),
            mask.data_ptr<scalar_t>(),
            S, B, H, W, shift_stride);
    });
}

void launch_pair_gather_forward(
    const torch::Tensor& guide,
    const torch::Tensor& shifts,
    torch::Tensor& out,
    torch::Tensor& mask)
{
    const int64_t B = guide.size(0);
    const int64_t C = guide.size(1);
    const int64_t H = guide.size(2);
    const int64_t W = guide.size(3);
    const int64_t S = shifts.size(0);
    const int64_t shift_stride = shifts.size(1);

    auto stream = at::cuda::getCurrentCUDAStream();

    const int64_t N = S * B * (2 * C) * H * W;
    AT_DISPATCH_FLOATING_TYPES(guide.scalar_type(), "pair_gather_fwd", [&] {
        pair_gather_forward_kernel<scalar_t><<<n_blocks(N), THREADS, 0, stream>>>(
            guide.data_ptr<scalar_t>(),
            shifts.data_ptr<int64_t>(),
            out.data_ptr<scalar_t>(),
            S, B, C, H, W, shift_stride);
    });

    const int64_t Nm = S * B * H * W;
    AT_DISPATCH_FLOATING_TYPES(guide.scalar_type(), "shift_mask", [&] {
        shift_mask_kernel<scalar_t><<<n_blocks(Nm), THREADS, 0, stream>>>(
            shifts.data_ptr<int64_t>(),
            mask.data_ptr<scalar_t>(),
            S, B, H, W, shift_stride);
    });
}

void launch_shift_gather_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& shifts,
    torch::Tensor& grad_guide)
{
    const int64_t B = grad_guide.size(0);
    const int64_t C = grad_guide.size(1);
    const int64_t H = grad_guide.size(2);
    const int64_t W = grad_guide.size(3);
    const int64_t S = shifts.size(0);
    const int64_t shift_stride = shifts.size(1);

    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t N = B * C * H * W; // one thread per guide element
    AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "shift_gather_bwd", [&] {
        shift_gather_backward_kernel<scalar_t><<<n_blocks(N), THREADS, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            shifts.data_ptr<int64_t>(),
            grad_guide.data_ptr<scalar_t>(),
            S, B, C, H, W, shift_stride);
    });
}

void launch_pair_gather_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& shifts,
    torch::Tensor& grad_guide)
{
    const int64_t B = grad_guide.size(0);
    const int64_t C = grad_guide.size(1);
    const int64_t H = grad_guide.size(2);
    const int64_t W = grad_guide.size(3);
    const int64_t S = shifts.size(0);
    const int64_t shift_stride = shifts.size(1);

    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t N = S * B * (2 * C) * H * W;
    AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "pair_gather_bwd", [&] {
        pair_gather_backward_kernel<scalar_t><<<n_blocks(N), THREADS, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            shifts.data_ptr<int64_t>(),
            grad_guide.data_ptr<scalar_t>(),
            S, B, C, H, W, shift_stride);
    });
}

void launch_accumulate_uz_forward(
    const torch::Tensor& x,
    const torch::Tensor& weights,
    const torch::Tensor& shifts,
    torch::Tensor& U,
    torch::Tensor& Z)
{
    const int64_t B = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t S = shifts.size(0);

    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t N = B * C * H * W;
    const int threads = reduction_threads(S);
    const int blocks = n_blocks(N, 1);
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "accumulate_uz_fwd", [&] {
        const size_t smem = 2 * threads * sizeof(scalar_t);
        accumulate_uz_forward_tree_kernel<scalar_t><<<blocks, threads, smem, stream>>>(
            x.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            shifts.data_ptr<int64_t>(),
            U.data_ptr<scalar_t>(),
            Z.data_ptr<scalar_t>(),
            S, B, C, H, W);
    });
}

void launch_accumulate_uz_backward(
    const torch::Tensor& x,
    const torch::Tensor& weights,
    const torch::Tensor& grad_U,
    const torch::Tensor& grad_Z,
    const torch::Tensor& shifts,
    torch::Tensor& grad_x,
    torch::Tensor& grad_w)
{
    const int64_t B = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t S = shifts.size(0);

    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t Nx = B * C * H * W;
    const int64_t Nw = S * B * C * H * W;
    const int threads = reduction_threads(S);
    const int blocks_x = n_blocks(Nx, 1);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "accumulate_uz_bwd_x", [&] {
        const size_t smem = threads * sizeof(scalar_t);
        accumulate_uz_backward_x_tree_kernel<scalar_t><<<blocks_x, threads, smem, stream>>>(
            weights.data_ptr<scalar_t>(),
            grad_U.data_ptr<scalar_t>(),
            shifts.data_ptr<int64_t>(),
            grad_x.data_ptr<scalar_t>(),
            S, B, C, H, W);
    });
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "accumulate_uz_bwd_w", [&] {
        accumulate_uz_backward_w_kernel<scalar_t><<<n_blocks(Nw), THREADS, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            grad_U.data_ptr<scalar_t>(),
            grad_Z.data_ptr<scalar_t>(),
            shifts.data_ptr<int64_t>(),
            grad_w.data_ptr<scalar_t>(),
            S, B, C, H, W);
    });
}

void launch_normalized_accumulate_uz_forward(
    const torch::Tensor& x,
    const torch::Tensor& weights,
    const torch::Tensor& shifts,
    torch::Tensor& D,
    torch::Tensor& log_C,
    bool log_weights)
{
    const int64_t B = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t S = shifts.size(0);
    const int64_t N = B * C * H * W;
    const int threads = reduction_threads(S);
    const int blocks = n_blocks(N, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "normalized_accumulate_uz_fwd", [&] {
        const size_t smem = 3 * threads * sizeof(scalar_t);
        normalized_accumulate_uz_forward_tree_kernel<scalar_t>
            <<<blocks, threads, smem, stream>>>(
                x.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(),
                shifts.data_ptr<int64_t>(), D.data_ptr<scalar_t>(),
                log_C.data_ptr<scalar_t>(), S, B, C, H, W, log_weights);
    });
}

void launch_normalized_accumulate_uz_backward(
    const torch::Tensor& x,
    const torch::Tensor& weights,
    const torch::Tensor& D,
    const torch::Tensor& log_C,
    const torch::Tensor& grad_D,
    const torch::Tensor& grad_log_C,
    const torch::Tensor& shifts,
    torch::Tensor& grad_x,
    torch::Tensor& grad_w,
    bool log_weights)
{
    const int64_t B = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t S = shifts.size(0);
    const int64_t Nx = B * C * H * W;
    const int64_t Nw = S * B * C * H * W;
    const int threads = reduction_threads(S);
    const int blocks_x = n_blocks(Nx, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "normalized_accumulate_uz_bwd_x", [&] {
        const size_t smem = threads * sizeof(scalar_t);
        normalized_accumulate_uz_backward_x_tree_kernel<scalar_t>
            <<<blocks_x, threads, smem, stream>>>(
                weights.data_ptr<scalar_t>(), log_C.data_ptr<scalar_t>(),
                grad_D.data_ptr<scalar_t>(), shifts.data_ptr<int64_t>(),
                grad_x.data_ptr<scalar_t>(), S, B, C, H, W, log_weights);
    });
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "normalized_accumulate_uz_bwd_w", [&] {
        normalized_accumulate_uz_backward_w_kernel<scalar_t>
            <<<n_blocks(Nw), THREADS, 0, stream>>>(
                x.data_ptr<scalar_t>(), weights.data_ptr<scalar_t>(),
                D.data_ptr<scalar_t>(), log_C.data_ptr<scalar_t>(),
                grad_D.data_ptr<scalar_t>(), grad_log_C.data_ptr<scalar_t>(),
                shifts.data_ptr<int64_t>(), grad_w.data_ptr<scalar_t>(),
                S, B, C, H, W, log_weights);
    });
}

void launch_accumulate_uz_scalar_forward(
    const torch::Tensor& x,
    const torch::Tensor& weights,
    const torch::Tensor& shifts,
    torch::Tensor& U,
    torch::Tensor& Z)
{
    const int64_t B = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t S = shifts.size(0);
    const int64_t shift_stride = shifts.size(1);

    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t N = B * C * H * W;
    const int threads = reduction_threads(S);
    const int blocks = n_blocks(N, 1);
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "accumulate_uz_scalar_fwd", [&] {
        const size_t smem = 2 * threads * sizeof(scalar_t);
        accumulate_uz_scalar_forward_tree_kernel<scalar_t><<<blocks, threads, smem, stream>>>(
            x.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            shifts.data_ptr<int64_t>(),
            U.data_ptr<scalar_t>(),
            Z.data_ptr<scalar_t>(),
            S, B, C, H, W, shift_stride);
    });
}

void launch_accumulate_uz_scalar_backward(
    const torch::Tensor& x,
    const torch::Tensor& weights,
    const torch::Tensor& grad_U,
    const torch::Tensor& grad_Z,
    const torch::Tensor& shifts,
    torch::Tensor& grad_x,
    torch::Tensor& grad_w)
{
    const int64_t B = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t S = shifts.size(0);
    const int64_t shift_stride = shifts.size(1);

    auto stream = at::cuda::getCurrentCUDAStream();
    const int64_t Nx = B * C * H * W;
    const int64_t Mw = S * B * C;
    const int reduction_t = reduction_threads(S);
    const int blocks_x = n_blocks(Nx, 1);

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "accumulate_uz_scalar_bwd_x", [&] {
        const size_t smem = reduction_t * sizeof(scalar_t);
        accumulate_uz_scalar_backward_x_tree_kernel<scalar_t><<<blocks_x, reduction_t, smem, stream>>>(
            weights.data_ptr<scalar_t>(),
            grad_U.data_ptr<scalar_t>(),
            shifts.data_ptr<int64_t>(),
            grad_x.data_ptr<scalar_t>(),
            S, B, C, H, W, shift_stride);
    });
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "accumulate_uz_scalar_bwd_w", [&] {
        // One block per (s, b, c); THREADS (a power of two) cooperatively
        // reduce H*W in shared memory.
        const int blocks = n_blocks(Mw, 1);
        const size_t smem = THREADS * sizeof(scalar_t);
        accumulate_uz_scalar_backward_w_kernel<scalar_t><<<blocks, THREADS, smem, stream>>>(
            x.data_ptr<scalar_t>(),
            grad_U.data_ptr<scalar_t>(),
            grad_Z.data_ptr<scalar_t>(),
            shifts.data_ptr<int64_t>(),
            grad_w.data_ptr<scalar_t>(),
            S, B, C, H, W, shift_stride);
    });
}
