// shift_ops_cuda.cu
// CUDA kernels for shift_gather / pair_gather / accumulate_uz used by nekre.
//
// Layout assumptions (HARD):
//   - All input/output tensors are contiguous NCHW (or (S*B, C, H, W) folded).
//   - Stride for an (N, C, H, W) tensor is (C*H*W, H*W, W, 1).
//   - Number of shifts S, batch B, channels C, spatial H, W all fit in int32.
//
// Shifts:
//   - "shifts" is an int32 tensor (S, 2) or (S, 3); we always read 3 ints per
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

namespace {

constexpr int THREADS = 256;

inline int n_blocks(int n, int threads = THREADS) {
    // Cap blocks to a sane value; CUDA_KERNEL_LOOP handles overflow.
    int b = (n + threads - 1) / threads;
    return b < 65535 ? b : 65535;
}

#define CUDA_KERNEL_LOOP(i, n)                                          \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n);        \
         i += blockDim.x * gridDim.x)

// ---------------------------------------------------------------------------
// Forward gather kernels
// ---------------------------------------------------------------------------

template <typename scalar_t>
__global__ void shift_gather_forward_kernel(
    const scalar_t* __restrict__ guide,   // (B, C, H, W)
    const int*      __restrict__ shifts,  // (S, shift_stride)
    scalar_t*       __restrict__ out,     // (S*B, C, H, W)
    int S, int B, int C, int H, int W,
    int shift_stride)
{
    const int N = S * B * C * H * W;
    CUDA_KERNEL_LOOP(idx, N) {
        int w = idx % W;
        int t = idx / W;
        int h = t % H;          t /= H;
        int c = t % C;          t /= C;
        int b = t % B;
        int s = t / B;

        const int dx = shifts[s * shift_stride + 0];
        const int dy = shifts[s * shift_stride + 1];

        int sh = h + dx;
        if (sh < 0)        sh += H;
        else if (sh >= H)  sh -= H;
        int sw = w + dy;
        if (sw < 0)        sw += W;
        else if (sw >= W)  sw -= W;

        const int gidx = ((b * C + c) * H + sh) * W + sw;
        out[idx] = guide[gidx];
    }
}

template <typename scalar_t>
__global__ void pair_gather_forward_kernel(
    const scalar_t* __restrict__ guide,   // (B, C, H, W)
    const int*      __restrict__ shifts,  // (S, shift_stride)
    scalar_t*       __restrict__ out,     // (S*B, 2C, H, W)
    int S, int B, int C, int H, int W,
    int shift_stride)
{
    const int Cout = 2 * C;
    const int N = S * B * Cout * H * W;
    CUDA_KERNEL_LOOP(idx, N) {
        int w = idx % W;
        int t = idx / W;
        int h = t % H;            t /= H;
        int cp = t % Cout;        t /= Cout;
        int b = t % B;
        int s = t / B;

        if (cp < C) {
            // center: guide[b, cp, h, w]
            const int gidx = ((b * C + cp) * H + h) * W + w;
            out[idx] = guide[gidx];
        } else {
            const int c = cp - C;
            const int dx = shifts[s * shift_stride + 0];
            const int dy = shifts[s * shift_stride + 1];

            int sh = h + dx;
            if (sh < 0)        sh += H;
            else if (sh >= H)  sh -= H;
            int sw = w + dy;
            if (sw < 0)        sw += W;
            else if (sw >= W)  sw -= W;

            const int gidx = ((b * C + c) * H + sh) * W + sw;
            out[idx] = guide[gidx];
        }
    }
}

template <typename scalar_t>
__global__ void shift_mask_kernel(
    const int* __restrict__ shifts,  // (S, shift_stride)
    scalar_t*  __restrict__ mask,    // (S*B, 1, H, W)
    int S, int B, int H, int W,
    int shift_stride)
{
    const int N = S * B * H * W;
    CUDA_KERNEL_LOOP(idx, N) {
        int w = idx % W;
        int t = idx / W;
        int h = t % H;            t /= H;
        // b dimension is collapsed (mask is the same for every b)
        int sb = t;
        int s = sb / B;

        const int dx = shifts[s * shift_stride + 0];
        const int dy = shifts[s * shift_stride + 1];

        const int hh = h + dx;
        const int ww = w + dy;
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
    const int*      __restrict__ shifts,     // (S, shift_stride)
    scalar_t*       __restrict__ grad_guide, // (B, C, H, W)
    int S, int B, int C, int H, int W,
    int shift_stride)
{
    // Gather formulation (atomic-free, deterministic): each circular shift is
    // a bijection, so guide[b,c,gh,gw] was read by exactly one output pixel
    // per shift s, namely out[s,b,c,h,w] with h=(gh-dx) mod H, w=(gw-dy) mod W.
    // One thread OWNS one grad_guide element and sums its S contributions.
    // This replaces the old scatter version whose S*B*C*H*W atomicAdds
    // contended S-to-1 on grad_guide and were nondeterministic.
    const int N = B * C * H * W;
    CUDA_KERNEL_LOOP(idx, N) {
        int gw = idx % W;
        int t = idx / W;
        int gh = t % H;           t /= H;
        int c = t % C;
        int b = t / C;

        scalar_t acc = scalar_t(0);
        for (int s = 0; s < S; ++s) {
            const int dx = shifts[s * shift_stride + 0];
            const int dy = shifts[s * shift_stride + 1];

            int h = gh - dx;                      // |dx| < H: one correction
            if (h < 0)        h += H;
            else if (h >= H)  h -= H;
            int w = gw - dy;
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
    const int*      __restrict__ shifts,      // (S, shift_stride)
    scalar_t*       __restrict__ grad_guide,  // (B, C, H, W)
    int S, int B, int C, int H, int W,
    int shift_stride)
{
    const int Cout = 2 * C;
    const int N = S * B * Cout * H * W;
    CUDA_KERNEL_LOOP(idx, N) {
        int w = idx % W;
        int t = idx / W;
        int h = t % H;            t /= H;
        int cp = t % Cout;        t /= Cout;
        int b = t % B;
        int s = t / B;

        const scalar_t go = grad_out[idx];

        if (cp < C) {
            // center half: identity scatter into (b, cp, h, w).
            // Multiple s map here, so we need atomicAdd.
            const int gidx = ((b * C + cp) * H + h) * W + w;
            atomicAdd(grad_guide + gidx, go);
        } else {
            const int c = cp - C;
            const int dx = shifts[s * shift_stride + 0];
            const int dy = shifts[s * shift_stride + 1];

            int sh = h + dx;
            if (sh < 0)        sh += H;
            else if (sh >= H)  sh -= H;
            int sw = w + dy;
            if (sw < 0)        sw += W;
            else if (sw >= W)  sw -= W;

            const int gidx = ((b * C + c) * H + sh) * W + sw;
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
__global__ void accumulate_uz_forward_kernel(
    const scalar_t* __restrict__ x,        // (B, C, H, W)
    const scalar_t* __restrict__ weights,  // (S*B, C, H, W), pre-masked
    const int*      __restrict__ shifts,   // (S, 3): (dx, dy, has_inv)
    scalar_t*       __restrict__ U,        // (B, C, H, W)
    scalar_t*       __restrict__ Z,        // (B, C, H, W)
    int S, int B, int C, int H, int W)
{
    const int N = B * C * H * W;
    CUDA_KERNEL_LOOP(idx, N) {
        int w = idx % W;
        int t = idx / W;
        int h = t % H;            t /= H;
        int c = t % C;
        int b = t / C;

        scalar_t u_val = scalar_t(0);
        scalar_t z_val = scalar_t(0);

        const int x_base   = (b * C + c) * H * W;

        for (int s = 0; s < S; ++s) {
            const int dx = shifts[s * 3 + 0];
            const int dy = shifts[s * 3 + 1];
            const int has_inv = shifts[s * 3 + 2];

            // ---- forward direction ----
            // x is read at circular shift (h+dx) mod H, (w+dy) mod W.
            // weight is read at the natural (h, w). Mask is already folded in.
            int sh = h + dx;
            if (sh < 0)        sh += H;
            else if (sh >= H)  sh -= H;
            int sw = w + dy;
            if (sw < 0)        sw += W;
            else if (sw >= W)  sw -= W;

            const int w_idx = (((s * B + b) * C + c) * H + h) * W + w;
            const int x_idx_fwd = x_base + sh * W + sw;

            const scalar_t wv = weights[w_idx];
            u_val += wv * x[x_idx_fwd];
            z_val += wv;

            // ---- inverse direction (symmetry) ----
            if (has_inv) {
                // mask(-s, h, w) = 1 iff 0 <= h-dx < H && 0 <= w-dy < W.
                const int hi = h - dx;
                const int wi = w - dy;
                if (hi >= 0 && hi < H && wi >= 0 && wi < W) {
                    const int w_idx_inv = (((s * B + b) * C + c) * H + hi) * W + wi;
                    const int x_idx_inv = x_base + hi * W + wi;
                    const scalar_t wv_inv = weights[w_idx_inv];
                    u_val += wv_inv * x[x_idx_inv];
                    z_val += wv_inv;
                }
            }
        }

        U[idx] = u_val;
        Z[idx] = z_val;
    }
}

template <typename scalar_t>
__global__ void accumulate_uz_backward_x_kernel(
    const scalar_t* __restrict__ weights,  // (S*B, C, H, W)
    const scalar_t* __restrict__ grad_U,   // (B, C, H, W)
    const int*      __restrict__ shifts,   // (S, 3)
    scalar_t*       __restrict__ grad_x,   // (B, C, H, W)
    int S, int B, int C, int H, int W)
{
    // grad_x receives no contribution from grad_Z (Z does not depend on x).
    const int N = B * C * H * W;
    CUDA_KERNEL_LOOP(idx, N) {
        int w = idx % W;
        int t = idx / W;
        int h = t % H;            t /= H;
        int c = t % C;
        int b = t / C;

        scalar_t gx = scalar_t(0);

        for (int s = 0; s < S; ++s) {
            const int dx = shifts[s * 3 + 0];
            const int dy = shifts[s * 3 + 1];
            const int has_inv = shifts[s * 3 + 2];

            // (A) forward case: x[b,c,h,w] is read from output position
            // (h_src=h-dx, w_src=w-dy) when 0 <= h-dx < H, 0 <= w-dy < W
            // (this is mask(-s, h, w) = 1).
            const int h_src = h - dx;
            const int w_src = w - dy;
            if (h_src >= 0 && h_src < H && w_src >= 0 && w_src < W) {
                const int w_idx  = (((s * B + b) * C + c) * H + h_src) * W + w_src;
                const int gu_idx = ((b * C + c) * H + h_src) * W + w_src;
                gx += weights[w_idx] * grad_U[gu_idx];
            }

            // (B) inverse case: x[b,c,h,w] is read from output position
            // (h_dst=h+dx, w_dst=w+dy) when has_inv and 0 <= h+dx < H, 0 <= w+dy < W.
            // In that branch weight[s,b,c,h,w] is the source weight.
            if (has_inv) {
                const int h_dst = h + dx;
                const int w_dst = w + dy;
                if (h_dst >= 0 && h_dst < H && w_dst >= 0 && w_dst < W) {
                    const int w_idx  = (((s * B + b) * C + c) * H + h) * W + w;
                    const int gu_idx = ((b * C + c) * H + h_dst) * W + w_dst;
                    gx += weights[w_idx] * grad_U[gu_idx];
                }
            }
        }

        grad_x[idx] = gx;
    }
}

template <typename scalar_t>
__global__ void accumulate_uz_backward_w_kernel(
    const scalar_t* __restrict__ x,        // (B, C, H, W)
    const scalar_t* __restrict__ grad_U,   // (B, C, H, W)
    const scalar_t* __restrict__ grad_Z,   // (B, C, H, W)
    const int*      __restrict__ shifts,   // (S, 3)
    scalar_t*       __restrict__ grad_w,   // (S*B, C, H, W)
    int S, int B, int C, int H, int W)
{
    const int N = S * B * C * H * W;
    CUDA_KERNEL_LOOP(idx, N) {
        int w = idx % W;
        int t = idx / W;
        int h = t % H;            t /= H;
        int c = t % C;            t /= C;
        int b = t % B;
        int s = t / B;

        const int dx = shifts[s * 3 + 0];
        const int dy = shifts[s * 3 + 1];
        const int has_inv = shifts[s * 3 + 2];

        // forward mask at (h, w): 1 iff 0 <= h+dx < H && 0 <= w+dy < W.
        const int sh = h + dx;
        const int sw = w + dy;
        const bool mfwd = (sh >= 0 && sh < H && sw >= 0 && sw < W);

        scalar_t gw = scalar_t(0);
        if (mfwd) {
            // forward contribution: x_shift * gU + gZ at (h, w)
            const int x_idx_fwd  = ((b * C + c) * H + sh) * W + sw;
            const int gu_idx_fwd = ((b * C + c) * H + h) * W + w;
            gw += x[x_idx_fwd]   * grad_U[gu_idx_fwd];
            gw += grad_Z[gu_idx_fwd];

            // inverse contribution: weight[s,b,c,h,w] is the source weight for
            // output position (h+dx, w+dy). The mask there (mask(-s, h+dx, w+dy))
            // equals mask(s, h, w) = mfwd, so we are inside the same branch.
            if (has_inv) {
                const int x_idx_self  = ((b * C + c) * H + h) * W + w;
                const int gu_idx_inv  = ((b * C + c) * H + sh) * W + sw;
                gw += x[x_idx_self] * grad_U[gu_idx_inv];
                gw += grad_Z[gu_idx_inv];
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
__global__ void accumulate_uz_scalar_forward_kernel(
    const scalar_t* __restrict__ x,        // (B, C, H, W)
    const scalar_t* __restrict__ weights,  // (S, B, C)
    const int*      __restrict__ shifts,   // (S, shift_stride)
    scalar_t*       __restrict__ U,        // (B, C, H, W)
    scalar_t*       __restrict__ Z,        // (B, C, H, W)
    int S, int B, int C, int H, int W,
    int shift_stride)
{
    const int N = B * C * H * W;
    CUDA_KERNEL_LOOP(idx, N) {
        int w = idx % W;
        int t = idx / W;
        int h = t % H;            t /= H;
        int c = t % C;
        int b = t / C;

        const int x_base = (b * C + c) * H * W;

        scalar_t u_val = scalar_t(0);
        scalar_t z_val = scalar_t(0);

        for (int s = 0; s < S; ++s) {
            const int dx = shifts[s * shift_stride + 0];
            const int dy = shifts[s * shift_stride + 1];

            int sh = h + dx;
            if (sh < 0)        sh += H;
            else if (sh >= H)  sh -= H;
            int sw = w + dy;
            if (sw < 0)        sw += W;
            else if (sw >= W)  sw -= W;

            const scalar_t wv = weights[(s * B + b) * C + c];
            u_val += wv * x[x_base + sh * W + sw];
            z_val += wv;
        }

        U[idx] = u_val;
        Z[idx] = z_val;
    }
}

template <typename scalar_t>
__global__ void accumulate_uz_scalar_backward_x_kernel(
    const scalar_t* __restrict__ weights,  // (S, B, C)
    const scalar_t* __restrict__ grad_U,   // (B, C, H, W)
    const int*      __restrict__ shifts,   // (S, shift_stride)
    scalar_t*       __restrict__ grad_x,   // (B, C, H, W)
    int S, int B, int C, int H, int W,
    int shift_stride)
{
    // Z is independent of x, so grad_Z contributes nothing to grad_x.
    // x[b,c,h,w] is read by output position (h-dx, w-dy) (circular) for shift s.
    const int N = B * C * H * W;
    CUDA_KERNEL_LOOP(idx, N) {
        int w = idx % W;
        int t = idx / W;
        int h = t % H;            t /= H;
        int c = t % C;
        int b = t / C;

        const int gu_base = (b * C + c) * H * W;

        scalar_t gx = scalar_t(0);
        for (int s = 0; s < S; ++s) {
            const int dx = shifts[s * shift_stride + 0];
            const int dy = shifts[s * shift_stride + 1];

            int oh = h - dx;
            if (oh < 0)        oh += H;
            else if (oh >= H)  oh -= H;
            int ow = w - dy;
            if (ow < 0)        ow += W;
            else if (ow >= W)  ow -= W;

            const scalar_t wv = weights[(s * B + b) * C + c];
            gx += wv * grad_U[gu_base + oh * W + ow];
        }
        grad_x[idx] = gx;
    }
}

template <typename scalar_t>
__global__ void accumulate_uz_scalar_backward_w_kernel(
    const scalar_t* __restrict__ x,        // (B, C, H, W)
    const scalar_t* __restrict__ grad_U,   // (B, C, H, W)
    const scalar_t* __restrict__ grad_Z,   // (B, C, H, W)
    const int*      __restrict__ shifts,   // (S, shift_stride)
    scalar_t*       __restrict__ grad_w,   // (S, B, C)
    int S, int B, int C, int H, int W,
    int shift_stride)
{
    // One BLOCK per (s, b, c); the block's threads cooperatively reduce the
    // H*W grid via shared memory. This keeps occupancy high (S*B*C blocks *
    // blockDim.x threads) instead of the old one-thread-per-(s,b,c) version,
    // whose S*B*C threads each ran a serial H*W loop and left the GPU idle.
    // Deterministic (tree reduction, no atomics). blockDim.x is a power of two.
    //   grad_w[s,b,c] = sum_{h,w} grad_U[b,c,h,w] * x[b,c,(h+dx)%H,(w+dy)%W]
    //                 + sum_{h,w} grad_Z[b,c,h,w]
    const int sbc = blockIdx.x;
    if (sbc >= S * B * C) return;

    const int c = sbc % C;
    int t = sbc / C;
    const int b = t % B;
    const int s = t / B;

    const int dx = shifts[s * shift_stride + 0];
    const int dy = shifts[s * shift_stride + 1];
    const int base = (b * C + c) * H * W;
    const int HW = H * W;

    scalar_t local = scalar_t(0);
    for (int p = threadIdx.x; p < HW; p += blockDim.x) {
        const int h = p / W;
        const int w = p - h * W;
        int sh = h + dx;
        if (sh < 0)        sh += H;
        else if (sh >= H)  sh -= H;
        int sw = w + dy;
        if (sw < 0)        sw += W;
        else if (sw >= W)  sw -= W;

        const int o_idx = base + p;
        const int x_idx = base + sh * W + sw;
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
    const int B = guide.size(0);
    const int C = guide.size(1);
    const int H = guide.size(2);
    const int W = guide.size(3);
    const int S = shifts.size(0);
    const int shift_stride = shifts.size(1);

    auto stream = at::cuda::getCurrentCUDAStream();

    // gather
    const int N = S * B * C * H * W;
    AT_DISPATCH_FLOATING_TYPES(guide.scalar_type(), "shift_gather_fwd", [&] {
        shift_gather_forward_kernel<scalar_t><<<n_blocks(N), THREADS, 0, stream>>>(
            guide.data_ptr<scalar_t>(),
            shifts.data_ptr<int>(),
            out.data_ptr<scalar_t>(),
            S, B, C, H, W, shift_stride);
    });

    // mask
    const int Nm = S * B * H * W;
    AT_DISPATCH_FLOATING_TYPES(guide.scalar_type(), "shift_mask", [&] {
        shift_mask_kernel<scalar_t><<<n_blocks(Nm), THREADS, 0, stream>>>(
            shifts.data_ptr<int>(),
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
    const int B = guide.size(0);
    const int C = guide.size(1);
    const int H = guide.size(2);
    const int W = guide.size(3);
    const int S = shifts.size(0);
    const int shift_stride = shifts.size(1);

    auto stream = at::cuda::getCurrentCUDAStream();

    const int N = S * B * (2 * C) * H * W;
    AT_DISPATCH_FLOATING_TYPES(guide.scalar_type(), "pair_gather_fwd", [&] {
        pair_gather_forward_kernel<scalar_t><<<n_blocks(N), THREADS, 0, stream>>>(
            guide.data_ptr<scalar_t>(),
            shifts.data_ptr<int>(),
            out.data_ptr<scalar_t>(),
            S, B, C, H, W, shift_stride);
    });

    const int Nm = S * B * H * W;
    AT_DISPATCH_FLOATING_TYPES(guide.scalar_type(), "shift_mask", [&] {
        shift_mask_kernel<scalar_t><<<n_blocks(Nm), THREADS, 0, stream>>>(
            shifts.data_ptr<int>(),
            mask.data_ptr<scalar_t>(),
            S, B, H, W, shift_stride);
    });
}

void launch_shift_gather_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& shifts,
    torch::Tensor& grad_guide)
{
    const int B = grad_guide.size(0);
    const int C = grad_guide.size(1);
    const int H = grad_guide.size(2);
    const int W = grad_guide.size(3);
    const int S = shifts.size(0);
    const int shift_stride = shifts.size(1);

    auto stream = at::cuda::getCurrentCUDAStream();
    const int N = B * C * H * W;   // gather form: one thread per guide element
    AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "shift_gather_bwd", [&] {
        shift_gather_backward_kernel<scalar_t><<<n_blocks(N), THREADS, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            shifts.data_ptr<int>(),
            grad_guide.data_ptr<scalar_t>(),
            S, B, C, H, W, shift_stride);
    });
}

void launch_pair_gather_backward(
    const torch::Tensor& grad_out,
    const torch::Tensor& shifts,
    torch::Tensor& grad_guide)
{
    const int B = grad_guide.size(0);
    const int C = grad_guide.size(1);
    const int H = grad_guide.size(2);
    const int W = grad_guide.size(3);
    const int S = shifts.size(0);
    const int shift_stride = shifts.size(1);

    auto stream = at::cuda::getCurrentCUDAStream();
    const int N = S * B * (2 * C) * H * W;
    AT_DISPATCH_FLOATING_TYPES(grad_out.scalar_type(), "pair_gather_bwd", [&] {
        pair_gather_backward_kernel<scalar_t><<<n_blocks(N), THREADS, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(),
            shifts.data_ptr<int>(),
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
    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int S = shifts.size(0);

    auto stream = at::cuda::getCurrentCUDAStream();
    const int N = B * C * H * W;
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "accumulate_uz_fwd", [&] {
        accumulate_uz_forward_kernel<scalar_t><<<n_blocks(N), THREADS, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            shifts.data_ptr<int>(),
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
    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int S = shifts.size(0);

    auto stream = at::cuda::getCurrentCUDAStream();
    const int Nx = B * C * H * W;
    const int Nw = S * B * C * H * W;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "accumulate_uz_bwd_x", [&] {
        accumulate_uz_backward_x_kernel<scalar_t><<<n_blocks(Nx), THREADS, 0, stream>>>(
            weights.data_ptr<scalar_t>(),
            grad_U.data_ptr<scalar_t>(),
            shifts.data_ptr<int>(),
            grad_x.data_ptr<scalar_t>(),
            S, B, C, H, W);
    });
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "accumulate_uz_bwd_w", [&] {
        accumulate_uz_backward_w_kernel<scalar_t><<<n_blocks(Nw), THREADS, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            grad_U.data_ptr<scalar_t>(),
            grad_Z.data_ptr<scalar_t>(),
            shifts.data_ptr<int>(),
            grad_w.data_ptr<scalar_t>(),
            S, B, C, H, W);
    });
}

void launch_accumulate_uz_scalar_forward(
    const torch::Tensor& x,
    const torch::Tensor& weights,
    const torch::Tensor& shifts,
    torch::Tensor& U,
    torch::Tensor& Z)
{
    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int S = shifts.size(0);
    const int shift_stride = shifts.size(1);

    auto stream = at::cuda::getCurrentCUDAStream();
    const int N = B * C * H * W;
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "accumulate_uz_scalar_fwd", [&] {
        accumulate_uz_scalar_forward_kernel<scalar_t><<<n_blocks(N), THREADS, 0, stream>>>(
            x.data_ptr<scalar_t>(),
            weights.data_ptr<scalar_t>(),
            shifts.data_ptr<int>(),
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
    const int B = x.size(0);
    const int C = x.size(1);
    const int H = x.size(2);
    const int W = x.size(3);
    const int S = shifts.size(0);
    const int shift_stride = shifts.size(1);

    auto stream = at::cuda::getCurrentCUDAStream();
    const int Nx = B * C * H * W;
    const int Mw = S * B * C;

    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "accumulate_uz_scalar_bwd_x", [&] {
        accumulate_uz_scalar_backward_x_kernel<scalar_t><<<n_blocks(Nx), THREADS, 0, stream>>>(
            weights.data_ptr<scalar_t>(),
            grad_U.data_ptr<scalar_t>(),
            shifts.data_ptr<int>(),
            grad_x.data_ptr<scalar_t>(),
            S, B, C, H, W, shift_stride);
    });
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "accumulate_uz_scalar_bwd_w", [&] {
        // One block per (s, b, c); THREADS (a power of two) cooperatively
        // reduce H*W in shared memory.
        const int blocks = Mw;
        const size_t smem = THREADS * sizeof(scalar_t);
        accumulate_uz_scalar_backward_w_kernel<scalar_t><<<blocks, THREADS, smem, stream>>>(
            x.data_ptr<scalar_t>(),
            grad_U.data_ptr<scalar_t>(),
            grad_Z.data_ptr<scalar_t>(),
            shifts.data_ptr<int>(),
            grad_w.data_ptr<scalar_t>(),
            S, B, C, H, W, shift_stride);
    });
}
