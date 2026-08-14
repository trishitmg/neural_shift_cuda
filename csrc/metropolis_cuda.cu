// metropolis_cuda.cu
// -----------------------------------------------------------------------------
// Fused CUDA kernels for the Metropolis-Hastings denoiser W used by
// nkd_metropolis_attn_v2 / nkd_metropolis_attn_v5.
//
// Notation (matches the PyTorch reference forward()):
//     K x       = sum_pi  N(xi, pi.xi) (.) (pi.x)          [unnormalised aggregate]
//     d         = K e     = sum_pi N(xi, pi.xi)            [degree image]
//     K_hat x   = sum_pi  N(xi, pi.xi) (.) (pi.x) (/) max{d, pi.d}
//     d_hat     = K_hat e
//     W x       = x - d_hat (.) x + K_hat x                [symmetric, stochastic]
//
// Inputs are the per-HALF-PLANE forward weights ALREADY produced by the network
// and head-mixed to C channels: w_half[b, s, c, y, x] = N(xi, pi_s.xi) for the
// canonical half-plane shift s = (dx, dy, has_inv). The inverse-shift twin is
// obtained here by a CIRCULAR SHIFT of the forward weight (NOT a second network
// call), which is exactly what makes K (hence K_hat, W) symmetric by
// construction -- and the reason this op is valid for v5 where the older
// accumulate_uz path was not (post-mix softmax breaks shift-equivariance, but
// the explicit circular-shift twin does not depend on it).
//
// `use_box != 0` reproduces the v2 comp_box behaviour: a neighbour reached only
// by wrap-around (out of [0,H) x [0,W) before the circular pad) contributes 0.
// `use_box == 0` is the v5 pure-circular behaviour.
//
// Two launches:
//   metropolis_degree_kernel   -> d                       (one pass)
//   metropolis_aggregate_kernel-> Khat_x, d_hat using d   (second pass)
// W x is assembled on the host side as x - d_hat*x + Khat_x (cheap, elementwise).
//
// Target build: A6000, TORCH_CUDA_ARCH_LIST="8.0".
// -----------------------------------------------------------------------------

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

namespace neural_shift {

namespace {

constexpr int THREADS = 256;

inline int n_blocks(int64_t n) {
  if (n <= 0) return 0;
  // Overflow-safe ceil division for a signed 64-bit element count.
  const int64_t blocks = 1 + (n - 1) / THREADS;
  return static_cast<int>(blocks < 65535 ? blocks : 65535);
}

#define CUDA_KERNEL_LOOP(i, n)                                           \
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x +       \
                   threadIdx.x;                                          \
       i < (n);                                                          \
       i += static_cast<int64_t>(blockDim.x) * gridDim.x)

// Wrap an index into [0, n) for circular (periodic) addressing.
__device__ __forceinline__ int64_t wrap(int64_t i, int64_t n) {
  i %= n;
  if (i < 0) i += n;
  return i;
}

// in-bounds (no wrap) test used by the comp_box (v2) path.
__device__ __forceinline__ bool in_bounds(int64_t i, int64_t n) {
  return (i >= 0) && (i < n);
}

template <typename scalar_t>
__global__ void metropolis_degree_kernel(
    const scalar_t* __restrict__ w_half,   // (B, S, C, H, W)
    const int64_t*  __restrict__ shifts,   // (S, 3) = (dx, dy, has_inv)
    scalar_t*       __restrict__ d_out,     // (B, C, H, W)
    int64_t B, int64_t S, int64_t C, int64_t H, int64_t W,
    const int use_box) {

  const int64_t total = B * C * H * W;
  CUDA_KERNEL_LOOP(idx, total) {

    const int64_t x = idx % W;
    const int64_t y = (idx / W) % H;
    const int64_t c = (idx / (W * H)) % C;
    const int64_t b = idx / (W * H * C);

    const int64_t HW = H * W;
    const int64_t CHW = C * HW;
    const int64_t SCHW = S * CHW;

    scalar_t acc = scalar_t(0);
    for (int64_t s = 0; s < S; ++s) {
      const int64_t dx = shifts[s * 3 + 0];
      const int64_t dy = shifts[s * 3 + 1];
      const bool has_inv = shifts[s * 3 + 2] != 0;
      const scalar_t* w_s = w_half + b * SCHW + s * CHW + c * HW;

      // forward: weight lives at (y, x); neighbour is (y+dx, x+dy).
      bool keep_f = true;
      if (use_box) keep_f = in_bounds(y + dx, H) && in_bounds(x + dy, W);
      if (keep_f) acc += w_s[y * W + x];

      // inverse twin: w_inv(y,x) = w_fwd((y-dx) mod H, (x-dy) mod W).
      if (has_inv) {
        const int64_t yi = wrap(y - dx, H);
        const int64_t xi = wrap(x - dy, W);
        bool keep_i = true;
        if (use_box) keep_i = in_bounds(y - dx, H) && in_bounds(x - dy, W);
        if (keep_i) acc += w_s[yi * W + xi];
      }
    }
    d_out[idx] = acc;
  }
}

template <typename scalar_t>
__global__ void metropolis_aggregate_kernel(
    const scalar_t* __restrict__ w_half,   // (B, S, C, H, W)
    const scalar_t* __restrict__ img,      // (B, C, H, W)   (unpadded; circular)
    const scalar_t* __restrict__ d,        // (B, C, H, W)   (degree, from pass 1)
    const int64_t*  __restrict__ shifts,   // (S, 3)
    scalar_t*       __restrict__ khat_x,    // (B, C, H, W)
    scalar_t*       __restrict__ d_hat,     // (B, C, H, W)
    int64_t B, int64_t S, int64_t C, int64_t H, int64_t W,
    const int use_box, const scalar_t eps) {

  const int64_t total = B * C * H * W;
  CUDA_KERNEL_LOOP(idx, total) {

    const int64_t x = idx % W;
    const int64_t y = (idx / W) % H;
    const int64_t c = (idx / (W * H)) % C;
    const int64_t b = idx / (W * H * C);

    const int64_t HW = H * W;
    const int64_t CHW = C * HW;
    const int64_t SCHW = S * CHW;

    const scalar_t* img_bc = img + b * CHW + c * HW;
    const scalar_t* d_bc   = d   + b * CHW + c * HW;
    const scalar_t  d_i    = d_bc[y * W + x]; // degree at this pixel

    scalar_t acc_kx = scalar_t(0);
    scalar_t acc_dh = scalar_t(0);

    for (int64_t s = 0; s < S; ++s) {
      const int64_t dx = shifts[s * 3 + 0];
      const int64_t dy = shifts[s * 3 + 1];
      const bool has_inv = shifts[s * 3 + 2] != 0;
      const scalar_t* w_s = w_half + b * SCHW + s * CHW + c * HW;

      // ---- forward: neighbour j = (y+dx, x+dy) (circular) ----
      {
        bool keep = true;
        if (use_box) keep = in_bounds(y + dx, H) && in_bounds(x + dy, W);
        if (keep) {
          const int64_t yn = wrap(y + dx, H);
          const int64_t xn = wrap(x + dy, W);
          const scalar_t wf  = w_s[y * W + x];
          const scalar_t d_j = d_bc[yn * W + xn];
          const scalar_t den = max(d_i, d_j);
          const scalar_t wk  = wf / (den > eps ? den : eps); // K_hat_ij
          acc_kx += wk * img_bc[yn * W + xn];
          acc_dh += wk;
        }
      }

      // ---- inverse twin: neighbour j = (y-dx, x-dy) (circular) ----
      if (has_inv) {
        bool keep = true;
        if (use_box) keep = in_bounds(y - dx, H) && in_bounds(x - dy, W);
        if (keep) {
          const int64_t yn = wrap(y - dx, H);
          const int64_t xn = wrap(x - dy, W);
          const scalar_t wf = w_s[yn * W + xn]; // shifted forward weight
          const scalar_t d_j = d_bc[yn * W + xn];
          const scalar_t den = max(d_i, d_j);
          const scalar_t wk  = wf / (den > eps ? den : eps);
          acc_kx += wk * img_bc[yn * W + xn];
          acc_dh += wk;
        }
      }
    }
    khat_x[idx] = acc_kx;
    d_hat[idx]  = acc_dh;
  }
}

#undef CUDA_KERNEL_LOOP

}  // namespace

// -----------------------------------------------------------------------------
// Host entry point. Returns (Wx, d_hat). Both passes share the same launch config.
// -----------------------------------------------------------------------------
std::vector<torch::Tensor> metropolis_aggregate_cuda(
    torch::Tensor w_half,   // (B, S, C, H, W), contiguous, float/half
    torch::Tensor img,      // (B, C, H, W)
    torch::Tensor shifts,   // (S, 3) int64 on the same device
    int64_t use_box,
    double eps) {

  TORCH_CHECK(w_half.is_cuda(), "w_half must be CUDA");
  TORCH_CHECK(img.is_cuda(), "img must be CUDA");
  TORCH_CHECK(shifts.is_cuda() && shifts.scalar_type() == torch::kLong,
              "shifts must be int64 CUDA (S,3)");
  w_half = w_half.contiguous();
  img    = img.contiguous();
  shifts = shifts.contiguous();

  const int64_t B = w_half.size(0);
  const int64_t S = w_half.size(1);
  const int64_t C = w_half.size(2);
  const int64_t H = w_half.size(3);
  const int64_t W = w_half.size(4);

  auto d      = torch::zeros_like(img);
  auto khat_x = torch::zeros_like(img);
  auto d_hat  = torch::zeros_like(img);

  const int64_t total = img.numel();
  const int blocks = n_blocks(total);
  auto stream = at::cuda::getCurrentCUDAStream();

  AT_DISPATCH_FLOATING_TYPES_AND_HALF(
      w_half.scalar_type(), "metropolis_aggregate_cuda", ([&] {
        metropolis_degree_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
            w_half.data_ptr<scalar_t>(), shifts.data_ptr<int64_t>(),
            d.data_ptr<scalar_t>(), B, S, C, H, W, (int)use_box);
        metropolis_aggregate_kernel<scalar_t><<<blocks, THREADS, 0, stream>>>(
            w_half.data_ptr<scalar_t>(), img.data_ptr<scalar_t>(),
            d.data_ptr<scalar_t>(), shifts.data_ptr<int64_t>(),
            khat_x.data_ptr<scalar_t>(), d_hat.data_ptr<scalar_t>(),
            B, S, C, H, W, (int)use_box, (scalar_t)eps);
      }));

  // W x = x - d_hat (.) x + K_hat x  (elementwise; ATen handles the launch).
  auto Wx = img - d_hat * img + khat_x;
  // Return (Wx, d_hat = K_hat e). d (= K e) is an internal-only intermediate.
  return {Wx, d_hat};
}

}  // namespace neural_shift
