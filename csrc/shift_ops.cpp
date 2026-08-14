// shift_ops.cpp -- pybind11 entry points for the neural_shift_cuda extension.
//
// The CUDA launchers live in shift_ops_cuda.cu. Here we only do tensor
// sanity-checks and allocate output buffers.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>

// Launcher prototypes (defined in shift_ops_cuda.cu)
void launch_shift_gather_forward(
    const torch::Tensor &guide, const torch::Tensor &shifts,
    torch::Tensor &out, torch::Tensor &mask);

void launch_pair_gather_forward(
    const torch::Tensor &guide, const torch::Tensor &shifts,
    torch::Tensor &out, torch::Tensor &mask);

void launch_shift_gather_backward(
    const torch::Tensor &grad_out, const torch::Tensor &shifts,
    torch::Tensor &grad_guide);

void launch_pair_gather_backward(
    const torch::Tensor &grad_out, const torch::Tensor &shifts,
    torch::Tensor &grad_guide);

void launch_accumulate_uz_forward(
    const torch::Tensor &x, const torch::Tensor &weights,
    const torch::Tensor &shifts, torch::Tensor &U, torch::Tensor &Z);

void launch_accumulate_uz_backward(
    const torch::Tensor &x, const torch::Tensor &weights,
    const torch::Tensor &grad_U, const torch::Tensor &grad_Z,
    const torch::Tensor &shifts,
    torch::Tensor &grad_x, torch::Tensor &grad_w);

void launch_normalized_accumulate_uz_forward(
    const torch::Tensor &x, const torch::Tensor &weights,
    const torch::Tensor &shifts, torch::Tensor &D, torch::Tensor &log_C,
    bool log_weights);

void launch_normalized_accumulate_uz_backward(
    const torch::Tensor &x, const torch::Tensor &weights,
    const torch::Tensor &D, const torch::Tensor &log_C,
    const torch::Tensor &grad_D, const torch::Tensor &grad_log_C,
    const torch::Tensor &shifts,
    torch::Tensor &grad_x, torch::Tensor &grad_w,
    bool log_weights);

void launch_accumulate_uz_scalar_forward(
    const torch::Tensor &x, const torch::Tensor &weights,
    const torch::Tensor &shifts, torch::Tensor &U, torch::Tensor &Z);

void launch_accumulate_uz_scalar_backward(
    const torch::Tensor &x, const torch::Tensor &weights,
    const torch::Tensor &grad_U, const torch::Tensor &grad_Z,
    const torch::Tensor &shifts,
    torch::Tensor &grad_x, torch::Tensor &grad_w);

// ---------- helper macros ----------
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_NCHW(x) TORCH_CHECK((x).dim() == 4, #x " must have 4 dims (N,C,H,W)")
static inline torch::Tensor _prep_shifts(torch::Tensor shifts)
{
    TORCH_CHECK(shifts.dim() == 2, "shifts must be 2-D (S, 2 or 3)");
    TORCH_CHECK(shifts.size(1) == 2 || shifts.size(1) == 3,
                "shifts must have 2 or 3 columns");
    if (shifts.scalar_type() != torch::kInt64)
    {
        shifts = shifts.to(torch::kInt64);
    }
    return shifts.contiguous();
}

// ===========================================================================
// shift_gather
// ===========================================================================

std::vector<torch::Tensor> shift_gather_forward(
    torch::Tensor guide, torch::Tensor shifts)
{
    CHECK_CUDA(guide);
    CHECK_CONTIG(guide);
    CHECK_NCHW(guide);
    shifts = _prep_shifts(shifts.to(guide.device()));

    const int64_t B = guide.size(0);
    const int64_t C = guide.size(1);
    const int64_t H = guide.size(2);
    const int64_t W = guide.size(3);
    const int64_t S = shifts.size(0);

    auto out = torch::empty({S * B, C, H, W}, guide.options());
    auto mask = torch::empty({S * B, 1, H, W}, guide.options());

    launch_shift_gather_forward(guide, shifts, out, mask);
    return {out, mask};
}

torch::Tensor shift_gather_backward(
    torch::Tensor grad_out, torch::Tensor shifts,
    int64_t B, int64_t C, int64_t H, int64_t W)
{
    CHECK_CUDA(grad_out);
    CHECK_CONTIG(grad_out);
    CHECK_NCHW(grad_out);
    shifts = _prep_shifts(shifts.to(grad_out.device()));

    // empty, not zeros: the gather-form backward kernel assigns every element.
    auto grad_guide = torch::empty({B, C, H, W}, grad_out.options());
    launch_shift_gather_backward(grad_out, shifts, grad_guide);
    return grad_guide;
}

// ===========================================================================
// pair_gather
// ===========================================================================

std::vector<torch::Tensor> pair_gather_forward(
    torch::Tensor guide, torch::Tensor shifts)
{
    CHECK_CUDA(guide);
    CHECK_CONTIG(guide);
    CHECK_NCHW(guide);
    shifts = _prep_shifts(shifts.to(guide.device()));

    const int64_t B = guide.size(0);
    const int64_t C = guide.size(1);
    const int64_t H = guide.size(2);
    const int64_t W = guide.size(3);
    const int64_t S = shifts.size(0);

    auto out = torch::empty({S * B, 2 * C, H, W}, guide.options());
    auto mask = torch::empty({S * B, 1, H, W}, guide.options());

    launch_pair_gather_forward(guide, shifts, out, mask);
    return {out, mask};
}

torch::Tensor pair_gather_backward(
    torch::Tensor grad_out, torch::Tensor shifts,
    int64_t B, int64_t C, int64_t H, int64_t W)
{
    CHECK_CUDA(grad_out);
    CHECK_CONTIG(grad_out);
    CHECK_NCHW(grad_out);
    shifts = _prep_shifts(shifts.to(grad_out.device()));

    auto grad_guide = torch::zeros({B, C, H, W}, grad_out.options());
    launch_pair_gather_backward(grad_out, shifts, grad_guide);
    return grad_guide;
}

// ===========================================================================
// accumulate_uz
// ===========================================================================

std::vector<torch::Tensor> accumulate_uz_forward(
    torch::Tensor x, torch::Tensor weights, torch::Tensor shifts)
{
    CHECK_CUDA(x);
    CHECK_CONTIG(x);
    CHECK_NCHW(x);
    CHECK_CUDA(weights);
    CHECK_CONTIG(weights);
    CHECK_NCHW(weights);
    TORCH_CHECK(shifts.dim() == 2 && shifts.size(1) == 3,
                "accumulate_uz: shifts must be (S, 3)");
    shifts = _prep_shifts(shifts.to(x.device()));

    const int64_t B = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t S = shifts.size(0);
    TORCH_CHECK(S > 0 && S <= 1024,
                "accumulate_uz: tree reduction requires 1 <= S <= 1024");
    TORCH_CHECK(weights.device() == x.device(),
                "accumulate_uz: x and weights must be on the same device");
    TORCH_CHECK(weights.scalar_type() == x.scalar_type(),
                "accumulate_uz: x and weights must have the same dtype");
    TORCH_CHECK(weights.size(0) == S * B && weights.size(1) == C &&
                    weights.size(2) == H && weights.size(3) == W,
                "weights shape must be (S*B, C, H, W)");

    auto U = torch::empty_like(x);
    auto Z = torch::empty_like(x);
    launch_accumulate_uz_forward(x, weights, shifts, U, Z);
    return {U, Z};
}

std::vector<torch::Tensor> accumulate_uz_backward(
    torch::Tensor x, torch::Tensor weights,
    torch::Tensor grad_U, torch::Tensor grad_Z, torch::Tensor shifts)
{
    CHECK_CUDA(x);
    CHECK_CONTIG(x);
    CHECK_CUDA(weights);
    CHECK_CONTIG(weights);
    CHECK_CUDA(grad_U);
    CHECK_CONTIG(grad_U);
    CHECK_CUDA(grad_Z);
    CHECK_CONTIG(grad_Z);
    shifts = _prep_shifts(shifts.to(x.device()));

    auto grad_x = torch::empty_like(x);
    auto grad_w = torch::empty_like(weights);
    launch_accumulate_uz_backward(x, weights, grad_U, grad_Z, shifts, grad_x, grad_w);
    return {grad_x, grad_w};
}

// ===========================================================================
// normalized_accumulate_uz -- overflow-safe D=K/C plus log(C)
// ===========================================================================

std::vector<torch::Tensor> normalized_accumulate_uz_forward(
    torch::Tensor x, torch::Tensor weights, torch::Tensor shifts,
    bool log_weights)
{
    CHECK_CUDA(x);
    CHECK_CONTIG(x);
    CHECK_NCHW(x);
    CHECK_CUDA(weights);
    CHECK_CONTIG(weights);
    CHECK_NCHW(weights);
    TORCH_CHECK(shifts.dim() == 2 && shifts.size(1) == 3,
                "normalized_accumulate_uz: shifts must be (S, 3)");
    shifts = _prep_shifts(shifts.to(x.device()));

    const int64_t B = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t S = shifts.size(0);
    TORCH_CHECK(S > 0 && S <= 1024,
                "normalized_accumulate_uz: tree reduction requires 1 <= S <= 1024");
    TORCH_CHECK(weights.device() == x.device(),
                "normalized_accumulate_uz: x and weights must be on the same device");
    TORCH_CHECK(weights.scalar_type() == x.scalar_type(),
                "normalized_accumulate_uz: x and weights must have the same dtype");
    TORCH_CHECK(weights.size(0) == S * B && weights.size(1) == C &&
                    weights.size(2) == H && weights.size(3) == W,
                "normalized_accumulate_uz: weights must be (S*B, C, H, W)");

    auto D = torch::empty_like(x);
    auto log_C = torch::empty_like(x);
    launch_normalized_accumulate_uz_forward(
        x, weights, shifts, D, log_C, log_weights);
    return {D, log_C};
}

std::vector<torch::Tensor> normalized_accumulate_uz_backward(
    torch::Tensor x, torch::Tensor weights,
    torch::Tensor D, torch::Tensor log_C,
    torch::Tensor grad_D, torch::Tensor grad_log_C,
    torch::Tensor shifts, bool log_weights)
{
    CHECK_CUDA(x);
    CHECK_CONTIG(x);
    CHECK_CUDA(weights);
    CHECK_CONTIG(weights);
    CHECK_CUDA(D);
    CHECK_CONTIG(D);
    CHECK_CUDA(log_C);
    CHECK_CONTIG(log_C);
    CHECK_CUDA(grad_D);
    CHECK_CONTIG(grad_D);
    CHECK_CUDA(grad_log_C);
    CHECK_CONTIG(grad_log_C);
    shifts = _prep_shifts(shifts.to(x.device()));

    auto grad_x = torch::empty_like(x);
    auto grad_w = torch::empty_like(weights);
    launch_normalized_accumulate_uz_backward(
        x, weights, D, log_C, grad_D, grad_log_C, shifts,
        grad_x, grad_w, log_weights);
    return {grad_x, grad_w};
}

// ===========================================================================
// accumulate_uz_scalar  (GASD scalar-per-transform)
//   weights is (S, B, C): one scalar per (transform, image, channel). shifts
//   is (S, 2) or (S, 3) (only the first two columns are read). No mask, no
//   inverse-symmetry branch.
// ===========================================================================

std::vector<torch::Tensor> accumulate_uz_scalar_forward(
    torch::Tensor x, torch::Tensor weights, torch::Tensor shifts)
{
    CHECK_CUDA(x);
    CHECK_CONTIG(x);
    CHECK_NCHW(x);
    CHECK_CUDA(weights);
    CHECK_CONTIG(weights);
    TORCH_CHECK(weights.dim() == 3, "accumulate_uz_scalar: weights must be (S, B, C)");
    shifts = _prep_shifts(shifts.to(x.device()));

    const int64_t B = x.size(0);
    const int64_t C = x.size(1);
    const int64_t S = shifts.size(0);
    TORCH_CHECK(S > 0 && S <= 1024,
                "accumulate_uz_scalar: tree reduction requires 1 <= S <= 1024");
    TORCH_CHECK(weights.device() == x.device(),
                "accumulate_uz_scalar: x and weights must be on the same device");
    TORCH_CHECK(weights.scalar_type() == x.scalar_type(),
                "accumulate_uz_scalar: x and weights must have the same dtype");
    TORCH_CHECK(weights.size(0) == S && weights.size(1) == B && weights.size(2) == C,
                "weights shape must be (S, B, C)");

    auto U = torch::empty_like(x);
    auto Z = torch::empty_like(x);
    launch_accumulate_uz_scalar_forward(x, weights, shifts, U, Z);
    return {U, Z};
}

std::vector<torch::Tensor> accumulate_uz_scalar_backward(
    torch::Tensor x, torch::Tensor weights,
    torch::Tensor grad_U, torch::Tensor grad_Z, torch::Tensor shifts)
{
    CHECK_CUDA(x);
    CHECK_CONTIG(x);
    CHECK_CUDA(weights);
    CHECK_CONTIG(weights);
    CHECK_CUDA(grad_U);
    CHECK_CONTIG(grad_U);
    CHECK_CUDA(grad_Z);
    CHECK_CONTIG(grad_Z);
    shifts = _prep_shifts(shifts.to(x.device()));

    auto grad_x = torch::empty_like(x);
    auto grad_w = torch::empty_like(weights);
    launch_accumulate_uz_scalar_backward(
        x, weights, grad_U, grad_Z, shifts, grad_x, grad_w);
    return {grad_x, grad_w};
}

// ===========================================================================

// Defined in metropolis.cpp -- registers metropolis_aggregate into this module.
namespace neural_shift
{
    void register_metropolis(pybind11::module_ &m);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    // ABI capability marker. Python checks this before passing int64 shift
    // buffers to prevent a stale pre-0.11.1 binary from reading them as int32.
    m.def("index_width_bits", []() { return 64; },
          "Width of CUDA tensor indices and shift offsets");
    m.def("shift_gather_forward", &shift_gather_forward, "shift_gather forward");
    m.def("shift_gather_backward", &shift_gather_backward, "shift_gather backward");
    m.def("pair_gather_forward", &pair_gather_forward, "pair_gather forward");
    m.def("pair_gather_backward", &pair_gather_backward, "pair_gather backward");
    m.def("accumulate_uz_forward", &accumulate_uz_forward, "accumulate_uz forward");
    m.def("accumulate_uz_backward", &accumulate_uz_backward, "accumulate_uz backward");
    m.def("normalized_accumulate_uz_forward", &normalized_accumulate_uz_forward,
          "overflow-safe normalized accumulate forward");
    m.def("normalized_accumulate_uz_backward", &normalized_accumulate_uz_backward,
          "overflow-safe normalized accumulate backward");
    m.def("accumulate_uz_scalar_forward", &accumulate_uz_scalar_forward,
          "accumulate_uz_scalar forward");
    m.def("accumulate_uz_scalar_backward", &accumulate_uz_scalar_backward,
          "accumulate_uz_scalar backward");
    neural_shift::register_metropolis(m);
}
