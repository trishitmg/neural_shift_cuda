// shift_ops.cpp -- pybind11 entry points for the neural_shift_cuda extension.
//
// The CUDA launchers live in shift_ops_cuda.cu. Here we only do tensor
// sanity-checks and allocate output buffers.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#include <vector>

// Launcher prototypes (defined in shift_ops_cuda.cu)
void launch_shift_gather_forward(
    const torch::Tensor& guide, const torch::Tensor& shifts,
    torch::Tensor& out, torch::Tensor& mask);

void launch_pair_gather_forward(
    const torch::Tensor& guide, const torch::Tensor& shifts,
    torch::Tensor& out, torch::Tensor& mask);

void launch_shift_gather_backward(
    const torch::Tensor& grad_out, const torch::Tensor& shifts,
    torch::Tensor& grad_guide);

void launch_pair_gather_backward(
    const torch::Tensor& grad_out, const torch::Tensor& shifts,
    torch::Tensor& grad_guide);

void launch_accumulate_uz_forward(
    const torch::Tensor& x, const torch::Tensor& weights,
    const torch::Tensor& shifts, torch::Tensor& U, torch::Tensor& Z);

void launch_accumulate_uz_backward(
    const torch::Tensor& x, const torch::Tensor& weights,
    const torch::Tensor& grad_U, const torch::Tensor& grad_Z,
    const torch::Tensor& shifts,
    torch::Tensor& grad_x, torch::Tensor& grad_w);

// ---------- helper macros ----------
#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIG(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_NCHW(x) TORCH_CHECK((x).dim() == 4, #x " must have 4 dims (N,C,H,W)")
#define CHECK_INT32(x) TORCH_CHECK((x).scalar_type() == torch::kInt32, #x " must be int32")

static inline torch::Tensor _prep_shifts(torch::Tensor shifts) {
    TORCH_CHECK(shifts.dim() == 2, "shifts must be 2-D (S, 2 or 3)");
    TORCH_CHECK(shifts.size(1) == 2 || shifts.size(1) == 3,
                "shifts must have 2 or 3 columns");
    if (shifts.scalar_type() != torch::kInt32) {
        shifts = shifts.to(torch::kInt32);
    }
    return shifts.contiguous();
}

// ===========================================================================
// shift_gather
// ===========================================================================

std::vector<torch::Tensor> shift_gather_forward(
    torch::Tensor guide, torch::Tensor shifts)
{
    CHECK_CUDA(guide); CHECK_CONTIG(guide); CHECK_NCHW(guide);
    shifts = _prep_shifts(shifts.to(guide.device()));

    const int64_t B = guide.size(0);
    const int64_t C = guide.size(1);
    const int64_t H = guide.size(2);
    const int64_t W = guide.size(3);
    const int64_t S = shifts.size(0);

    auto out  = torch::empty({S * B, C, H, W}, guide.options());
    auto mask = torch::empty({S * B, 1, H, W}, guide.options());

    launch_shift_gather_forward(guide, shifts, out, mask);
    return {out, mask};
}

torch::Tensor shift_gather_backward(
    torch::Tensor grad_out, torch::Tensor shifts,
    int64_t B, int64_t C, int64_t H, int64_t W)
{
    CHECK_CUDA(grad_out); CHECK_CONTIG(grad_out); CHECK_NCHW(grad_out);
    shifts = _prep_shifts(shifts.to(grad_out.device()));

    auto grad_guide = torch::zeros({B, C, H, W}, grad_out.options());
    launch_shift_gather_backward(grad_out, shifts, grad_guide);
    return grad_guide;
}

// ===========================================================================
// pair_gather
// ===========================================================================

std::vector<torch::Tensor> pair_gather_forward(
    torch::Tensor guide, torch::Tensor shifts)
{
    CHECK_CUDA(guide); CHECK_CONTIG(guide); CHECK_NCHW(guide);
    shifts = _prep_shifts(shifts.to(guide.device()));

    const int64_t B = guide.size(0);
    const int64_t C = guide.size(1);
    const int64_t H = guide.size(2);
    const int64_t W = guide.size(3);
    const int64_t S = shifts.size(0);

    auto out  = torch::empty({S * B, 2 * C, H, W}, guide.options());
    auto mask = torch::empty({S * B, 1, H, W}, guide.options());

    launch_pair_gather_forward(guide, shifts, out, mask);
    return {out, mask};
}

torch::Tensor pair_gather_backward(
    torch::Tensor grad_out, torch::Tensor shifts,
    int64_t B, int64_t C, int64_t H, int64_t W)
{
    CHECK_CUDA(grad_out); CHECK_CONTIG(grad_out); CHECK_NCHW(grad_out);
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
    CHECK_CUDA(x); CHECK_CONTIG(x); CHECK_NCHW(x);
    CHECK_CUDA(weights); CHECK_CONTIG(weights); CHECK_NCHW(weights);
    TORCH_CHECK(shifts.dim() == 2 && shifts.size(1) == 3,
                "accumulate_uz: shifts must be (S, 3)");
    shifts = _prep_shifts(shifts.to(x.device()));

    const int64_t B = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t S = shifts.size(0);
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
    CHECK_CUDA(x); CHECK_CONTIG(x);
    CHECK_CUDA(weights); CHECK_CONTIG(weights);
    CHECK_CUDA(grad_U); CHECK_CONTIG(grad_U);
    CHECK_CUDA(grad_Z); CHECK_CONTIG(grad_Z);
    shifts = _prep_shifts(shifts.to(x.device()));

    auto grad_x = torch::empty_like(x);
    auto grad_w = torch::empty_like(weights);
    launch_accumulate_uz_backward(x, weights, grad_U, grad_Z, shifts, grad_x, grad_w);
    return {grad_x, grad_w};
}

// ===========================================================================

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("shift_gather_forward",   &shift_gather_forward,   "shift_gather forward");
    m.def("shift_gather_backward",  &shift_gather_backward,  "shift_gather backward");
    m.def("pair_gather_forward",    &pair_gather_forward,    "pair_gather forward");
    m.def("pair_gather_backward",   &pair_gather_backward,   "pair_gather backward");
    m.def("accumulate_uz_forward",  &accumulate_uz_forward,  "accumulate_uz forward");
    m.def("accumulate_uz_backward", &accumulate_uz_backward, "accumulate_uz backward");
}
