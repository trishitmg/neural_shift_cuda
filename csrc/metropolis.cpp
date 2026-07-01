// metropolis.cpp
// -----------------------------------------------------------------------------
// Host-side declaration + pybind registration for the fused Metropolis
// aggregate op. Add this file to the `sources` list of the existing extension
// in setup.py (next to the current shift_gather / pair_gather / accumulate_uz
// sources) so it compiles into the SAME module, neural_shift_cuda._C.
//
// The repo already owns the PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) block, so
// this file must NOT declare a second one (two PYBIND11_MODULE for the same
// module is a redefinition). Instead it exposes a plain registration helper,
//     neural_shift::register_metropolis(m);
// which you call from inside the existing module block. See INTEGRATION.md.
// -----------------------------------------------------------------------------

#include <torch/extension.h>
#include <vector>

namespace neural_shift
{

  // Defined in metropolis_cuda.cu
  std::vector<torch::Tensor> metropolis_aggregate_cuda(
      torch::Tensor w_half,
      torch::Tensor img,
      torch::Tensor shifts,
      int64_t use_box,
      double eps);

  std::vector<torch::Tensor> metropolis_aggregate(
      torch::Tensor w_half,
      torch::Tensor img,
      torch::Tensor shifts,
      int64_t use_box,
      double eps)
  {
    return metropolis_aggregate_cuda(w_half, img, shifts, use_box, eps);
  }

  // Call this from the repo's existing PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
  // block: add `neural_shift::register_metropolis(m);` (and a forward decl
  // `namespace neural_shift { void register_metropolis(pybind11::module_&); }`).
  void register_metropolis(pybind11::module_ &m)
  {
    m.def("metropolis_aggregate", &metropolis_aggregate,
          "Fused Metropolis-Hastings aggregate -> (Wx, d_hat) (CUDA)",
          pybind11::arg("w_half"), pybind11::arg("img"), pybind11::arg("shifts"),
          pybind11::arg("use_box"), pybind11::arg("eps") = 1e-6);
  }

} // namespace neural_shift

// -----------------------------------------------------------------------------
// Standalone fallback ONLY for building this op as its own private module during
// isolated testing. Disabled by default so it cannot collide with the repo's
// real module. Define NEURAL_SHIFT_METROPOLIS_STANDALONE to enable.
// -----------------------------------------------------------------------------
#ifdef NEURAL_SHIFT_METROPOLIS_STANDALONE
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  neural_shift::register_metropolis(m);
}
#endif