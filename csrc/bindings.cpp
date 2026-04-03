/*
 * ManthanQuant x86 — pybind11 bindings for TurboQuant CUDA kernels
 * BiltIQ AI
 */

#include <torch/extension.h>
#include <tuple>

// Forward declarations from turboquant_kernel.cu
std::tuple<torch::Tensor, torch::Tensor> tq_encode(
    torch::Tensor input, int seed, int bits);
torch::Tensor tq_decode(
    torch::Tensor radii, torch::Tensor packed, int D, int seed, int bits);

PYBIND11_MODULE(_C, m) {
    m.doc() = "ManthanQuant x86 TurboQuant CUDA kernels (BiltIQ AI)";

    m.def("tq_encode", &tq_encode,
          "TurboQuant encode: [N, D] float → (radii [N], packed [N, words])",
          py::arg("input"),
          py::arg("seed") = 42,
          py::arg("bits") = 3);

    m.def("tq_decode", &tq_decode,
          "TurboQuant decode: (radii, packed) → [N, D] float",
          py::arg("radii"),
          py::arg("packed"),
          py::arg("D"),
          py::arg("seed") = 42,
          py::arg("bits") = 3);
}
