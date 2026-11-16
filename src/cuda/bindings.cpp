#include <pybind11/pybind11.h>
#include <torch/csrc/utils/pybind.h>
using at::Tensor;
Tensor w4_cuda(Tensor, Tensor, Tensor, Tensor, int64_t, int64_t);
Tensor rms_cuda(Tensor, Tensor, double);
Tensor rope_cuda(Tensor, Tensor, int64_t, double, bool);
Tensor swiglu_cuda(Tensor, Tensor);
Tensor attention_cuda(Tensor, Tensor, Tensor, Tensor, double, bool);
Tensor decode_cuda(Tensor, Tensor, Tensor, Tensor, double);
Tensor qk_norm_rope_cuda(Tensor, Tensor, Tensor, double, double);
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("decode_online", &decode_cuda);
    m.def("qk_norm_rope", &qk_norm_rope_cuda);
    m.def("w4", &w4_cuda, "Packed W4: scalar (0), warp matvec (1), WMMA (2)");
    m.def("rms_norm", &rms_cuda);
    m.def("rope", &rope_cuda);
    m.def("swiglu", &swiglu_cuda);
    m.def("attention", &attention_cuda);
}
