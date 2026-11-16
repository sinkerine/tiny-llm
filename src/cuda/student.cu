// Week 2 learner seams. Implement the CUDA kernels here; never link the reference.
#include <ATen/ATen.h>
using at::Tensor;
// Reference-only selected scheduling controls are not additional learner tasks.
Tensor decode_cuda(Tensor, Tensor, Tensor, Tensor, double) {
    TORCH_CHECK(false, "online decode is a reference-only scheduling optimization");
}
Tensor qk_norm_rope_cuda(Tensor, Tensor, Tensor, double, double) {
    TORCH_CHECK(false, "QK norm/RoPE fusion is a reference-only scheduling optimization");
}
Tensor w4_cuda(Tensor, Tensor, Tensor, Tensor, int64_t, int64_t) {
    TORCH_CHECK(false, "Week 2 Days 2–3: implement packed W4 CUDA kernels");
}
Tensor rms_cuda(Tensor, Tensor, double) { TORCH_CHECK(false, "Week 2 Day 4: implement RMSNorm CUDA"); }
Tensor rope_cuda(Tensor, Tensor, int64_t, double, bool) { TORCH_CHECK(false, "Week 2 Day 4: implement RoPE CUDA"); }
Tensor swiglu_cuda(Tensor, Tensor) { TORCH_CHECK(false, "Week 2 Day 4: implement SwiGLU CUDA"); }
Tensor attention_cuda(Tensor, Tensor, Tensor, Tensor, double, bool) {
    TORCH_CHECK(false, "Week 2 Day 5: implement tiled CUDA attention");
}
