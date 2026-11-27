// Week 2 learner seams. Implement the CUDA kernels here; never link the reference.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cstdint>

using at::Tensor;

namespace {

__device__ __forceinline__ __nv_bfloat16 unpack_w4_as_bf16(
    uint32_t word, int nibble_idx, float scale, float bias) {
    uint32_t code = (word >> (4 * nibble_idx)) & 15u;
    return __float2bfloat16(static_cast<float>(code) * scale + bias);
}

__device__ __forceinline__ float unpack_w4_bf16(
    uint32_t word, int nibble_idx, float scale, float bias) {
    return __bfloat162float(unpack_w4_as_bf16(word, nibble_idx, scale, bias));
}

__global__ void w4_scalar_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ biases,
    __nv_bfloat16* __restrict__ out,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t group_size) {
    int64_t n = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t m = static_cast<int64_t>(blockIdx.y) * blockDim.y + threadIdx.y;
    if (m >= M || n >= N) {
        return;
    }

    int64_t packed_k = K >> 3;
    int64_t groups = K / group_size;
    const __nv_bfloat16* x_row = x + m * K;
    const int32_t* p_row = packed + n * packed_k;
    const __nv_bfloat16* s_row = scales + n * groups;
    const __nv_bfloat16* b_row = biases + n * groups;

    float acc = 0.0f;
    for (int64_t k = 0; k < K; ++k) {
        uint32_t word = static_cast<uint32_t>(p_row[k >> 3]);
        int64_t g = k / group_size;
        float scale = __bfloat162float(s_row[g]);
        float bias = __bfloat162float(b_row[g]);
        float w = unpack_w4_bf16(word, static_cast<int>(k & 7), scale, bias);
        acc += __bfloat162float(x_row[k]) * w;
    }
    out[m * N + n] = __float2bfloat16(acc);
}

constexpr int kW4WarpMatvecWarps = 4;

__global__ void w4_warp_matvec_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ biases,
    __nv_bfloat16* __restrict__ out,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t group_size) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    int64_t n = static_cast<int64_t>(blockIdx.x) * kW4WarpMatvecWarps + warp_id;
    int64_t m = static_cast<int64_t>(blockIdx.y);
    if (m >= M || n >= N) {
        return;
    }

    int64_t packed_k = K >> 3;
    int64_t groups = K / group_size;
    const __nv_bfloat16* x_row = x + m * K;
    const int32_t* p_row = packed + n * packed_k;
    const __nv_bfloat16* s_row = scales + n * groups;
    const __nv_bfloat16* b_row = biases + n * groups;

    float acc = 0.0f;
    for (int64_t k = lane; k < K; k += 32) {
        uint32_t word = static_cast<uint32_t>(p_row[k >> 3]);
        int64_t g = k / group_size;
        float scale = __bfloat162float(s_row[g]);
        float bias = __bfloat162float(b_row[g]);
        float w = unpack_w4_bf16(word, static_cast<int>(k & 7), scale, bias);
        acc += __bfloat162float(x_row[k]) * w;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (lane == 0) {
        out[m * N + n] = __float2bfloat16(acc);
    }
}

__global__ void w4_wmma_kernel(
    const __nv_bfloat16* __restrict__ x,
    const int32_t* __restrict__ packed,
    const __nv_bfloat16* __restrict__ scales,
    const __nv_bfloat16* __restrict__ biases,
    __nv_bfloat16* __restrict__ out,
    int64_t M,
    int64_t N,
    int64_t K,
    int64_t group_size) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    using namespace nvcuda;
    int64_t m0 = static_cast<int64_t>(blockIdx.y) * 16;
    int64_t n0 = static_cast<int64_t>(blockIdx.x) * 16;
    int lane = threadIdx.x;

    __shared__ alignas(16) __nv_bfloat16 a_sh[16 * 16];
    __shared__ alignas(16) __nv_bfloat16 b_sh[16 * 16];
    __shared__ alignas(16) float c_sh[16 * 16];

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major>
        a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major>
        b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);

    int64_t packed_k = K >> 3;
    int64_t groups = K / group_size;

    for (int64_t k0 = 0; k0 < K; k0 += 16) {
        for (int idx = lane; idx < 256; idx += 32) {
            int r = idx >> 4;
            int c = idx & 15;

            int64_t m = m0 + r;
            int64_t k_a = k0 + c;
            a_sh[idx] = (m < M && k_a < K)
                            ? x[m * K + k_a]
                            : __float2bfloat16(0.0f);

            int64_t n = n0 + r;
            int64_t k_b = k0 + c;
            if (n < N && k_b < K) {
                uint32_t word =
                    static_cast<uint32_t>(packed[n * packed_k + (k_b >> 3)]);
                int64_t g = k_b / group_size;
                float scale = __bfloat162float(scales[n * groups + g]);
                float bias = __bfloat162float(biases[n * groups + g]);
                b_sh[idx] =
                    unpack_w4_as_bf16(word, static_cast<int>(k_b & 7), scale, bias);
            } else {
                b_sh[idx] = __float2bfloat16(0.0f);
            }
        }
        __syncwarp();

        wmma::load_matrix_sync(a_frag, a_sh, 16);
        wmma::load_matrix_sync(b_frag, b_sh, 16);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncwarp();
    }

    wmma::store_matrix_sync(c_sh, c_frag, 16, wmma::mem_row_major);
    __syncwarp();

    for (int idx = lane; idx < 256; idx += 32) {
        int r = idx >> 4;
        int c = idx & 15;
        int64_t m = m0 + r;
        int64_t n = n0 + c;
        if (m < M && n < N) {
            out[m * N + n] = __float2bfloat16(c_sh[idx]);
        }
    }
#endif
}

}  // namespace

// Reference-only selected scheduling controls are not additional learner tasks.
Tensor decode_cuda(Tensor, Tensor, Tensor, Tensor, double) {
    TORCH_CHECK(false, "online decode is a reference-only scheduling optimization");
}
Tensor qk_norm_rope_cuda(Tensor, Tensor, Tensor, double, double) {
    TORCH_CHECK(false, "QK norm/RoPE fusion is a reference-only scheduling optimization");
}
Tensor w4_cuda(
    Tensor x,
    Tensor packed,
    Tensor scales,
    Tensor biases,
    int64_t group_size,
    int64_t mode) {
    c10::cuda::CUDAGuard guard(x.device());
    int64_t K = x.size(-1);
    int64_t M = x.numel() / K;
    int64_t N = packed.size(0);
    Tensor out = at::empty({M, N}, x.options());
    if (M == 0 || N == 0) {
        return out;
    }

    const auto* x_ptr =
        reinterpret_cast<const __nv_bfloat16*>(x.data_ptr<at::BFloat16>());
    const int32_t* p_ptr = packed.data_ptr<int32_t>();
    const auto* s_ptr =
        reinterpret_cast<const __nv_bfloat16*>(scales.data_ptr<at::BFloat16>());
    const auto* b_ptr =
        reinterpret_cast<const __nv_bfloat16*>(biases.data_ptr<at::BFloat16>());
    auto* out_ptr =
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());
    auto stream = at::cuda::getCurrentCUDAStream();

    if (mode == 0) {
        dim3 block(16, 16);
        dim3 grid(
            static_cast<unsigned int>((N + block.x - 1) / block.x),
            static_cast<unsigned int>((M + block.y - 1) / block.y));
        w4_scalar_kernel<<<grid, block, 0, stream>>>(
            x_ptr, p_ptr, s_ptr, b_ptr, out_ptr, M, N, K, group_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
    if (mode == 1) {
        TORCH_CHECK(M <= 8, "w4 warp matvec supports M <= 8");
        dim3 block(kW4WarpMatvecWarps * 32);
        dim3 grid(
            static_cast<unsigned int>((N + kW4WarpMatvecWarps - 1) / kW4WarpMatvecWarps),
            static_cast<unsigned int>(M));
        w4_warp_matvec_kernel<<<grid, block, 0, stream>>>(
            x_ptr, p_ptr, s_ptr, b_ptr, out_ptr, M, N, K, group_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
    if (mode == 2) {
        dim3 block(32);
        dim3 grid(
            static_cast<unsigned int>((N + 15) / 16),
            static_cast<unsigned int>((M + 15) / 16));
        w4_wmma_kernel<<<grid, block, 0, stream>>>(
            x_ptr, p_ptr, s_ptr, b_ptr, out_ptr, M, N, K, group_size);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return out;
    }
    TORCH_CHECK(false, "unknown W4 kernel mode");
}
Tensor rms_cuda(Tensor, Tensor, double) { TORCH_CHECK(false, "Week 2 Day 4: implement RMSNorm CUDA"); }
Tensor rope_cuda(Tensor, Tensor, int64_t, double, bool) { TORCH_CHECK(false, "Week 2 Day 4: implement RoPE CUDA"); }
Tensor swiglu_cuda(Tensor, Tensor) { TORCH_CHECK(false, "Week 2 Day 4: implement SwiGLU CUDA"); }
Tensor attention_cuda(Tensor, Tensor, Tensor, Tensor, double, bool) {
    TORCH_CHECK(false, "Week 2 Day 5: implement tiled CUDA attention");
}
