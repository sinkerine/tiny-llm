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

constexpr int kRmsThreads = 256;
constexpr int kRmsRegisterCap = 4096;
constexpr int kRmsMaxPerThread = kRmsRegisterCap / kRmsThreads;

template <typename scalar_t, bool RegisterCached>
__global__ void rms_norm_kernel(
    const scalar_t* __restrict__ x,
    const scalar_t* __restrict__ weight,
    scalar_t* __restrict__ out,
    int64_t D,
    float eps) {
    int64_t row = static_cast<int64_t>(blockIdx.x);
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;

    const scalar_t* x_row = x + row * D;
    scalar_t* out_row = out + row * D;

    float regs[RegisterCached ? kRmsMaxPerThread : 1];
    float sum_sq = 0.0f;
    if constexpr (RegisterCached) {
        int idx = 0;
        for (int64_t col = tid; col < D; col += kRmsThreads) {
            float xf = static_cast<float>(x_row[col]);
            regs[idx++] = xf;
            sum_sq += xf * xf;
        }
    } else {
        for (int64_t col = tid; col < D; col += kRmsThreads) {
            float xf = static_cast<float>(x_row[col]);
            sum_sq += xf * xf;
        }
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        sum_sq += __shfl_down_sync(0xffffffffu, sum_sq, offset);
    }
    __shared__ float warp_sums[kRmsThreads / 32];
    if (lane == 0) {
        warp_sums[warp_id] = sum_sq;
    }
    __syncthreads();

    if (warp_id == 0) {
        float val = (lane < (kRmsThreads / 32)) ? warp_sums[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffffu, val, offset);
        }
        if (lane == 0) {
            warp_sums[0] = rsqrtf(val / static_cast<float>(D) + eps);
        }
    }
    __syncthreads();

    float inv_rms = warp_sums[0];
    if constexpr (RegisterCached) {
        int idx = 0;
        for (int64_t col = tid; col < D; col += kRmsThreads) {
            scalar_t normed = static_cast<scalar_t>(regs[idx++] * inv_rms);
            out_row[col] = static_cast<scalar_t>(
                static_cast<float>(normed) * static_cast<float>(weight[col]));
        }
    } else {
        for (int64_t col = tid; col < D; col += kRmsThreads) {
            float xf = static_cast<float>(x_row[col]);
            scalar_t normed = static_cast<scalar_t>(xf * inv_rms);
            out_row[col] = static_cast<scalar_t>(
                static_cast<float>(normed) * static_cast<float>(weight[col]));
        }
    }
}

constexpr int kRopeThreads = 256;

template <typename scalar_t>
__global__ void rope_kernel(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ offsets,
    scalar_t* __restrict__ out,
    int64_t L,
    int64_t H,
    int64_t D,
    int64_t total_pairs,
    float base,
    bool traditional) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) {
        return;
    }

    int64_t half_d = D >> 1;
    int64_t j = idx % half_d;
    int64_t tmp = idx / half_d;
    int64_t h = tmp % H;
    tmp /= H;
    int64_t l = tmp % L;
    int64_t b = tmp / L;

    float pos = static_cast<float>(offsets[b] + l);
    float freq = powf(base, -static_cast<float>(j) / static_cast<float>(half_d));
    float angle = pos * freq;
    scalar_t co = static_cast<scalar_t>(cosf(angle));
    scalar_t si = static_cast<scalar_t>(sinf(angle));

    int64_t row_base = ((b * L + l) * H + h) * D;
    int64_t i0 = traditional ? (2 * j) : j;
    int64_t i1 = traditional ? (2 * j + 1) : (j + half_d);

    float uf = static_cast<float>(x[row_base + i0]);
    float vf = static_cast<float>(x[row_base + i1]);
    float cf = static_cast<float>(co);
    float sf = static_cast<float>(si);

    scalar_t u_co = static_cast<scalar_t>(uf * cf);
    scalar_t v_si = static_cast<scalar_t>(vf * sf);
    scalar_t u_si = static_cast<scalar_t>(uf * sf);
    scalar_t v_co = static_cast<scalar_t>(vf * cf);

    out[row_base + i0] = static_cast<scalar_t>(
        static_cast<float>(u_co) - static_cast<float>(v_si));
    out[row_base + i1] = static_cast<scalar_t>(
        static_cast<float>(u_si) + static_cast<float>(v_co));
}

constexpr int kSwigluThreads = 256;

template <typename scalar_t>
__global__ void swiglu_kernel(
    const scalar_t* __restrict__ gate,
    const scalar_t* __restrict__ up,
    scalar_t* __restrict__ out,
    int64_t n) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }

    float g = static_cast<float>(gate[idx]);
    float e = expf(-fabsf(g));
    float silu_f = (g >= 0.0f) ? (g / (1.0f + e)) : (g * e / (1.0f + e));
    scalar_t silu_t = static_cast<scalar_t>(silu_f);
    out[idx] = static_cast<scalar_t>(
        static_cast<float>(silu_t) * static_cast<float>(up[idx]));
}

constexpr int kAttnBq = 4;
constexpr int kAttnBk = 16;
constexpr int kAttnHeadDim = 128;
constexpr int kAttnThreads = kAttnBq * 32;
constexpr int kAttnElemPerLane = kAttnHeadDim / 32;

__global__ void tiled_prefill_attention_kernel(
    const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k,
    const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ mask,
    __nv_bfloat16* __restrict__ out,
    int64_t Hq,
    int64_t Hkv,
    int64_t L,
    int64_t S,
    float scale,
    bool causal) {
    int tid = threadIdx.x;
    int warp_id = tid >> 5;
    int lane = tid & 31;

    int64_t bhq = static_cast<int64_t>(blockIdx.y);
    int64_t b = bhq / Hq;
    int64_t hq = bhq % Hq;
    int64_t hkv = hq / (Hq / Hkv);
    int64_t bhkv = b * Hkv + hkv;

    int64_t q_row = static_cast<int64_t>(blockIdx.x) * kAttnBq + warp_id;
    bool q_valid = q_row < L;

    float q_reg[kAttnElemPerLane] = {0.0f, 0.0f, 0.0f, 0.0f};
    float o_reg[kAttnElemPerLane] = {0.0f, 0.0f, 0.0f, 0.0f};
    float m = -INFINITY;
    float l_sum = 0.0f;

    const __nv_bfloat16* q_ptr = q + (bhq * L + q_row) * kAttnHeadDim;
    if (q_valid) {
#pragma unroll
        for (int i = 0; i < kAttnElemPerLane; ++i) {
            q_reg[i] = __bfloat162float(q_ptr[lane + i * 32]);
        }
    }

    const __nv_bfloat16* k_head = k + bhkv * S * kAttnHeadDim;
    const __nv_bfloat16* v_head = v + bhkv * S * kAttnHeadDim;
    const float* mask_row =
        (mask != nullptr && q_valid) ? (mask + (bhq * L + q_row) * S) : nullptr;

    __shared__ __nv_bfloat16 k_tile[kAttnBk][kAttnHeadDim];
    __shared__ __nv_bfloat16 v_tile[kAttnBk][kAttnHeadDim];

    int64_t max_causal_k = S - L + q_row;

    for (int64_t k_start = 0; k_start < S; k_start += kAttnBk) {
#pragma unroll
        for (int r = 0; r < kAttnBk; ++r) {
            int64_t k_pos = k_start + r;
            if (k_pos < S) {
                k_tile[r][tid] = k_head[k_pos * kAttnHeadDim + tid];
                v_tile[r][tid] = v_head[k_pos * kAttnHeadDim + tid];
            }
        }
        __syncthreads();

        if (q_valid) {
            int tile_len = static_cast<int>(
                (k_start + kAttnBk < S) ? kAttnBk : (S - k_start));
            for (int j = 0; j < tile_len; ++j) {
                int64_t k_pos = k_start + j;
                if (causal && k_pos > max_causal_k) {
                    break;
                }
                float mask_val = 0.0f;
                if (mask_row != nullptr) {
                    mask_val = mask_row[k_pos];
                    if (!isfinite(mask_val)) {
                        continue;
                    }
                }

                float dot = 0.0f;
#pragma unroll
                for (int i = 0; i < kAttnElemPerLane; ++i) {
                    dot += q_reg[i] * __bfloat162float(k_tile[j][lane + i * 32]);
                }
#pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1) {
                    dot += __shfl_xor_sync(0xffffffffu, dot, offset);
                }

                float s = dot * scale + mask_val;
                float new_m = fmaxf(m, s);
                float alpha = (m == -INFINITY) ? 0.0f : expf(m - new_m);
                float p = expf(s - new_m);
                l_sum = alpha * l_sum + p;
#pragma unroll
                for (int i = 0; i < kAttnElemPerLane; ++i) {
                    o_reg[i] = alpha * o_reg[i] +
                               p * __bfloat162float(v_tile[j][lane + i * 32]);
                }
                m = new_m;
            }
        }
        __syncthreads();
    }

    if (q_valid) {
        __nv_bfloat16* out_ptr = out + (bhq * L + q_row) * kAttnHeadDim;
        float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
#pragma unroll
        for (int i = 0; i < kAttnElemPerLane; ++i) {
            out_ptr[lane + i * 32] = __float2bfloat16(o_reg[i] * inv_l);
        }
    }
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
Tensor rms_cuda(Tensor x, Tensor weight, double eps) {
    c10::cuda::CUDAGuard guard(x.device());
    int64_t D = x.size(-1);
    int64_t rows = x.numel() / D;
    Tensor out = at::empty_like(x);
    if (rows == 0) {
        return out;
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    float eps_f = static_cast<float>(eps);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "rms_cuda",
        [&] {
            const auto* x_ptr = x.data_ptr<scalar_t>();
            const auto* w_ptr = weight.data_ptr<scalar_t>();
            auto* out_ptr = out.data_ptr<scalar_t>();
            if (D <= kRmsRegisterCap) {
                rms_norm_kernel<scalar_t, true>
                    <<<static_cast<unsigned int>(rows), kRmsThreads, 0, stream>>>(
                        x_ptr, w_ptr, out_ptr, D, eps_f);
            } else {
                rms_norm_kernel<scalar_t, false>
                    <<<static_cast<unsigned int>(rows), kRmsThreads, 0, stream>>>(
                        x_ptr, w_ptr, out_ptr, D, eps_f);
            }
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
Tensor rope_cuda(
    Tensor x,
    Tensor offsets,
    int64_t dims,
    double base,
    bool traditional) {
    c10::cuda::CUDAGuard guard(x.device());
    int64_t B = x.size(0);
    int64_t L = x.size(1);
    int64_t H = x.size(2);
    int64_t D = x.size(3);
    int64_t total_pairs = B * L * H * (D >> 1);
    Tensor out = at::empty_like(x);
    if (total_pairs == 0) {
        return out;
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    unsigned int grid =
        static_cast<unsigned int>((total_pairs + kRopeThreads - 1) / kRopeThreads);
    float base_f = static_cast<float>(base);
    const int64_t* off_ptr = offsets.data_ptr<int64_t>();

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        x.scalar_type(),
        "rope_cuda",
        [&] {
            rope_kernel<scalar_t><<<grid, kRopeThreads, 0, stream>>>(
                x.data_ptr<scalar_t>(),
                off_ptr,
                out.data_ptr<scalar_t>(),
                L,
                H,
                D,
                total_pairs,
                base_f,
                traditional);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
Tensor swiglu_cuda(Tensor gate, Tensor up) {
    c10::cuda::CUDAGuard guard(gate.device());
    int64_t n = gate.numel();
    Tensor out = at::empty_like(gate);
    if (n == 0) {
        return out;
    }

    auto stream = at::cuda::getCurrentCUDAStream();
    unsigned int grid =
        static_cast<unsigned int>((n + kSwigluThreads - 1) / kSwigluThreads);
    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half,
        at::ScalarType::BFloat16,
        gate.scalar_type(),
        "swiglu_cuda",
        [&] {
            swiglu_kernel<scalar_t><<<grid, kSwigluThreads, 0, stream>>>(
                gate.data_ptr<scalar_t>(),
                up.data_ptr<scalar_t>(),
                out.data_ptr<scalar_t>(),
                n);
        });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
Tensor attention_cuda(
    Tensor q,
    Tensor k,
    Tensor v,
    Tensor mask,
    double scale,
    bool causal) {
    c10::cuda::CUDAGuard guard(q.device());
    int64_t B = q.size(0);
    int64_t Hq = q.size(1);
    int64_t L = q.size(2);
    int64_t Hkv = k.size(1);
    int64_t S = k.size(2);

    if (B == 0 || Hq == 0 || L == 0) {
        return at::empty_like(q);
    }
    if (S == 0) {
        return at::zeros_like(q);
    }

    Tensor out = at::empty_like(q);
    auto stream = at::cuda::getCurrentCUDAStream();
    dim3 block(kAttnThreads);
    dim3 grid(
        static_cast<unsigned int>((L + kAttnBq - 1) / kAttnBq),
        static_cast<unsigned int>(B * Hq));

    const auto* q_ptr =
        reinterpret_cast<const __nv_bfloat16*>(q.data_ptr<at::BFloat16>());
    const auto* k_ptr =
        reinterpret_cast<const __nv_bfloat16*>(k.data_ptr<at::BFloat16>());
    const auto* v_ptr =
        reinterpret_cast<const __nv_bfloat16*>(v.data_ptr<at::BFloat16>());
    const float* mask_ptr =
        (mask.numel() > 0) ? mask.data_ptr<float>() : nullptr;
    auto* out_ptr =
        reinterpret_cast<__nv_bfloat16*>(out.data_ptr<at::BFloat16>());

    tiled_prefill_attention_kernel<<<grid, block, 0, stream>>>(
        q_ptr,
        k_ptr,
        v_ptr,
        mask_ptr,
        out_ptr,
        Hq,
        Hkv,
        L,
        S,
        static_cast<float>(scale),
        causal);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
Tensor paged_cache_update_cuda(Tensor, Tensor, int64_t, int64_t) {
    TORCH_CHECK(false, "Week 3 Day 3: implement CUDA paged cache update");
}
Tensor paged_attention_cuda(
    Tensor, Tensor, Tensor, Tensor, Tensor, int64_t, double, bool) {
    TORCH_CHECK(false, "Week 3 Day 4: implement CUDA paged attention");
}
