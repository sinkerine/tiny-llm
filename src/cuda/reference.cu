// Week 2 CUDA reference kernels. All launches use PyTorch's current stream.
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <mma.h>
using at::Tensor;
using bf16 = __nv_bfloat16;
namespace wmma = nvcuda::wmma;

__device__ float weight_at(const int32_t *w, const bf16 *s, const bf16 *b, int n, int k, int K, int group) {
    unsigned word = static_cast<unsigned>(w[n * (K / 8) + k / 8]);
    float q = (word >> (4 * (k % 8))) & 15;
    int g = n * (K / group) + k / group;
    // Round reconstructed weights to the same BF16 used by the readable control.
    return __bfloat162float(__float2bfloat16(q * __bfloat162float(s[g]) + __bfloat162float(b[g])));
}
__global__ void w4_scalar(const bf16 *x, const int32_t *w, const bf16 *s, const bf16 *b, bf16 *y, int M, int N, int K,
                          int group) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    int m = i / N, n = i % N;
    float sum = 0;
    for (int k = 0; k < K; k++) sum += __bfloat162float(x[m * K + k]) * weight_at(w, s, b, n, k, K, group);
    y[i] = __float2bfloat16(sum);
}
__global__ void w4_matvec(const bf16 *x, const int32_t *w, const bf16 *s, const bf16 *b, bf16 *y, int M, int N, int K,
                          int group) {
    int lane = threadIdx.x % 32, n = blockIdx.x * 4 + threadIdx.x / 32, m = blockIdx.y;
    float sum = 0;
    if (n < N)
        for (int k = lane; k < K; k += 32) sum += __bfloat162float(x[m * K + k]) * weight_at(w, s, b, n, k, K, group);
    for (int d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
    if (lane == 0 && n < N) y[m * N + n] = __float2bfloat16(sum);
}
// One warp cooperatively loads a 16x16 activation/weight tile. No dense W copy.
template <bool FullTiles>
__global__ void w4_wmma(const bf16 *x, const int32_t *w, const bf16 *s, const bf16 *b, bf16 *y, int M, int N, int K,
                        int group) {
    __shared__ bf16 A[256], B[256];
    __shared__ float C[256];
    int m0 = blockIdx.y * 16, n0 = blockIdx.x * 16;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.0f);
    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = threadIdx.x; i < 256; i += 32) {
            int r = i / 16, c = i % 16;
            A[i] = (FullTiles || (m0 + r < M && k0 + c < K)) ? x[(m0 + r) * K + k0 + c] : __float2bfloat16(0);
            // Column-major B stores [K,N] reconstructed from packed [N,K].
            B[i] = (FullTiles || (n0 + r < N && k0 + c < K))
                       ? __float2bfloat16(weight_at(w, s, b, n0 + r, k0 + c, K, group))
                       : __float2bfloat16(0);
        }
        __syncwarp();
        wmma::fragment<wmma::matrix_a, 16, 16, 16, bf16, wmma::row_major> af;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, bf16, wmma::col_major> bf;
        wmma::load_matrix_sync(af, A, 16);
        wmma::load_matrix_sync(bf, B, 16);
        wmma::mma_sync(acc, af, bf, acc);
        __syncwarp();
    }
    wmma::store_matrix_sync(C, acc, 16, wmma::mem_row_major);
    __syncwarp();
    for (int i = threadIdx.x; i < 256; i += 32) {
        int r = i / 16, c = i % 16;
        if (FullTiles || (m0 + r < M && n0 + c < N)) y[(m0 + r) * N + n0 + c] = __float2bfloat16(C[i]);
    }
}
Tensor w4_cuda(Tensor x, Tensor w, Tensor s, Tensor b, int64_t group, int64_t mode) {
    c10::cuda::CUDAGuard guard(x.device());
    int K = x.size(-1), M = x.numel() / K, N = w.size(0);
    auto y = at::empty({M, N}, x.options());
    auto stream = at::cuda::getCurrentCUDAStream();
    if (M && N) {
        auto xp = (bf16 *)x.data_ptr();
        auto wp = w.data_ptr<int32_t>();
        auto sp = (bf16 *)s.data_ptr();
        auto bp = (bf16 *)b.data_ptr();
        auto yp = (bf16 *)y.data_ptr();
        if (mode == 2) {
            if (M % 16 == 0 && N % 16 == 0 && K % 16 == 0)
                w4_wmma<true><<<dim3(N / 16, M / 16), 32, 0, stream>>>(xp, wp, sp, bp, yp, M, N, K, group);
            else
                w4_wmma<false>
                    <<<dim3((N + 15) / 16, (M + 15) / 16), 32, 0, stream>>>(xp, wp, sp, bp, yp, M, N, K, group);
        } else if (mode == 1)
            w4_matvec<<<dim3((N + 3) / 4, M), 128, 0, stream>>>(xp, wp, sp, bp, yp, M, N, K, group);
        else
            w4_scalar<<<(M * N + 127) / 128, 128, 0, stream>>>(xp, wp, sp, bp, yp, M, N, K, group);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return y;
}

template <class T>
__global__ void rms_kernel(const T *x, const T *w, T *y, int D, float eps) {
    __shared__ float sums[256];
    int row = blockIdx.x, t = threadIdx.x;
    // Register cache for <=4096 elements; wide rows reread after reduction.
    float vals[16], sum = 0;
    for (int d = t, j = 0; d < D; d += 256, j++) {
        float v = (float)x[row * D + d];
        if (D <= 4096) vals[j] = v;
        sum += v * v;
    }
    sums[t] = sum;
    __syncthreads();
    for (int stride = 128; stride; stride /= 2) {
        if (t < stride) sums[t] += sums[t + stride];
        __syncthreads();
    }
    float inv = rsqrtf(sums[0] / D + eps);
    for (int d = t, j = 0; d < D; d += 256, j++) {
        float v = D <= 4096 ? vals[j] : (float)x[row * D + d];
        // Match Week 1: round normalized activation before learned scale.
        y[row * D + d] = (T)((float)(T)(v * inv) * (float)w[d]);
    }
}
Tensor rms_cuda(Tensor x, Tensor w, double eps) {
    c10::cuda::CUDAGuard guard(x.device());
    auto y = at::empty_like(x);
    int D = x.size(-1);
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rms", [&] {
        rms_kernel<<<x.numel() / D, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), D, eps);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
template <class T>
__global__ void rope_kernel(const T *x, const int64_t *offsets, T *y, int L, int H, int D, int dims, float base,
                            bool traditional, int count) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    int pair = i % (D / 2), h = (i / (D / 2)) % H, l = (i / (D / 2) / H) % L, b = i / (D / 2) / H / L;
    int start = ((b * L + l) * H + h) * D;
    if (pair >= dims / 2) {
        y[start + pair] = x[start + pair];
        y[start + pair + D / 2] = x[start + pair + D / 2];
        return;
    }
    int a = traditional ? 2 * pair : pair, c = traditional ? 2 * pair + 1 : pair + dims / 2;
    float angle = (offsets[b] + l) * powf(base, -2.0f * pair / dims);
    T co = (T)cosf(angle), si = (T)sinf(angle), u = x[start + a], v = x[start + c];
    // Preserve BF16 multiply/add rounding of the readable RoPE.
    y[start + a] = (T)((float)(T)((float)u * (float)co) - (float)(T)((float)v * (float)si));
    y[start + c] = (T)((float)(T)((float)u * (float)si) + (float)(T)((float)v * (float)co));
}
Tensor rope_cuda(Tensor x, Tensor offsets, int64_t dims, double base, bool traditional) {
    c10::cuda::CUDAGuard guard(x.device());
    auto y = x.clone();
    int n = x.numel() / 2;
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, x.scalar_type(), "rope", [&] {
        rope_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            x.data_ptr<scalar_t>(), offsets.data_ptr<int64_t>(), y.data_ptr<scalar_t>(), x.size(1), x.size(2),
            x.size(3), dims, base, traditional, n);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
template <class T>
__global__ void swiglu_kernel(const T *g, const T *u, T *y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = (float)g[i];
        float e = expf(-fabsf(v));
        float silu = v >= 0 ? v / (1 + e) : v * e / (1 + e);
        y[i] = (T)((float)(T)silu * (float)u[i]);
    }
}
Tensor swiglu_cuda(Tensor g, Tensor u) {
    c10::cuda::CUDAGuard guard(g.device());
    auto y = at::empty_like(g);
    int n = g.numel();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf, at::kBFloat16, g.scalar_type(), "swiglu", [&] {
        swiglu_kernel<<<(n + 255) / 256, 256, 0, at::cuda::getCurrentCUDAStream()>>>(
            g.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(), y.data_ptr<scalar_t>(), n);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}
// D128 online softmax. A warp owns one query row; 4 warps/block form BQ4.
// K/V are staged in BK16 shared tiles and reused across query rows.
__global__ void attention_kernel(const bf16 *q, const bf16 *k, const bf16 *v, const float *mask, bf16 *out, int H,
                                 int HK, int L, int S, float scale, bool causal, bool has_mask) {
    __shared__ bf16 kt[16 * 128], vt[16 * 128];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    int row = blockIdx.x * 4 + warp, h = blockIdx.y % H, b = blockIdx.y / H, kh = h / (H / HK);
    float qv[4], o[4] = {0, 0, 0, 0};
    for (int j = 0; j < 4; j++)
        qv[j] = row < L ? __bfloat162float(q[((b * H + h) * L + row) * 128 + lane + 32 * j]) : 0;
    float maxv = -INFINITY, den = 0;
    for (int s0 = 0; s0 < S; s0 += 16) {
        for (int i = threadIdx.x; i < 16 * 128; i += 128) {
            int t = i / 128, d = i % 128;
            kt[i] = s0 + t < S ? k[((b * HK + kh) * S + s0 + t) * 128 + d] : __float2bfloat16(0);
            vt[i] = s0 + t < S ? v[((b * HK + kh) * S + s0 + t) * 128 + d] : __float2bfloat16(0);
        }
        __syncthreads();
        for (int t = 0; t < 16 && s0 + t < S; t++) {
            int s = s0 + t;
            float score = 0;
            for (int j = 0; j < 4; j++) score += qv[j] * __bfloat162float(kt[t * 128 + lane + 32 * j]);
            for (int d = 16; d; d /= 2) score += __shfl_down_sync(0xffffffff, score, d);
            score = __shfl_sync(0xffffffff, score, 0) * scale;
            if (row >= L || (causal && s > row + S - L)) score = -INFINITY;
            if (has_mask && row < L) score += mask[((b * H + h) * L + row) * S + s];
            if (score != -INFINITY) {
                float next = fmaxf(maxv, score), alpha = expf(maxv - next), p = expf(score - next);
                den = den * alpha + p;
                for (int j = 0; j < 4; j++) o[j] = o[j] * alpha + p * __bfloat162float(vt[t * 128 + lane + 32 * j]);
                maxv = next;
            }
        }
        __syncthreads();
    }
    if (row < L)
        for (int j = 0; j < 4; j++)
            out[((b * H + h) * L + row) * 128 + lane + 32 * j] = __float2bfloat16(den ? o[j] / den : 0);
}
Tensor attention_cuda(Tensor q, Tensor k, Tensor v, Tensor mask, double scale, bool causal) {
    c10::cuda::CUDAGuard guard(q.device());
    auto y = at::empty_like(q);
    attention_kernel<<<dim3((q.size(2) + 3) / 4, q.size(0) * q.size(1)), 128, 0, at::cuda::getCurrentCUDAStream()>>>(
        (bf16 *)q.data_ptr(), (bf16 *)k.data_ptr(), (bf16 *)v.data_ptr(),
        mask.numel() ? mask.data_ptr<float>() : nullptr, (bf16 *)y.data_ptr(), q.size(1), k.size(1), q.size(2),
        k.size(2), scale, causal, mask.numel() > 0);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return y;
}

// Upstream selected reference optimization: stride-aware partitioned online decode.
__global__ void decode_online_kernel(const bf16 *q, const bf16 *k, const bf16 *v, const float *mask, float *partials,
                                     bf16 *out, int H, int HK, int S, int partitions, int64_t kb, int64_t kh,
                                     int64_t ks, int64_t kd, int64_t vb, int64_t vh, int64_t vs, int64_t vd,
                                     float scale) {
    int lane = threadIdx.x, head = blockIdx.y % H, batch = blockIdx.y / H;
    int kv = head / (H / HK), part = blockIdx.x;
    float query[4], result[4] = {0, 0, 0, 0};
    for (int j = 0; j < 4; j++) query[j] = __bfloat162float(q[(batch * H + head) * 128 + lane + 32 * j]);
    float maximum = -INFINITY, denominator = 0;
    for (int s = part * 256; s < min(S, (part + 1) * 256); s++) {
        float score = 0;
        for (int j = 0; j < 4; j++)
            score += query[j] * __bfloat162float(k[batch * kb + kv * kh + s * ks + (lane + 32 * j) * kd]);
        for (int d = 16; d; d /= 2) score += __shfl_down_sync(0xffffffff, score, d);
        score = __shfl_sync(0xffffffff, score, 0) * scale;
        if (mask) score += mask[(batch * H + head) * S + s];
        if (score != -INFINITY) {
            float next = fmaxf(maximum, score), alpha = expf(maximum - next), p = expf(score - next);
            denominator = denominator * alpha + p;
            for (int j = 0; j < 4; j++)
                result[j] =
                    result[j] * alpha + p * __bfloat162float(v[batch * vb + kv * vh + s * vs + (lane + 32 * j) * vd]);
            maximum = next;
        }
    }
    int row = (batch * H + head);
    if (partitions == 1) {
        for (int j = 0; j < 4; j++)
            out[row * 128 + lane + 32 * j] = __float2bfloat16(denominator ? result[j] / denominator : 0);
    } else {
        int base = (row * partitions + part) * 130;
        if (lane == 0) {
            partials[base] = maximum;
            partials[base + 1] = denominator;
        }
        for (int j = 0; j < 4; j++) partials[base + 2 + lane + 32 * j] = result[j];
    }
}

// Four query heads share a staged K/V tile on the long-context GQA4 route.
__global__ void decode_grouped_kernel(const bf16 *q, const bf16 *k, const bf16 *v, const float *mask, float *partials,
                                      int H, int HK, int S, int partitions, int64_t kb, int64_t kh, int64_t ks,
                                      int64_t kd, int64_t vb, int64_t vh, int64_t vs, int64_t vd, float scale) {
    __shared__ bf16 kt[16 * 128], vt[16 * 128];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32, kv = blockIdx.y % HK, batch = blockIdx.y / HK;
    int head = kv * 4 + warp, part = blockIdx.x;
    float query[4], result[4] = {0, 0, 0, 0};
    for (int j = 0; j < 4; j++) query[j] = __bfloat162float(q[(batch * H + head) * 128 + lane + 32 * j]);
    float maximum = -INFINITY, denominator = 0;
    for (int start = part * 256; start < min(S, (part + 1) * 256); start += 16) {
        for (int i = threadIdx.x; i < 16 * 128; i += 128) {
            int t = i / 128, d = i % 128;
            kt[i] = start + t < S ? k[batch * kb + kv * kh + (start + t) * ks + d * kd] : __float2bfloat16(0);
            vt[i] = start + t < S ? v[batch * vb + kv * vh + (start + t) * vs + d * vd] : __float2bfloat16(0);
        }
        __syncthreads();
        for (int t = 0; t < 16 && start + t < S; t++) {
            float score = 0;
            for (int j = 0; j < 4; j++) score += query[j] * __bfloat162float(kt[t * 128 + lane + 32 * j]);
            for (int d = 16; d; d /= 2) score += __shfl_down_sync(0xffffffff, score, d);
            score = __shfl_sync(0xffffffff, score, 0) * scale;
            if (mask) score += mask[(batch * H + head) * S + start + t];
            if (score != -INFINITY) {
                float next = fmaxf(maximum, score), alpha = expf(maximum - next), p = expf(score - next);
                denominator = denominator * alpha + p;
                for (int j = 0; j < 4; j++)
                    result[j] = result[j] * alpha + p * __bfloat162float(vt[t * 128 + lane + 32 * j]);
                maximum = next;
            }
        }
        __syncthreads();
    }
    int base = ((batch * H + head) * partitions + part) * 130;
    if (lane == 0) {
        partials[base] = maximum;
        partials[base + 1] = denominator;
    }
    for (int j = 0; j < 4; j++) partials[base + 2 + lane + 32 * j] = result[j];
}

__global__ void decode_merge_kernel(const float *partials, bf16 *out, int P) {
    int lane = threadIdx.x, row = blockIdx.x;
    float maximum = -INFINITY, denominator = 0, result[4] = {0, 0, 0, 0};
    for (int p = 0; p < P; p++) {
        int base = (row * P + p) * 130;
        float pm = partials[base], pl = partials[base + 1];
        if (pl > 0) {
            float next = fmaxf(maximum, pm), a = expf(maximum - next), b = expf(pm - next);
            denominator = a * denominator + b * pl;
            for (int j = 0; j < 4; j++) result[j] = a * result[j] + b * partials[base + 2 + lane + 32 * j];
            maximum = next;
        }
    }
    for (int j = 0; j < 4; j++)
        out[row * 128 + lane + 32 * j] = __float2bfloat16(denominator ? result[j] / denominator : 0);
}

Tensor decode_cuda(Tensor q, Tensor k, Tensor v, Tensor mask, double scale) {
    c10::cuda::CUDAGuard guard(q.device());
    auto out = at::empty_like(q);
    int S = k.size(2), H = q.size(1), HK = k.size(1), B = q.size(0), P = (S + 255) / 256;
    auto partials = at::empty({B, H, P, 130}, q.options().dtype(at::kFloat));
    auto stream = at::cuda::getCurrentCUDAStream();
    auto qp = (bf16 *)q.data_ptr(), kp = (bf16 *)k.data_ptr(), vp = (bf16 *)v.data_ptr(), yp = (bf16 *)out.data_ptr();
    const float *mp = mask.numel() ? mask.data_ptr<float>() : nullptr;
    if (S >= 8192 && H == 4 * HK) {
        decode_grouped_kernel<<<dim3(P, B * HK), 128, 0, stream>>>(
            qp, kp, vp, mp, partials.data_ptr<float>(), H, HK, S, P, k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3), scale);
    } else {
        decode_online_kernel<<<dim3(P, B * H), 32, 0, stream>>>(
            qp, kp, vp, mp, partials.data_ptr<float>(), yp, H, HK, S, P, k.stride(0), k.stride(1), k.stride(2),
            k.stride(3), v.stride(0), v.stride(1), v.stride(2), v.stride(3), scale);
    }
    if (P > 1) decode_merge_kernel<<<B * H, 32, 0, stream>>>(partials.data_ptr<float>(), yp, P);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}

// Explicit CUDA equivalent of MLX's deferred RMSNorm -> RoPE graph fusion.
__global__ void qk_norm_rope_kernel(const bf16 *x, const bf16 *w, const int64_t *offsets, bf16 *y, float eps,
                                    float base) {
    int lane = threadIdx.x, head = blockIdx.x;
    float values[4], sum = 0;
    for (int j = 0; j < 4; j++) {
        values[j] = __bfloat162float(x[head * 128 + lane + 32 * j]);
        sum += values[j] * values[j];
    }
    for (int d = 16; d; d /= 2) sum += __shfl_down_sync(0xffffffff, sum, d);
    float inv = rsqrtf(__shfl_sync(0xffffffff, sum, 0) / 128 + eps);
    bf16 norm[4];
    for (int j = 0; j < 4; j++)
        norm[j] =
            __float2bfloat16(__bfloat162float(__float2bfloat16(values[j] * inv)) * __bfloat162float(w[lane + 32 * j]));
    for (int j = 0; j < 2; j++) {
        int pair = lane + 32 * j;
        float angle = offsets[0] * powf(base, -2.0f * pair / 128);
        float co = __bfloat162float(__float2bfloat16(cosf(angle))),
              si = __bfloat162float(__float2bfloat16(sinf(angle)));
        float u = __bfloat162float(norm[j]), v = __bfloat162float(norm[j + 2]);
        y[head * 128 + pair] =
            __float2bfloat16(__bfloat162float(__float2bfloat16(u * co)) - __bfloat162float(__float2bfloat16(v * si)));
        y[head * 128 + pair + 64] =
            __float2bfloat16(__bfloat162float(__float2bfloat16(u * si)) + __bfloat162float(__float2bfloat16(v * co)));
    }
}
Tensor qk_norm_rope_cuda(Tensor x, Tensor w, Tensor offsets, double eps, double base) {
    c10::cuda::CUDAGuard guard(x.device());
    auto out = at::empty_like(x);
    qk_norm_rope_kernel<<<x.size(2), 32, 0, at::cuda::getCurrentCUDAStream()>>>(
        (bf16 *)x.data_ptr(), (bf16 *)w.data_ptr(), offsets.data_ptr<int64_t>(), (bf16 *)out.data_ptr(), eps, base);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
