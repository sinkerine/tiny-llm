"""Checked native boundary for learner-owned kernels; no reference fallback."""

from functools import lru_cache
import torch
from tiny_llm_cuda.build import load_extension


@lru_cache(None)
def extension():
    return load_extension(reference=False)


def check(*xs):
    if not xs or any(not x.is_cuda for x in xs):
        raise ValueError("native Week 2 kernels require CUDA tensors")
    if any(x.device != xs[0].device for x in xs):
        raise ValueError("all tensors must use the same CUDA device")
    if any(not x.is_contiguous() for x in xs):
        raise ValueError("native inputs must be contiguous")


def w4(x, packed, scales, biases, group_size, mode=0):
    check(x, packed, scales, biases)
    if x.ndim < 2 or packed.ndim != 2 or x.shape[-1] == 0:
        raise ValueError("W4 expects [...,K] activations and [N,K/8] packed weights")
    k = x.shape[-1]
    if group_size <= 0 or k % group_size or k % 8:
        raise ValueError("K must be divisible by group_size and 8")
    if (
        packed.shape[1] * 8 != k
        or scales.shape != (packed.shape[0], k // group_size)
        or biases.shape != scales.shape
    ):
        raise ValueError("incompatible W4 shapes")
    if (
        x.dtype != torch.bfloat16
        or scales.dtype != x.dtype
        or biases.dtype != x.dtype
        or packed.dtype != torch.int32
    ):
        raise ValueError(
            "W4 uses BF16 activations/scales/biases and int32 packed words"
        )
    if mode not in (0, 1, 2):
        raise ValueError("unknown W4 kernel mode")
    result = extension().w4(x, packed, scales, biases, group_size, mode)
    return result.reshape(*x.shape[:-1], packed.shape[0])


def rms_norm(x, weight, eps):
    check(x, weight)
    if (
        x.ndim < 1
        or x.shape[-1] == 0
        or weight.shape != (x.shape[-1],)
        or weight.dtype != x.dtype
        or x.dtype not in (torch.float32, torch.float16, torch.bfloat16)
        or eps <= 0
    ):
        raise ValueError("invalid RMSNorm shape, dtype or epsilon")
    return extension().rms_norm(x, weight, eps)


def rope(x, offsets, dims, base, traditional=False):
    check(x, offsets)
    if (
        x.ndim != 4
        or dims != x.shape[-1]
        or dims % 2
        or offsets.shape != (x.shape[0],)
        or offsets.dtype != torch.long
        or base <= 0
        or x.dtype not in (torch.float32, torch.float16, torch.bfloat16)
    ):
        raise ValueError("RoPE requires [B,L,H,D], full even D, and int64 [B] offsets")
    return extension().rope(x, offsets, dims, base, traditional)


def swiglu(gate, up):
    check(gate, up)
    if (
        gate.shape != up.shape
        or gate.dtype != up.dtype
        or gate.dtype not in (torch.float32, torch.float16, torch.bfloat16)
    ):
        raise ValueError("SwiGLU shape/dtype mismatch")
    return extension().swiglu(gate, up)


def attention(q, k, v, mask, scale, causal):
    check(q, k, v, mask)
    if (
        q.ndim != 4
        or k.ndim != 4
        or v.shape != k.shape
        or q.shape[0] != k.shape[0]
        or q.shape[-1] != 128
        or k.shape[-1] != 128
        or k.shape[1] == 0
        or q.shape[1] % k.shape[1]
        or q.dtype != torch.bfloat16
        or k.dtype != q.dtype
        or v.dtype != q.dtype
    ):
        raise ValueError("tiled attention requires BF16 [B,H,L,128] with valid GQA")
    if mask.dtype != torch.float32 or (
        mask.numel() and mask.shape != (*q.shape[:-1], k.shape[2])
    ):
        raise ValueError("invalid additive attention mask")
    return extension().attention(q, k, v, mask, scale, causal)
