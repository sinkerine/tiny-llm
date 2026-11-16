"""Week 2 native primitives and readable attention control."""

import torch
from . import cuda_ops

DENSE_PREFILL_MIN_QUERY = 9


class FastRMSNorm:
    def __init__(self, dim, weight, eps=1e-5):
        self.dim, self.weight, self.eps = dim, weight, eps
        self.dispatch_counts = {"register_cached": 0, "fixed_width_fallback": 0}

    def __call__(self, x):
        key = "register_cached" if x.shape[-1] <= 4096 else "fixed_width_fallback"
        self.dispatch_counts[key] += 1
        return cuda_ops.rms_norm(
            x.contiguous(), self.weight.to(x.dtype).contiguous(), self.eps
        )


class ReadableRoPE:
    def __init__(self, dims, seq_len, base=10000, traditional=False):
        self.dims, self.base, self.traditional = dims, base, traditional

    def __call__(self, x, offset=0):
        if isinstance(offset, slice):
            offset = offset.start
        elif isinstance(offset, list) and offset and isinstance(offset[0], slice):
            offset = [s.start for s in offset]
        offsets = torch.as_tensor(offset, device=x.device).reshape(-1, 1)
        positions = offsets + torch.arange(x.shape[1], device=x.device)[None]
        freq = self.base ** (
            -torch.arange(self.dims // 2, device=x.device).float() / (self.dims // 2)
        )
        angles = positions.float()[..., None] * freq
        co, si = (
            angles.cos().to(x.dtype)[:, :, None],
            angles.sin().to(x.dtype)[:, :, None],
        )
        if self.traditional:
            u, v = x[..., ::2], x[..., 1::2]
            return torch.stack((u * co - v * si, u * si + v * co), -1).flatten(-2)
        u, v = x.chunk(2, -1)
        return torch.cat((u * co - v * si, u * si + v * co), -1)


class FastRoPE:
    def __init__(self, dims, seq_len, base=10000, traditional=False):
        self.dims, self.seq_len, self.base, self.traditional = (
            dims,
            seq_len,
            base,
            traditional,
        )

    def __call__(self, x, offset=0):
        offsets = torch.as_tensor(offset, device=x.device, dtype=torch.long)
        if offsets.ndim == 0:
            offsets = offsets.expand(x.shape[0])
        if offsets.shape != (x.shape[0],):
            raise ValueError("FastRoPE needs one offset per batch row")
        return cuda_ops.rope(
            x.contiguous(), offsets.contiguous(), self.dims, self.base, self.traditional
        )


def swiglu(gate, up):
    return cuda_ops.swiglu(gate.contiguous(), up.contiguous())


def scaled_dot_product_attention(query, key, value, scale, mask=None):
    if isinstance(mask, str):
        if mask != "causal":
            raise ValueError("unsupported attention mask")
        L, S = query.shape[-2], key.shape[-2]
        mask = (
            torch.arange(S, device=query.device)[None, :]
            <= torch.arange(L, device=query.device)[:, None] + S - L
        )
    return torch.nn.functional.scaled_dot_product_attention(
        query, key, value, attn_mask=mask, scale=scale, enable_gqa=True
    )


def dense_prefill_attention_mma(query, key, value, scale, mask=None):
    if query.ndim != 4 or query.shape[-2] < DENSE_PREFILL_MIN_QUERY:
        raise ValueError("tiled prefill requires [B,H,L,D], L>=9")
    causal = isinstance(mask, str) and mask == "causal"
    if isinstance(mask, str) and not causal:
        raise ValueError("unsupported attention mask")
    if isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            raise ValueError("native attention mask must be additive")
        mask = (
            torch.broadcast_to(mask, (*query.shape[:-1], key.shape[-2]))
            .float()
            .contiguous()
        )
    else:
        mask = torch.empty(0, device=query.device, dtype=torch.float32)
    return cuda_ops.attention(
        query.contiguous(), key.contiguous(), value.contiguous(), mask, scale, causal
    )


def decode_attention_online(query, key, value, scale, mask=None):
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4:
        raise ValueError("online decode expects [B,H,L,D]")
    if isinstance(mask, str) and mask != "causal":
        raise ValueError("unsupported online decode mask")
    if isinstance(mask, torch.Tensor):
        if mask.dtype == torch.bool:
            raise ValueError("online decode expects an additive mask")
        mask = (
            torch.broadcast_to(mask, (*query.shape[:-1], key.shape[-2]))
            .float()
            .contiguous()
        )
    elif mask is None or isinstance(mask, str):
        mask = torch.empty(0, device=query.device, dtype=torch.float32)
    else:
        raise ValueError("unsupported online decode mask type")
    return cuda_ops.decode_online(query.contiguous(), key, value, mask, scale)


def _qk_norm_rope4(normalized, rope, offset=0):
    # A PyTorch tensor is already materialized: preserve the upstream fallback.
    return rope(normalized, offset)


def qk_norm_rope(x, norm, rope, offset=0):
    # CUDA must fuse explicitly before normalization, not inspect a deferred graph.
    eligible = (
        type(norm) is FastRMSNorm
        and type(rope) is FastRoPE
        and x.shape in ((1, 1, 8, 128), (1, 1, 32, 128))
        and x.dtype == torch.bfloat16
        and rope.dims == 128
        and not rope.traditional
    )
    if not eligible:
        return rope(norm(x), offset)
    offsets = torch.as_tensor(offset, device=x.device, dtype=torch.long).reshape(-1)
    if offsets.shape != (1,):
        raise ValueError("fused RoPE needs one offset")
    norm.dispatch_counts["register_cached"] += 1
    return cuda_ops.qk_norm_rope(
        x.contiguous(),
        norm.weight.to(x.dtype).contiguous(),
        offsets.contiguous(),
        norm.eps,
        rope.base,
    )


def decode_attention_custom(query, key, value, scale, mask=None):
    return scaled_dot_product_attention(query, key, value, scale, mask)
