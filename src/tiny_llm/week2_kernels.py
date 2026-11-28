import torch
from . import cuda_ops


DENSE_PREFILL_MIN_QUERY = 9


class FastRMSNorm:
    def __init__(self, dim: int, weight: torch.Tensor, eps: float = 1e-5):
        self.dim = dim
        self.weight = weight
        self.eps = eps
        self.dispatch_counts = {
            "register_cached": 0,
            "fixed_width_fallback": 0,
        }

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] <= 4096:
            self.dispatch_counts["register_cached"] += 1
        else:
            self.dispatch_counts["fixed_width_fallback"] += 1
        return cuda_ops.rms_norm(x.contiguous(), self.weight.contiguous(), self.eps)


class FastRoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        self.dims = dims
        self.seq_len = seq_len
        self.base = base
        self.traditional = traditional

    def __call__(
        self, x: torch.Tensor, offset: int | list[int] | torch.Tensor = 0
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"FastRoPE expects 4D [B,L,H,D] input, got shape {tuple(x.shape)}")
        batch = x.shape[0]
        if isinstance(offset, int):
            offsets = torch.full((batch,), offset, dtype=torch.long, device=x.device)
        else:
            t = torch.as_tensor(offset, dtype=torch.long, device=x.device)
            if t.ndim == 0:
                offsets = t.expand(batch).contiguous()
            elif t.ndim == 1 and t.shape[0] == batch:
                offsets = t.contiguous()
            else:
                raise ValueError(
                    f"FastRoPE expects scalar or [{batch}] offsets, got shape {tuple(t.shape)}"
                )
        return cuda_ops.rope(
            x.contiguous(), offsets, self.dims, float(self.base), self.traditional
        )


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return cuda_ops.swiglu(gate.contiguous(), up.contiguous())


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    pass


def decode_attention_custom(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    # The bounded decode experiment is not learner-owned. Keep this compatibility
    # name on the readable path for Week 3 callers.
    return scaled_dot_product_attention(query, key, value, scale, mask)


def dense_prefill_attention_mma(query, key, value, scale, mask=None):
    if query.ndim != 4 or key.ndim != 4 or value.ndim != 4 or key.shape != value.shape:
        raise ValueError(
            "dense_prefill_attention_mma expects 4D Q [B,Hq,L,128] and matching K/V [B,Hkv,S,128]"
        )
    batch, num_heads, query_len, head_dim = query.shape
    key_batch, num_kv_heads, seq_len, key_head_dim = key.shape
    if (
        batch != key_batch
        or head_dim != 128
        or key_head_dim != 128
        or num_kv_heads == 0
        or num_heads % num_kv_heads != 0
    ):
        raise ValueError("incompatible GQA or head_dim!=128 shape for tiled attention")
    if query_len < DENSE_PREFILL_MIN_QUERY:
        raise ValueError(
            f"tiled prefill attention requires L >= {DENSE_PREFILL_MIN_QUERY}, got {query_len}"
        )
    if (
        query.dtype != torch.bfloat16
        or key.dtype != torch.bfloat16
        or value.dtype != torch.bfloat16
    ):
        raise ValueError("tiled attention requires BF16 Q, K, and V")
    if not (query.is_cuda and key.is_cuda and value.is_cuda) or not (
        query.device == key.device == value.device
    ):
        raise ValueError("tiled attention requires Q, K, and V on the same CUDA device")

    causal = False
    if mask is None:
        mask_tensor = torch.empty(0, dtype=torch.float32, device=query.device)
    elif isinstance(mask, str):
        if mask != "causal":
            raise ValueError(f"unsupported attention mask string: {mask!r}")
        causal = True
        mask_tensor = torch.empty(0, dtype=torch.float32, device=query.device)
    elif isinstance(mask, torch.Tensor):
        if not mask.is_floating_point():
            raise ValueError("additive attention mask must be a floating-point tensor")
        try:
            mask_tensor = torch.broadcast_to(
                mask, (batch, num_heads, query_len, seq_len)
            )
        except RuntimeError as exc:
            raise ValueError(
                f"attention mask shape {tuple(mask.shape)} is not broadcastable to "
                f"{(batch, num_heads, query_len, seq_len)}"
            ) from exc
        mask_tensor = mask_tensor.to(
            device=query.device, dtype=torch.float32
        ).contiguous()
    else:
        raise ValueError(f"unsupported attention mask type: {type(mask).__name__}")

    return cuda_ops.attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        mask_tensor,
        float(scale),
        causal,
    )
