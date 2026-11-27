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
    # Day 5: validate BF16/D128 inputs, then call cuda_ops.attention.
    raise NotImplementedError("Week 2 Day 5: implement tiled CUDA attention wrapper")
