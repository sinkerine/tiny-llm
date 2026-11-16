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
        pass


class FastRoPE:
    def __init__(
        self,
        dims: int,
        seq_len: int,
        base: int = 10000,
        traditional: bool = False,
    ):
        pass

    def __call__(
        self, x: torch.Tensor, offset: int | list[int] | torch.Tensor = 0
    ) -> torch.Tensor:
        pass


def swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    pass


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
