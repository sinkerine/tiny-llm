import torch
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if scale is None:
        scale = 1 / query.shape[-1] ** 0.5
    scores = torch.einsum("...qd , ...kd -> ...qk", query, key) * scale
    if mask is not None:
        scores += mask
    attention = softmax(scores, -1)
    return torch.einsum("...qk , ...kd -> ...qd", attention, value)


class SimpleMultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
    ):
        pass

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pass


def causal_mask(L: int, S: int, dtype: torch.dtype) -> torch.Tensor:
    pass


def scaled_dot_product_attention_grouped(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    pass


def paged_attention(
    query: torch.Tensor,
    key_pages: torch.Tensor,
    value_pages: torch.Tensor,
    block_table: torch.Tensor,
    context_lens: torch.Tensor,
    page_size: int,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    """Attend to paged K/V storage without reconstructing a dense cache.

    Week 3 Day 4 owns the correctness-first decode and long-prefill paths for
    both float32 and BF16. Day 5 optimizes the same public boundary.
    """
    pass
