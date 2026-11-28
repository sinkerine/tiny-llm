import torch
from .basics import softmax, linear


def scaled_dot_product_attention_simple(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    A simple implementation of scaled dot product attention. Assuming Q, K, V are of the same shape.
    Assuming mask is always a float array that you can add to the scores.
    """
    factor = query.shape[-1] ** -0.5 if scale is None else scale
    scores = torch.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        scores = scores + mask
    return torch.matmul(softmax(scores, axis=-1), value)


def causal_mask(L: int, S: int, dtype: torch.dtype) -> torch.Tensor:
    mask = torch.tril(torch.ones((L, S), dtype=torch.bool), diagonal=S - L)
    mask = torch.where(mask, 0.0, -torch.inf).to(dtype)
    return mask


def scaled_dot_product_attention_grouped(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    scale: float | None = None,
    mask: torch.Tensor | str | None = None,
) -> torch.Tensor:
    """
    Potential input of the mask:
    - torch.Tensor that can broadcast to B * H_q * L * S, which needs to be reshaped to match multi-head dimensions
    - None which will be ignored
    """
    factor = query.shape[-1] ** -0.5 if scale is None else scale
    expected_shape = query.shape

    H_q, L, D = query.shape[-3:]
    H, S, _ = key.shape[-3:]
    B = query.shape[:-3]
    assert H_q % H == 0
    n_repeats = H_q // H

    query = query.reshape(*B, -1, H, n_repeats, L, D)
    key = key.reshape(*B, -1, H, 1, S, D)
    value = value.reshape(*B, -1, H, 1, S, D)

    scores = torch.matmul(query, key.swapaxes(-2, -1)) * factor
    if mask is not None:
        if isinstance(mask, str) and mask == "causal":
            if L > S:
                raise ValueError("causal attention requires S >= L")
            mask = causal_mask(L, S, scores.dtype).to(scores.device)
            scores = scores + mask
        else:
            mask = torch.broadcast_to(mask, (*B, H_q, L, S))
            mask = mask.reshape(*B, 1, H, n_repeats, L, S)
            scores = scores + mask
    result = torch.matmul(softmax(scores, axis=-1), value)
    return result.reshape(expected_shape)


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
    """
    Paged attention backed by native CUDA kernels.

    Keeps the model-facing shape as [B, H_q, L, D]. Handles both float32 and
    BF16 inputs, direct decode/generic prefill (Day 4), and tiled BF16 D=128
    long prefill (Day 5) behind this stable boundary.
    """
    if isinstance(mask, torch.Tensor):
        raise NotImplementedError("Paged attention only supports mask=None or causal")
    if mask is not None and mask != "causal":
        raise NotImplementedError

    if query.ndim != 4:
        raise ValueError("query must be 4D [B, H_q, L, D]")
    if key_pages.ndim != 4 or value_pages.ndim != 4:
        raise ValueError("page tensors must be 4D [P, H_kv, page_size, D]")
    if key_pages.shape != value_pages.shape:
        raise ValueError("key pages and value pages must have the same shape")
    if block_table.ndim != 2 or context_lens.ndim != 1:
        raise ValueError("block_table must be 2D and context_lens must be 1D")
    if block_table.dtype != torch.int32 or context_lens.dtype != torch.int32:
        raise ValueError("block_table and context_lens must be int32")
    if not isinstance(page_size, int) or isinstance(page_size, bool) or page_size <= 0:
        raise ValueError("page_size must be a positive integer")

    factor = query.shape[-1] ** -0.5 if scale is None else float(scale)
    B, H_q, L, D = query.shape
    num_physical_pages, H, stored_page_size, stored_dim = key_pages.shape
    if min(B, H_q, L, D, H, stored_page_size, stored_dim) <= 0:
        raise ValueError("paged attention dimensions must be positive")
    if num_physical_pages <= 0:
        raise ValueError("paged attention requires nonempty physical page storage")
    if H_q % H != 0:
        raise ValueError("query heads must be divisible by K/V heads")
    if stored_dim != D:
        raise ValueError("query and page tensors must have the same head dimension")
    if stored_page_size != page_size:
        raise ValueError(
            f"page_size={page_size} does not match page storage {stored_page_size}"
        )
    if block_table.shape[0] != B or context_lens.shape[0] != B:
        raise ValueError("query, block_table, and context_lens batch sizes must match")
    max_pages = block_table.shape[1]
    if max_pages <= 0:
        raise ValueError("block_table must provide at least one page slot")

    if query.dtype != key_pages.dtype or query.dtype != value_pages.dtype:
        raise ValueError("query, key pages, and value pages must have the same dtype")
    if query.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError("paged attention supports float32 or bfloat16 inputs")

    context_values = context_lens.tolist()
    block_rows = block_table.tolist()
    live_page_ids = set()
    for batch_idx, (context_len, row) in enumerate(zip(context_values, block_rows)):
        if context_len < 0:
            raise ValueError(f"context_lens[{batch_idx}] must be nonnegative")
        live_pages = (context_len + page_size - 1) // page_size
        if live_pages > max_pages:
            raise ValueError(f"context_lens[{batch_idx}] is not covered by block_table")
        for logical_page, page_id in enumerate(row):
            if logical_page < live_pages:
                if page_id < 0 or page_id >= num_physical_pages:
                    raise ValueError(
                        f"Live page id {page_id} at [{batch_idx}, {logical_page}] "
                        "is outside physical page storage"
                    )
                if page_id in live_page_ids:
                    raise ValueError(f"Live page id {page_id} is aliased")
                live_page_ids.add(page_id)
            elif page_id != -1:
                raise ValueError(
                    f"Unused block_table entry [{batch_idx}, {logical_page}] "
                    "must use the -1 sentinel"
                )
        if 0 < context_len < L:
            raise ValueError(
                f"context_lens[{batch_idx}] must be zero or at least query length {L}"
            )

    from . import cuda_ops

    is_causal = mask == "causal"
    return cuda_ops.paged_attention(
        query.contiguous(),
        key_pages.contiguous(),
        value_pages.contiguous(),
        block_table.to(device=query.device, dtype=torch.int32).contiguous(),
        context_lens.to(device=query.device, dtype=torch.int32).contiguous(),
        page_size,
        factor,
        is_causal,
    )


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
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        assert hidden_size % num_heads == 0
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim**-0.5
        assert wq.shape == (num_heads * self.head_dim, hidden_size)
        assert wk.shape == (num_heads * self.head_dim, hidden_size)
        assert wv.shape == (num_heads * self.head_dim, hidden_size)
        assert wo.shape == (hidden_size, num_heads * self.head_dim)
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        N, L, _ = query.shape
        assert query.shape == key.shape == value.shape
        projection_q = (
            linear(query, self.wq)
            .reshape(N, L, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        projection_k = (
            linear(key, self.wk)
            .reshape(N, L, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        projection_v = (
            linear(value, self.wv)
            .reshape(N, L, self.num_heads, self.head_dim)
            .permute(0, 2, 1, 3)
        )
        x = scaled_dot_product_attention_simple(
            projection_q,
            projection_k,
            projection_v,
            scale=self.scale,
            mask=mask,
        )
        x = x.permute(0, 2, 1, 3).reshape(N, L, self.hidden_size)
        return linear(x, self.wo)
