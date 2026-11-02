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
