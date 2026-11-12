import torch
import math


def softmax(x: torch.Tensor, axis: int) -> torch.Tensor:
    # Supplied for Day 1; a manual implementation is an optional bonus exercise.
    return torch.softmax(x, dim=axis)


def linear(
    x: torch.Tensor,
    w: torch.Tensor,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    result = torch.einsum("...i,...oi->...o", x, w)
    if bias is not None:
        result += bias
    return result


def silu(x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    v = torch.exp(-torch.abs(x))
    result = x * torch.where(x >= 0, 1 / (1 + v), v / (v + 1))
    return result.to(dtype)
