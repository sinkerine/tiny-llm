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
    if bias is not None:
        return torch.matmul(x, w.T) + bias
    else:
        return torch.matmul(x, w.T)


def silu(x: torch.Tensor) -> torch.Tensor:
    # Use abs(x) to avoid exp(large positive) when x is a large negative value.
    # For x < 0, 1 / (1 + exp(-x)) = exp(x) / (1 + exp(x)).
    z = torch.exp(-torch.abs(x))
    sigmoid = torch.where(x < 0, z / (1 + z), 1 / (1 + z))
    return x * sigmoid
