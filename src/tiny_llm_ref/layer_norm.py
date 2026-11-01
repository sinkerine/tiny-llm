import torch


class RMSNorm:
    def __init__(self, dim: int, weight: torch.Tensor, eps: float = 1e-5):
        self.dim = dim
        self.eps = eps
        self.weight = weight

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(
            torch.mean(torch.square(x), dim=-1, keepdim=True) + self.eps
        )
        x = x.to(orig_dtype)
        return x * self.weight.to(orig_dtype)
