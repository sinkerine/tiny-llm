import torch


class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: torch.Tensor):
        self.weight = weight

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight[x]

    def as_linear(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("...D,VD->...V", x, self.weight)
