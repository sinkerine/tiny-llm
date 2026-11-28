import torch

from .quantize import QuantizedWeights


def grouped_expert_linear(
    x: torch.Tensor,
    w_experts: QuantizedWeights,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
    raise NotImplementedError("Week 3 Day 6 (Optional): implement grouped_expert_linear")


def route_topk(
    x: torch.Tensor,
    w_router: QuantizedWeights,
    top_k: int,
    norm_topk_prob: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raise NotImplementedError("Week 3 Day 6 (Optional): implement route_topk")


class Moe:
    def __init__(
        self,
        w_router: QuantizedWeights,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
        num_experts_per_tok: int,
        norm_topk_prob: bool = False,
    ):
        raise NotImplementedError("Week 3 Day 6 (Optional): implement Moe")

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Week 3 Day 6 (Optional): implement Moe")
