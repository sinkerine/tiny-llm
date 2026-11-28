import torch

from .basics import silu
from .quantize import QuantizedWeights, quantized_linear


def grouped_expert_linear(
    x: torch.Tensor,
    w_experts: QuantizedWeights,
    expert_ids: torch.Tensor,
) -> torch.Tensor:
    *leading_dims, D = x.shape
    flat_x = x.reshape(-1, D)
    flat_expert_ids = expert_ids.reshape(-1).long()
    out_dim = w_experts.weight.shape[-2]
    out = torch.empty((flat_x.shape[0], out_dim), dtype=x.dtype, device=x.device)
    if flat_x.shape[0] == 0:
        return out.reshape(*leading_dims, out_dim)

    sort_idx = torch.argsort(flat_expert_ids, stable=True)
    sorted_ids = flat_expert_ids[sort_idx]
    unique_experts, counts = torch.unique_consecutive(sorted_ids, return_counts=True)
    offset = 0
    for expert_id, count in zip(unique_experts.tolist(), counts.tolist()):
        group_idx = sort_idx[offset : offset + count]
        offset += count
        w_e = QuantizedWeights(
            scales=w_experts.scales[expert_id],
            biases=w_experts.biases[expert_id],
            group_size=w_experts.group_size,
            bits=w_experts.bits,
            weight=w_experts.weight[expert_id],
            use_simdgroup_matmul=w_experts.use_simdgroup_matmul,
            use_simdgroup_matvec=w_experts.use_simdgroup_matvec,
            use_split_k_matmul=w_experts.use_split_k_matmul,
            use_torch_quantized_linear=w_experts.use_torch_quantized_linear,
        )
        out[group_idx] = quantized_linear(flat_x[group_idx], w_e)
    return out.reshape(*leading_dims, out_dim)


def route_topk(
    x: torch.Tensor,
    w_router: QuantizedWeights,
    top_k: int,
    norm_topk_prob: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    router_logits = quantized_linear(x, w_router)
    router_probs = torch.softmax(router_logits.float(), dim=-1).to(router_logits.dtype)
    expert_scores, expert_ids = torch.topk(router_probs, k=top_k, dim=-1)
    if norm_topk_prob:
        expert_scores = (
            expert_scores.float() / expert_scores.float().sum(dim=-1, keepdim=True)
        ).to(router_probs.dtype)
    return router_probs, expert_ids, expert_scores


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
        self.w_router = w_router
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        _, expert_ids, expert_scores = route_topk(
            x,
            self.w_router,
            top_k=self.num_experts_per_tok,
            norm_topk_prob=self.norm_topk_prob,
        )
        expanded_x = (
            x.unsqueeze(-2)
            .expand(B, L, self.num_experts_per_tok, D)
            .reshape(-1, D)
        )
        flat_expert_ids = expert_ids.reshape(-1)

        gate = grouped_expert_linear(expanded_x, self.w_gate, flat_expert_ids)
        up = grouped_expert_linear(expanded_x, self.w_up, flat_expert_ids)
        expert_output = grouped_expert_linear(
            silu(gate) * up,
            self.w_down,
            flat_expert_ids,
        ).reshape(B, L, self.num_experts_per_tok, D)
        return (expert_output * expert_scores.unsqueeze(-1)).sum(dim=-2)
