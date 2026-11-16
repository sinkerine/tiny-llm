import torch
import copy


def make_sampler(temp: float, top_p: float | None, top_k: int | None):
    def sample(logprobs: torch.Tensor):
        if temp == 0:
            return torch.argmax(logprobs, dim=-1)
        probs = logprobs
        if top_k and top_k > 0:
            if top_k > logprobs.shape[-1]:
                raise ValueError()
            values, indices = torch.topk(logprobs, top_k, -1)
            probs = torch.full_like(logprobs, -torch.inf)
            probs = probs.scatter(-1, indices, values)
        if top_p and top_p > 0 and top_p < 1:
            sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
            exp_sorted_probs = sorted_probs.exp()
            cum_probs = exp_sorted_probs.cumsum(-1)
            cum_diff = cum_probs - exp_sorted_probs
            keep = cum_diff < top_p
            selected_probs = torch.where(keep, sorted_probs, -torch.inf)
            probs = probs.scatter(-1, sorted_indices, selected_probs)
        probs = probs / temp
        probs = torch.softmax(probs, -1)
        return torch.multinomial(probs, 1).squeeze(-1)

    return sample
