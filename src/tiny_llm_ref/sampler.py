import torch


def make_sampler(temp: float, top_p: float | None, top_k: int | None):
    def sample(logprobs: torch.Tensor) -> torch.Tensor:
        if temp == 0:
            return logprobs.argmax(dim=-1)
        scores = logprobs.clone()
        if top_k is not None and top_k > 0:
            if top_k > scores.shape[-1]:
                raise ValueError("top_k exceeds vocabulary size")
            indices = scores.topk(top_k, dim=-1).indices
            filtered = torch.full_like(scores, -torch.inf)
            scores = filtered.scatter(-1, indices, scores.gather(-1, indices))
        if top_p is not None and 0 < top_p < 1:
            sorted_scores, indices = scores.sort(dim=-1, descending=True)
            probabilities = sorted_scores.exp()
            keep = probabilities.cumsum(dim=-1) - probabilities < top_p
            sorted_scores = sorted_scores.masked_fill(~keep, -torch.inf)
            scores = torch.empty_like(scores).scatter(-1, indices, sorted_scores)
        return torch.multinomial(torch.softmax(scores / temp, dim=-1), 1).squeeze(-1)

    return sample
