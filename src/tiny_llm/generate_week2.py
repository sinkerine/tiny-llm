"""Week 2 Day 1 seroing-loop exercise."""

from .qwen3_week2 import Qwen3ModelWeek2
import torch
from transformers import PreTrainedTokenizerBase, TextStreamer


def cached_generate(
    model: Qwen3ModelWeek2,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    sampler=None,
    max_tokens=256,
):
    def _step(
        model: Qwen3ModelWeek2, x: torch.Tensor, offset, kv_cache
    ) -> torch.Tensor:
        scores = model(x, offset, kv_cache, logits_to_keep=1)
        last_logits = scores[:, -1, :]
        last_logits = torch.log_softmax(last_logits.float(), 1)
        if sampler is None:
            return torch.argmax(last_logits, 1)
        else:
            return sampler(last_logits)

    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 0
    ):
        raise ValueError()
    if max_tokens == 0:
        return
    tokens = tokenizer.encode(prompt, add_special_tokens=False)
    if len(tokens) == 0:
        raise ValueError()
    capacity = (
        len(tokens) + max_tokens
        if getattr(model, "use_bounded_kv_capacity", False)
        else None
    )
    kv_cache = model.create_kv_cache(capacity)
    try:
        with torch.inference_mode():
            if hasattr(model, "embedding"):
                device = model.embedding.weight.device
            else:
                device = model.device
            x = torch.tensor([tokens], dtype=torch.long, device=device)
            streamer = TextStreamer(tokenizer)
            offset = 0
            for _ in range(0, max_tokens):
                next_token = _step(model, x, offset, kv_cache)
                offset += x.shape[1]
                if next_token == tokenizer.eos_token_id:
                    break
                streamer.put(next_token.cpu())
                x = next_token.unsqueeze(1)
            streamer.end()
    finally:
        for cache in kv_cache:
            cache.release()
