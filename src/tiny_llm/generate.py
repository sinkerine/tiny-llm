import torch
from transformers import PreTrainedTokenizerBase, TextStreamer
from .qwen3_week1 import Qwen3ModelWeek1
from typing import Callable


def _release_kv_cache(kv_cache):
    if kv_cache is None:
        return
    for layer in kv_cache:
        layer.release()


def simple_generate(
    model: Qwen3ModelWeek1,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    sampler: Callable[[torch.Tensor], torch.Tensor] | None,
    max_tokens: int = 256,
) -> None:
    def _step(model, y):
        scores = model(y.unsqueeze(0))
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
    if hasattr(model, "embed_tokens"):
        device = model.embed_tokens.weight.device
    else:
        device = model.device
    y = torch.tensor(tokens, dtype=torch.long, device=device)
    streamer = TextStreamer(tokenizer)
    for _ in range(0, max_tokens):
        next_token = _step(model, y)
        if next_token == tokenizer.eos_token_id:
            break
        streamer.put(next_token.cpu())
        y = torch.cat([y, next_token], 0)
    streamer.end()


def speculative_generate(
    draft_model,
    model,
    draft_tokenizer,
    tokenizer,
    prompt: str,
    proposal_length: int = 4,
    max_tokens: int = 256,
) -> str:
    raise NotImplementedError("Week 3 Day 7 (Optional): implement speculative_generate")
