from typing import Callable

import torch
from transformers import PreTrainedTokenizerBase, TextStreamer


@torch.inference_mode()
def simple_generate(
    model,
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    sampler: Callable[[torch.Tensor], torch.Tensor] | None,
    max_tokens: int = 256,
) -> None:
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or max_tokens < 0
    ):
        raise ValueError("max_tokens must be a non-negative integer")
    if max_tokens == 0:
        return
    tokens = torch.tensor(
        tokenizer.encode(prompt, add_special_tokens=False),
        dtype=torch.long,
        device=getattr(model, "device", "cpu"),
    )
    if tokens.numel() == 0:
        raise ValueError("prompt must encode to at least one token")
    streamer = TextStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)
    for _ in range(max_tokens):
        logits = model(tokens[None])[:, -1, :].float()
        logprobs = torch.log_softmax(logits, dim=-1)
        token = logprobs.argmax(dim=-1) if sampler is None else sampler(logprobs)
        if token.item() == tokenizer.eos_token_id:
            break
        tokens = torch.cat((tokens, token.to(device=tokens.device, dtype=torch.long)))
        streamer.put(token.cpu())
    streamer.end()
