"""Cache-aware single-request generation (Week 2 reference)."""

import torch
from transformers import TextStreamer


@torch.inference_mode()
def cached_generate(model, tokenizer, prompt, sampler=None, max_tokens=256):
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
        device=model.device,
        dtype=torch.long,
    )[None]
    if not tokens.numel():
        raise ValueError("prompt must encode to at least one token")
    cache = model.create_kv_cache(
        capacity=tokens.shape[1] + max_tokens if model.use_bounded_kv_capacity else None
    )
    streamer = TextStreamer(tokenizer, skip_special_tokens=True)
    offset = 0
    try:
        for _ in range(max_tokens):
            logits = model(tokens, offset, cache, logits_to_keep=1)[:, -1].float()
            offset += tokens.shape[1]
            logprobs = logits.log_softmax(-1)
            token = logprobs.argmax(-1) if sampler is None else sampler(logprobs)
            if token.item() == tokenizer.eos_token_id:
                break
            streamer.put(token.cpu())
            tokens = token.to(device=model.device, dtype=torch.long).reshape(1, 1)
    finally:
        for layer in cache:
            layer.release()
        streamer.end()
