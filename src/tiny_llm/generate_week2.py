"""Week 2 Day 1 serving-loop exercise."""


def cached_generate(model, tokenizer, prompt, sampler=None, max_tokens=256):
    # TODO: prefill once, then send only the new token with the current offset.
    # Bound cache capacity by prompt length + max_tokens; release on every exit.
    raise NotImplementedError("Week 2 Day 1: cache-aware generation")
