from typing import Any

import torch


def _step(model, y, offsets, kv_cache):
    raise NotImplementedError("Week 3 Day 1: implement continuous batching")


class Request:
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        prompt: str,
        prefill_max_step: int = 128,
        prompt_idx: int = 0,
        max_seq_len: int | None = None,
    ):
        raise NotImplementedError("Week 3 Day 1: implement Request")

    def try_prefill(self):
        raise NotImplementedError("Week 3 Day 2: implement chunked prefill")

    def decode_done(self, token, update_offset=True):
        raise NotImplementedError("Week 3 Day 1: implement decode_done")

    def text(self):
        raise NotImplementedError("Week 3 Day 1: implement text")

    def reaches_max_seq_len(self, max_seq_len: int) -> bool:
        raise NotImplementedError("Week 3 Day 1: implement reaches_max_seq_len")


def batch_generate(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    max_seq_len: int = 512,
    batch_size: int = 5,
    prefill_step: int = 128,
):
    raise NotImplementedError("Week 3 Day 1: implement batch_generate")
