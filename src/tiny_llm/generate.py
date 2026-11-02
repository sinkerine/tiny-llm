import torch
from transformers import PreTrainedTokenizerBase
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
        pass
