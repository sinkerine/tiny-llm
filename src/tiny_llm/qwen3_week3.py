from typing import Any

import torch

from .kv_cache import TinyKvCache


def is_qwen3_moe_sparse_layer(config: Any, layer_idx: int) -> bool:
    return (
        getattr(config, "num_experts", 0) > 0
        and layer_idx not in getattr(config, "mlp_only_layers", [])
        and (layer_idx + 1) % getattr(config, "decoder_sparse_step", 1) == 0
    )


class Qwen3ModelWeek3:
    def __init__(
        self,
        hf_model: Any,
        page_size: int = 128,
        enable_paged_attention: bool = True,
        use_torch_quantized_linear: bool = False,
    ):
        raise NotImplementedError("Week 3: implement Qwen3ModelWeek3")

    def create_kv_cache(self) -> list[TinyKvCache]:
        raise NotImplementedError("Week 3: implement create_kv_cache")

    def __call__(
        self,
        inputs: torch.Tensor,
        offset: int | list[int] | torch.Tensor,
        cache: list[TinyKvCache],
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        raise NotImplementedError("Week 3: implement Qwen3ModelWeek3.__call__")
