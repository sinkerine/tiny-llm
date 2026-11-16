from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from .embedding import Embedding  # noqa: F401 - learner checkpoint dependency
from .kv_cache import TinyKvCache
from .quantize import QuantizedWeights, dequantize_linear  # noqa: F401
from .week2_kernels import (
    FastRMSNorm,  # noqa: F401 - learner checkpoint dependency
    FastRoPE,  # noqa: F401 - learner checkpoint dependency
    scaled_dot_product_attention,  # noqa: F401 - learner checkpoint dependency
    swiglu,  # noqa: F401 - learner checkpoint dependency
)


@dataclass(frozen=True)
class Week2CheckpointFeatures:
    bounded_kv_capacity: bool = False
    quantized_weights: bool = False
    fast_rms_norm: bool = False
    fast_rope: bool = False
    fast_swiglu: bool = False
    simdgroup_matmul: bool = False
    tiled_prefill_attention: bool = False


WEEK2_CHECKPOINT_FEATURES = MappingProxyType(
    {
        "kv-cache": Week2CheckpointFeatures(),
        "capacity-cache": Week2CheckpointFeatures(bounded_kv_capacity=True),
        "quantized-matvec": Week2CheckpointFeatures(
            bounded_kv_capacity=True, quantized_weights=True
        ),
        "simd-matmul": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
        ),
        "rmsnorm": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
        ),
        "rope": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
        ),
        "swiglu": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
        ),
        "tiled-prefill": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
            tiled_prefill_attention=True,
        ),
        "selected": Week2CheckpointFeatures(
            bounded_kv_capacity=True,
            quantized_weights=True,
            simdgroup_matmul=True,
            fast_rms_norm=True,
            fast_rope=True,
            fast_swiglu=True,
            tiled_prefill_attention=True,
        ),
    }
)
WEEK2_CHECKPOINTS = tuple(WEEK2_CHECKPOINT_FEATURES)


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: torch.Tensor | QuantizedWeights,
        wk: torch.Tensor | QuantizedWeights,
        wv: torch.Tensor | QuantizedWeights,
        wo: torch.Tensor | QuantizedWeights,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_tiled_prefill_attention: bool = False,
    ):
        pass

    def __call__(
        self,
        x: torch.Tensor,
        offsets: int | list[int] | torch.Tensor,
        cache: TinyKvCache,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        pass


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: torch.Tensor | QuantizedWeights,
        w_up: torch.Tensor | QuantizedWeights,
        w_down: torch.Tensor | QuantizedWeights,
        use_fast_swiglu: bool = True,
    ):
        pass

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        pass


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: torch.Tensor | QuantizedWeights,
        wk: torch.Tensor | QuantizedWeights,
        wv: torch.Tensor | QuantizedWeights,
        wo: torch.Tensor | QuantizedWeights,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        w_gate: torch.Tensor | QuantizedWeights,
        w_up: torch.Tensor | QuantizedWeights,
        w_down: torch.Tensor | QuantizedWeights,
        w_input_layernorm: torch.Tensor,
        w_post_attention_layernorm: torch.Tensor,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_fast_rms_norm: bool = True,
        use_fast_rope: bool = True,
        use_fast_swiglu: bool = True,
        use_tiled_prefill_attention: bool = False,
    ):
        pass

    def __call__(
        self,
        x: torch.Tensor,
        offset: int,
        cache: TinyKvCache,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        pass


class Qwen3ModelWeek2:
    def __init__(
        self,
        hf_model: Any,
        checkpoint: str = "selected",
        use_torch_quantized_linear: bool = False,
        use_bounded_kv_capacity: bool | None = None,
        use_register_cached_rms_norm: bool | None = None,
        use_tiled_prefill_attention: bool | None = None,
    ):
        self.num_hidden_layers = hf_model.config.num_hidden_layers
        pass

    def create_kv_cache(self, capacity: int | None = None) -> list[TinyKvCache]:
        pass

    def __call__(
        self,
        inputs: torch.Tensor,
        offset: int,
        cache: list[TinyKvCache],
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        pass
