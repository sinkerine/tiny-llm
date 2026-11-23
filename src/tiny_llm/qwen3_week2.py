from dataclasses import dataclass
from types import MappingProxyType
from typing import Any
from .basics import linear, silu
from .positional_encoding import RoPE
from .layer_norm import RMSNorm

import torch

from .embedding import Embedding  # noqa: F401 - learner checkpoint dependency
from .kv_cache import TinyKvCache, TinyKvFullCache
from .quantize import QuantizedWeights, dequantize_linear, quantized_linear  # noqa: F401
from .quantized_embedding import QuantizedEmbedding
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


def _project(x: torch.Tensor, w: torch.Tensor | QuantizedWeights) -> torch.Tensor:
    if isinstance(w, QuantizedWeights):
        return quantized_linear(x, w)
    return linear(x, w)


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
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.rms_norm_eps = rms_norm_eps
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.q_norm = RMSNorm(head_dim, q_norm, eps=rms_norm_eps)
        self.k_norm = RMSNorm(head_dim, k_norm, eps=rms_norm_eps)
        self.rope = RoPE(head_dim, max_seq_len, theta)

    def __call__(
        self,
        x: torch.Tensor,
        offsets: int | list[int] | torch.Tensor,
        cache: TinyKvCache,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        q = _project(x, self.wq)
        k = _project(x, self.wk)
        v = _project(x, self.wv)
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope(q, slice(offsets, L + offsets))
        k = self.rope(k, slice(offsets, L + offsets))
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        keys, values, _, mask = cache.update_and_fetch(k, v, mask=mask)
        S = keys.shape[2]
        attention_mask = mask
        if isinstance(mask, str) and mask == "causal":
            query_pos = torch.arange(offsets, L + offsets, device=q.device)
            key_pos = torch.arange(0, S, device=k.device)
            attention_mask = key_pos[None, :] <= query_pos[:, None]
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            keys,
            values,
            attn_mask=attention_mask,
            is_causal=False,
            dropout_p=0.0,
            enable_gqa=True,
        )
        out = (
            out.transpose(1, 2)
            .reshape(B, L, self.num_heads * self.head_dim)
            .to(dtype=x.dtype)
        )
        return _project(out, self.wo)


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
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(_project(x, self.w_gate))
        up = _project(x, self.w_up)
        up_gated = gate * up
        return _project(up_gated, self.w_down)


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
        self.input_layernorm = RMSNorm(hidden_size, w_input_layernorm, rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            hidden_size, w_post_attention_layernorm, rms_norm_eps
        )
        self.mlp = Qwen3MLP(hidden_size, intermediate_size, w_gate, w_up, w_down)
        self.self_attn = Qwen3MultiHeadAttention(
            hidden_size,
            num_attention_heads,
            num_kv_heads,
            head_dim,
            wq,
            wk,
            wv,
            wo,
            q_norm,
            k_norm,
            max_seq_len,
            theta,
            rms_norm_eps,
        )

    def __call__(
        self,
        x: torch.Tensor,
        offset: int,
        cache: TinyKvCache,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, offset, cache, mask)
        x = x + residual
        post_residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + post_residual
        return x


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
        features = WEEK2_CHECKPOINT_FEATURES[checkpoint]
        self.use_bounded_kv_capacity = (
            features.bounded_kv_capacity
            if use_bounded_kv_capacity is None
            else use_bounded_kv_capacity
        )
        self.num_hidden_layers = hf_model.config.num_hidden_layers
        config = hf_model.config
        self.device = hf_model.model.embed_tokens.weight.device

        def pack_weight(layer: Any) -> torch.Tensor | QuantizedWeights:
            if features.quantized_weights:
                return QuantizedWeights.from_layer(
                    layer,
                    use_simdgroup_matmul=features.simdgroup_matmul,
                    use_torch_quantized_linear=use_torch_quantized_linear,
                )
            return layer.weight

        if features.quantized_weights:
            self.embedding = QuantizedEmbedding(
                config.vocab_size,
                config.hidden_size,
                pack_weight(hf_model.model.embed_tokens),
            )
        else:
            self.embedding = Embedding(
                config.vocab_size, config.hidden_size, hf_model.model.embed_tokens.weight
            )
        self.norm = RMSNorm(
            config.hidden_size, hf_model.model.norm.weight, config.rms_norm_eps
        )
        self.layers_inner = []
        for layer in hf_model.model.layers:
            self.layers_inner.append(
                Qwen3TransformerBlock(
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.hidden_size,
                    config.head_dim,
                    config.intermediate_size,
                    config.rms_norm_eps,
                    pack_weight(layer.self_attn.q_proj),
                    pack_weight(layer.self_attn.k_proj),
                    pack_weight(layer.self_attn.v_proj),
                    pack_weight(layer.self_attn.o_proj),
                    layer.self_attn.q_norm.weight,
                    layer.self_attn.k_norm.weight,
                    pack_weight(layer.mlp.gate_proj),
                    pack_weight(layer.mlp.up_proj),
                    pack_weight(layer.mlp.down_proj),
                    layer.input_layernorm.weight,
                    layer.post_attention_layernorm.weight,
                    config.max_position_embeddings,
                    config.rope_theta,
                )
            )
        self.tie_word_embeddings = config.tie_word_embeddings
        if not config.tie_word_embeddings:
            self.lm_head_weight = pack_weight(hf_model.lm_head)

    def create_kv_cache(self, capacity: int | None = None) -> list[TinyKvCache]:
        return [TinyKvFullCache(capacity=capacity) for _ in self.layers_inner]

    def __call__(
        self,
        inputs: torch.Tensor,
        offset: int,
        cache: list[TinyKvCache],
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        x = self.embedding(inputs)
        for c in cache:
            if c.offset != offset:
                raise ValueError("offset not match")
        for i in range(0, len(cache)):
            layer = self.layers_inner[i]
            c = cache[i]
            x = layer(x, offset, c, mask="causal")
        if logits_to_keep == 1:
            x = x[:, -1:, :]
        x = self.norm(x)
        if self.tie_word_embeddings:
            x = self.embedding.as_linear(x)
        else:
            x = _project(x, self.lm_head_weight)
        return x
