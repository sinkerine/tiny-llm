from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

from .basics import linear, silu
from .week2_kernels import (
    scaled_dot_product_attention as scaled_dot_product_attention_grouped,
)
from .embedding import Embedding
from .quantized_embedding import QuantizedEmbedding
from .kv_cache import TinyKvCache
from .layer_norm import RMSNorm
from .week2_kernels import ReadableRoPE as RoPE
from .quantize import QuantizedWeights, dequantize_linear, quantized_linear
from .week2_kernels import (
    DENSE_PREFILL_MIN_QUERY,
    FastRMSNorm,
    FastRoPE,
    dense_prefill_attention_mma,
    swiglu,
    decode_attention_online,
    qk_norm_rope,
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
    native_online_decode: bool = False
    fused_qk_norm_rope4: bool = False
    decode_submit18: bool = False  # CUDA operations submit eagerly.


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
            native_online_decode=True,
            fused_qk_norm_rope4=True,
        ),
    }
)
WEEK2_CHECKPOINTS = tuple(WEEK2_CHECKPOINT_FEATURES)


def _linear(x: torch.Tensor, weight: torch.Tensor | QuantizedWeights) -> torch.Tensor:
    if isinstance(weight, QuantizedWeights):
        return quantized_linear(x, weight)
    return linear(x, weight)


def _readable_rope_offset(
    offset: int | list[int] | torch.Tensor, sequence_length: int
) -> slice | list[slice]:
    if isinstance(offset, int):
        return slice(offset, offset + sequence_length)
    if isinstance(offset, list):
        return [slice(value, value + sequence_length) for value in offset]
    values = offset.tolist()
    if not isinstance(values, list):
        values = [values]
    return [slice(value, value + sequence_length) for value in values]


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
        use_native_online_decode: bool = False,
        use_fused_qk_norm_rope4: bool = False,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert hidden_size % num_heads == 0, (
            f"hidden_size {hidden_size} must be divisible by num_heads {num_heads}"
        )
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )
        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.use_fast_rope = use_fast_rope
        self.use_tiled_prefill_attention = use_tiled_prefill_attention
        self.use_native_online_decode = (
            use_native_online_decode and use_tiled_prefill_attention
        )
        self.use_fused_qk_norm_rope4 = (
            use_fused_qk_norm_rope4 and use_fast_rms_norm and use_fast_rope
        )
        self.attention_dispatch_counts = {
            "tiled_prefill": 0,
            "tiled_prefill_fallback": 0,
            "readable": 0,
            "native_online_decode": 0,
            "native_split_decode": 0,
            "fused_qk_norm_rope4": 0,
        }
        rope_cls = FastRoPE if use_fast_rope else RoPE
        norm_cls = FastRMSNorm if use_fast_rms_norm else RMSNorm
        self.rope = rope_cls(self.head_dim, max_seq_len, theta)
        self.q_norm = norm_cls(self.head_dim, q_norm, eps=rms_norm_eps)
        self.k_norm = norm_cls(self.head_dim, k_norm, eps=rms_norm_eps)

    def __call__(
        self,
        x: torch.Tensor,
        offsets: int | list[int] | torch.Tensor,
        cache: TinyKvCache,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        projection_q = _linear(x, self.wq).reshape(B, L, self.num_heads, self.head_dim)
        projection_k = _linear(x, self.wk).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        projection_v = _linear(x, self.wv).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        rope_offsets = offsets
        if not self.use_fast_rope:
            rope_offsets = _readable_rope_offset(rope_offsets, L)
        can_fuse = (
            self.use_fused_qk_norm_rope4
            and type(self.q_norm) is FastRMSNorm
            and type(self.k_norm) is FastRMSNorm
            and type(self.rope) is FastRoPE
            and projection_q.shape == (1, 1, 32, 128)
            and projection_k.shape == (1, 1, 8, 128)
            and projection_q.dtype == projection_k.dtype == torch.bfloat16
            and not self.rope.traditional
        )
        if can_fuse:
            self.attention_dispatch_counts["fused_qk_norm_rope4"] += 1
            projection_q = qk_norm_rope(
                projection_q, self.q_norm, self.rope, rope_offsets
            )
            projection_k = qk_norm_rope(
                projection_k, self.k_norm, self.rope, rope_offsets
            )
        else:
            projection_q = self.rope(self.q_norm(projection_q), offset=rope_offsets)
            projection_k = self.rope(self.k_norm(projection_k), offset=rope_offsets)
        projection_q = projection_q.permute(0, 2, 1, 3)
        projection_k = projection_k.permute(0, 2, 1, 3)
        projection_v = projection_v.permute(0, 2, 1, 3)
        projection_k, projection_v, _, mask = cache.update_and_fetch(
            projection_k, projection_v, mask_length=L, mask=mask
        )
        can_tiled_prefill = (
            L >= DENSE_PREFILL_MIN_QUERY
            and projection_q.dtype == torch.bfloat16
            and self.head_dim == 128
        )
        if self.use_tiled_prefill_attention and not can_tiled_prefill:
            self.attention_dispatch_counts["tiled_prefill_fallback"] += 1

        if self.use_tiled_prefill_attention and can_tiled_prefill:
            self.attention_dispatch_counts["tiled_prefill"] += 1
            x = dense_prefill_attention_mma(
                projection_q,
                projection_k,
                projection_v,
                scale=self.scale,
                mask=mask,
            )
        elif (
            self.use_native_online_decode
            and L == 1
            and self.head_dim == 128
            and 0 < projection_k.shape[2] <= 32768
            and all(
                a.dtype == torch.bfloat16
                for a in (projection_q, projection_k, projection_v)
            )
        ):
            self.attention_dispatch_counts["native_online_decode"] += 1
            if projection_k.shape[2] > 256:
                self.attention_dispatch_counts["native_split_decode"] += 1
            x = decode_attention_online(
                projection_q, projection_k, projection_v, self.scale, mask
            )
        else:
            self.attention_dispatch_counts["readable"] += 1
            x = scaled_dot_product_attention_grouped(
                projection_q,
                projection_k,
                projection_v,
                scale=self.scale,
                mask=mask,
            ).to(x.dtype)
        x = x.permute(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        return _linear(x, self.wo)


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
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down
        self.use_fast_swiglu = use_fast_swiglu

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gate = _linear(x, self.w_gate)
        up = _linear(x, self.w_up)
        hidden = swiglu(gate, up) if self.use_fast_swiglu else silu(gate) * up
        return _linear(hidden, self.w_down)


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
        use_native_online_decode: bool = False,
        use_fused_qk_norm_rope4: bool = False,
    ):
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.mlp = Qwen3MLP(
            hidden_size,
            intermediate_size,
            w_gate,
            w_up,
            w_down,
            use_fast_swiglu=use_fast_swiglu,
        )
        norm_cls = FastRMSNorm if use_fast_rms_norm else RMSNorm
        self.input_layernorm = norm_cls(
            hidden_size, w_input_layernorm, eps=rms_norm_eps
        )
        self.post_attention_layernorm = norm_cls(
            hidden_size, w_post_attention_layernorm, eps=rms_norm_eps
        )
        self.self_attn = Qwen3MultiHeadAttention(
            num_heads=num_attention_heads,
            hidden_size=hidden_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            wq=wq,
            wk=wk,
            wv=wv,
            wo=wo,
            q_norm=q_norm,
            k_norm=k_norm,
            max_seq_len=max_seq_len,
            theta=theta,
            rms_norm_eps=rms_norm_eps,
            use_fast_rms_norm=use_fast_rms_norm,
            use_fast_rope=use_fast_rope,
            use_tiled_prefill_attention=use_tiled_prefill_attention,
            use_native_online_decode=use_native_online_decode,
            use_fused_qk_norm_rope4=use_fused_qk_norm_rope4,
        )

    def __call__(
        self,
        x: torch.Tensor,
        offset: int,
        cache: TinyKvCache,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), offset, cache, mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


class Qwen3ModelWeek2:
    def __init__(
        self,
        hf_model: Any,
        checkpoint: str = "selected",
        use_torch_quantized_linear: bool = False,
        use_bounded_kv_capacity: bool | None = None,
        use_register_cached_rms_norm: bool | None = None,
        use_tiled_prefill_attention: bool | None = None,
        use_native_online_decode: bool | None = None,
        use_fused_qk_norm_rope4: bool | None = None,
        use_decode_submit18: bool | None = None,
    ):
        if checkpoint not in WEEK2_CHECKPOINTS:
            raise ValueError(
                f"unknown Week 2 checkpoint {checkpoint!r}; "
                f"choose one of {WEEK2_CHECKPOINTS}"
            )
        for name, value in (
            ("use_bounded_kv_capacity", use_bounded_kv_capacity),
            ("use_register_cached_rms_norm", use_register_cached_rms_norm),
            ("use_tiled_prefill_attention", use_tiled_prefill_attention),
            ("use_native_online_decode", use_native_online_decode),
            ("use_fused_qk_norm_rope4", use_fused_qk_norm_rope4),
            ("use_decode_submit18", use_decode_submit18),
        ):
            if value is not None and not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool or None")
        if use_decode_submit18:
            raise ValueError(
                "MLX submit18 has no CUDA operation: CUDA kernels already submit eagerly"
            )
        self.use_decode_submit18 = False
        self.checkpoint = checkpoint
        features = WEEK2_CHECKPOINT_FEATURES[checkpoint]
        if use_bounded_kv_capacity is None:
            use_bounded_kv_capacity = features.bounded_kv_capacity
        if use_register_cached_rms_norm is None:
            use_register_cached_rms_norm = features.fast_rms_norm
        if use_tiled_prefill_attention is None:
            use_tiled_prefill_attention = features.tiled_prefill_attention
        if use_native_online_decode is None:
            use_native_online_decode = features.native_online_decode
        if use_fused_qk_norm_rope4 is None:
            use_fused_qk_norm_rope4 = features.fused_qk_norm_rope4
        use_quantized_weights = features.quantized_weights
        use_fast_rms_norm = use_register_cached_rms_norm
        use_fast_rope = features.fast_rope
        use_fast_swiglu = features.fast_swiglu
        use_simdgroup_matmul = features.simdgroup_matmul
        use_split_k_matmul = False
        self.num_hidden_layers = hf_model.config.num_hidden_layers
        self.use_bounded_kv_capacity = use_bounded_kv_capacity
        self.use_register_cached_rms_norm = use_register_cached_rms_norm
        self.use_tiled_prefill_attention = use_tiled_prefill_attention
        self.mechanism_controls = MappingProxyType(
            {
                "capacity_cache": use_bounded_kv_capacity,
                "register_cached_rmsnorm": use_register_cached_rms_norm,
                "tiled_prefill": use_tiled_prefill_attention,
                "native_online_decode": use_native_online_decode,
                "fused_qk_norm_rope4": use_fused_qk_norm_rope4,
                "decode_submit18": False,
            }
        )
        self.use_fast_rope = use_fast_rope
        self.hidden_size = hf_model.config.hidden_size
        self.vocab_size = hf_model.config.vocab_size
        precision = torch.bfloat16
        self.precision = precision

        def model_weight(layer: Any) -> torch.Tensor | QuantizedWeights:
            if use_quantized_weights:
                return QuantizedWeights.from_layer(
                    layer,
                    use_simdgroup_matmul=use_simdgroup_matmul,
                    use_split_k_matmul=use_split_k_matmul,
                    use_torch_quantized_linear=use_torch_quantized_linear,
                )
            return dequantize_linear(layer).to(torch.bfloat16)

        embedding_weight = model_weight(hf_model.model.embed_tokens)
        if isinstance(embedding_weight, QuantizedWeights):
            self.embedding = QuantizedEmbedding(
                vocab_size=self.vocab_size,
                embedding_dim=self.hidden_size,
                weight=embedding_weight,
            )
        else:
            self.embedding = Embedding(
                vocab_size=self.vocab_size,
                embedding_dim=self.hidden_size,
                weight=embedding_weight,
            )
        self.layers_inner = []

        for i in range(hf_model.config.num_hidden_layers):
            wq = model_weight(hf_model.model.layers[i].self_attn.q_proj)
            wk = model_weight(hf_model.model.layers[i].self_attn.k_proj)
            wv = model_weight(hf_model.model.layers[i].self_attn.v_proj)
            wo = model_weight(hf_model.model.layers[i].self_attn.o_proj)
            w_gate = model_weight(hf_model.model.layers[i].mlp.gate_proj)
            w_up = model_weight(hf_model.model.layers[i].mlp.up_proj)
            w_down = model_weight(hf_model.model.layers[i].mlp.down_proj)

            layer = Qwen3TransformerBlock(
                num_attention_heads=hf_model.config.num_attention_heads,
                num_kv_heads=hf_model.config.num_key_value_heads,
                hidden_size=hf_model.config.hidden_size,
                head_dim=hf_model.config.head_dim,
                intermediate_size=hf_model.config.intermediate_size,
                rms_norm_eps=hf_model.config.rms_norm_eps,
                wq=wq,
                wk=wk,
                wv=wv,
                wo=wo,
                q_norm=hf_model.model.layers[i].self_attn.q_norm.weight,
                k_norm=hf_model.model.layers[i].self_attn.k_norm.weight,
                w_gate=w_gate,
                w_up=w_up,
                w_down=w_down,
                w_input_layernorm=hf_model.model.layers[i].input_layernorm.weight,
                w_post_attention_layernorm=hf_model.model.layers[
                    i
                ].post_attention_layernorm.weight,
                max_seq_len=hf_model.config.max_position_embeddings,
                theta=hf_model.config.rope_theta,
                use_fast_rms_norm=use_fast_rms_norm,
                use_fast_rope=use_fast_rope,
                use_fast_swiglu=use_fast_swiglu,
                use_tiled_prefill_attention=use_tiled_prefill_attention,
                use_native_online_decode=use_native_online_decode,
                use_fused_qk_norm_rope4=use_fused_qk_norm_rope4,
            )
            self.layers_inner.append(layer)
        norm_cls = FastRMSNorm if use_fast_rms_norm else RMSNorm
        self.norm = norm_cls(
            hf_model.config.hidden_size,
            weight=hf_model.model.norm.weight,
            eps=hf_model.config.rms_norm_eps,
        )
        if not hf_model.config.tie_word_embeddings:
            self.w_lm_head = model_weight(hf_model.lm_head)
        else:
            self.w_lm_head = None
        self.device = hf_model.device

    def create_kv_cache(self, capacity: int | None = None) -> list[TinyKvCache]:
        from .kv_cache import TinyKvFullCache

        return [
            TinyKvFullCache(capacity=capacity) for _ in range(self.num_hidden_layers)
        ]

    def __call__(
        self,
        inputs: torch.Tensor,
        offset: int,
        cache: list[TinyKvCache],
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        if isinstance(offset, int):
            for layer, layer_cache in enumerate(cache):
                cache_offset = getattr(layer_cache, "offset", None)
                if cache_offset is not None and cache_offset != offset:
                    raise ValueError(
                        f"layer {layer} cache offset {cache_offset} "
                        f"does not match model offset {offset}"
                    )
        h = self.embedding(inputs)
        mask = None if inputs.shape[1] == 1 else "causal"
        if not getattr(self, "use_fast_rope", True):
            rope_offsets = offset
        elif isinstance(offset, int):
            rope_offsets = torch.full(
                (inputs.shape[0],), offset, dtype=torch.long, device=inputs.device
            )
        elif isinstance(offset, list):
            rope_offsets = torch.tensor(offset, dtype=torch.long, device=inputs.device)
        else:
            rope_offsets = offset
        for layer in range(self.num_hidden_layers):
            h = self.layers_inner[layer](h, rope_offsets, cache[layer], mask=mask)
        if logits_to_keep is not None:
            if logits_to_keep <= 0:
                raise ValueError("logits_to_keep must be positive")
            h = h[:, -logits_to_keep:, :]
        h = self.norm(h)
        if self.w_lm_head is not None:
            return _linear(h, self.w_lm_head)
        else:
            return self.embedding.as_linear(h)
