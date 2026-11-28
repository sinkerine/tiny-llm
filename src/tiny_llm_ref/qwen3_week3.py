from typing import Any

import torch

from .attention import paged_attention
from .kv_cache import TinyKvCache
from .moe import Moe
from .paged_kv_cache import TinyKvPagedCache, TinyKvPagedPool
from .quantize import QuantizedWeights, quantized_linear
from .quantized_embedding import QuantizedEmbedding
from .week2_kernels import (
    FastRMSNorm,
    FastRoPE,
    scaled_dot_product_attention,
    swiglu,
)


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
        use_paged_attention: bool = True,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        assert num_heads % num_kv_heads == 0, (
            f"num_heads {num_heads} must be divisible by num_kv_heads {num_kv_heads}"
        )
        self.head_dim = head_dim
        self.scale = self.head_dim**-0.5
        self.wq = wq
        self.wk = wk
        self.wv = wv
        self.wo = wo
        self.rope = FastRoPE(self.head_dim, max_seq_len, theta)
        self.q_norm = FastRMSNorm(self.head_dim, q_norm, eps=rms_norm_eps)
        self.k_norm = FastRMSNorm(self.head_dim, k_norm, eps=rms_norm_eps)
        self.use_paged_attention = use_paged_attention

    def __call__(
        self,
        x: torch.Tensor,
        offsets: int | list[int] | torch.Tensor,
        cache: TinyKvCache,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        projection_q = quantized_linear(x, self.wq).reshape(
            B, L, self.num_heads, self.head_dim
        )
        projection_k = quantized_linear(x, self.wk).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        projection_q = self.q_norm(projection_q)
        projection_k = self.k_norm(projection_k)
        projection_v = quantized_linear(x, self.wv).reshape(
            B, L, self.num_kv_heads, self.head_dim
        )
        projection_q = self.rope(projection_q, offset=offsets)
        projection_k = self.rope(projection_k, offset=offsets)
        projection_q = projection_q.permute(0, 2, 1, 3)
        projection_k = projection_k.permute(0, 2, 1, 3)
        projection_v = projection_v.permute(0, 2, 1, 3)

        if self.use_paged_attention:
            metadata = cache.update_and_fetch_paged(
                projection_k,
                projection_v,
                mask_length=L,
                mask=mask,
            )
            x = paged_attention(
                projection_q,
                metadata.key_pages,
                metadata.value_pages,
                metadata.block_table,
                metadata.context_lens,
                metadata.page_size,
                scale=self.scale,
                mask=metadata.mask,
            )
        else:
            key, value, _, mask = cache.update_and_fetch(
                projection_k,
                projection_v,
                mask_length=L,
                mask=mask,
            )
            x = scaled_dot_product_attention(
                projection_q,
                key,
                value,
                scale=self.scale,
                mask=mask,
            )
        x = x.permute(0, 2, 1, 3).reshape(B, L, self.num_heads * self.head_dim)
        return quantized_linear(x, self.wo)


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: QuantizedWeights,
        w_up: QuantizedWeights,
        w_down: QuantizedWeights,
    ):
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return quantized_linear(
            swiglu(
                quantized_linear(x, self.w_gate),
                quantized_linear(x, self.w_up),
            ),
            self.w_down,
        )


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        rms_norm_eps: float,
        wq: QuantizedWeights,
        wk: QuantizedWeights,
        wv: QuantizedWeights,
        wo: QuantizedWeights,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        w_input_layernorm: torch.Tensor,
        w_post_attention_layernorm: torch.Tensor,
        mlp: Qwen3MLP | Moe,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        use_paged_attention: bool = True,
    ):
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.mlp = mlp
        self.input_layernorm = FastRMSNorm(
            hidden_size, w_input_layernorm, eps=rms_norm_eps
        )
        self.post_attention_layernorm = FastRMSNorm(
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
            use_paged_attention=use_paged_attention,
        )

    def __call__(
        self,
        x: torch.Tensor,
        offset: int | list[int] | torch.Tensor,
        cache: TinyKvCache,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        r = self.self_attn(self.input_layernorm(x), offset, cache, mask)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        out = h + r
        return out


def is_qwen3_moe_sparse_layer(config: Any, layer_idx: int) -> bool:
    return (
        getattr(config, "num_experts", 0) > 0
        and layer_idx not in getattr(config, "mlp_only_layers", [])
        and (layer_idx + 1) % getattr(config, "decoder_sparse_step", 1) == 0
    )


def _pack_expert_weights(mlp_module: Any, proj_name: str, **kwargs) -> QuantizedWeights:
    if hasattr(mlp_module, "switch_mlp"):
        return QuantizedWeights.from_layer(
            getattr(mlp_module.switch_mlp, proj_name), **kwargs
        )
    stacked = torch.stack(
        [getattr(expert, proj_name).weight for expert in mlp_module.experts], dim=0
    )
    from types import SimpleNamespace

    return QuantizedWeights.from_layer(SimpleNamespace(weight=stacked), **kwargs)


class Qwen3ModelWeek3:
    def __init__(
        self,
        hf_model: Any,
        page_size: int = 128,
        enable_paged_attention: bool = True,
        use_torch_quantized_linear: bool = False,
    ):
        config = hf_model.config
        self.num_hidden_layers = config.num_hidden_layers
        self.hidden_size = config.hidden_size
        self.vocab_size = config.vocab_size
        self.page_size = page_size
        self.device = hf_model.model.embed_tokens.weight.device
        self.page_pools = [
            TinyKvPagedPool(page_size=self.page_size)
            for _ in range(self.num_hidden_layers)
        ]
        self.precision = torch.bfloat16
        self.use_torch_quantized_linear = use_torch_quantized_linear

        def week3_weights(layer: Any) -> QuantizedWeights:
            return QuantizedWeights.from_layer(
                layer,
                use_simdgroup_matmul=True,
                use_torch_quantized_linear=self.use_torch_quantized_linear,
            )

        self.embedding = QuantizedEmbedding(
            vocab_size=self.vocab_size,
            embedding_dim=self.hidden_size,
            weight=week3_weights(hf_model.model.embed_tokens),
        )
        self.layers_inner = []

        for i in range(config.num_hidden_layers):
            src_layer = hf_model.model.layers[i]
            wq = week3_weights(src_layer.self_attn.q_proj)
            wk = week3_weights(src_layer.self_attn.k_proj)
            wv = week3_weights(src_layer.self_attn.v_proj)
            wo = week3_weights(src_layer.self_attn.o_proj)
            if is_qwen3_moe_sparse_layer(config, i):
                mlp = Moe(
                    w_router=week3_weights(src_layer.mlp.gate),
                    w_gate=_pack_expert_weights(
                        src_layer.mlp,
                        "gate_proj",
                        use_simdgroup_matmul=True,
                        use_torch_quantized_linear=self.use_torch_quantized_linear,
                    ),
                    w_up=_pack_expert_weights(
                        src_layer.mlp,
                        "up_proj",
                        use_simdgroup_matmul=True,
                        use_torch_quantized_linear=self.use_torch_quantized_linear,
                    ),
                    w_down=_pack_expert_weights(
                        src_layer.mlp,
                        "down_proj",
                        use_simdgroup_matmul=True,
                        use_torch_quantized_linear=self.use_torch_quantized_linear,
                    ),
                    num_experts_per_tok=config.num_experts_per_tok,
                    norm_topk_prob=getattr(config, "norm_topk_prob", False),
                )
            else:
                mlp = Qwen3MLP(
                    config.hidden_size,
                    config.intermediate_size,
                    week3_weights(src_layer.mlp.gate_proj),
                    week3_weights(src_layer.mlp.up_proj),
                    week3_weights(src_layer.mlp.down_proj),
                )

            layer = Qwen3TransformerBlock(
                num_attention_heads=config.num_attention_heads,
                num_kv_heads=config.num_key_value_heads,
                hidden_size=config.hidden_size,
                head_dim=config.head_dim,
                rms_norm_eps=config.rms_norm_eps,
                wq=wq,
                wk=wk,
                wv=wv,
                wo=wo,
                q_norm=src_layer.self_attn.q_norm.weight,
                k_norm=src_layer.self_attn.k_norm.weight,
                w_input_layernorm=src_layer.input_layernorm.weight,
                w_post_attention_layernorm=src_layer.post_attention_layernorm.weight,
                mlp=mlp,
                max_seq_len=config.max_position_embeddings,
                theta=config.rope_theta,
                use_paged_attention=enable_paged_attention,
            )
            self.layers_inner.append(layer)
        self.norm = FastRMSNorm(
            config.hidden_size,
            weight=hf_model.model.norm.weight,
            eps=config.rms_norm_eps,
        )
        if not config.tie_word_embeddings:
            self.w_lm_head = week3_weights(hf_model.lm_head)
        else:
            self.w_lm_head = None
        self.hf_model = hf_model

    def create_kv_cache(self) -> list[TinyKvCache]:
        return [TinyKvPagedCache(pool=pool) for pool in self.page_pools]

    def __call__(
        self,
        inputs: torch.Tensor,
        offset: int | list[int] | torch.Tensor,
        cache: list[TinyKvCache],
        logits_to_keep: int | None = None,
    ) -> torch.Tensor:
        h = self.embedding(inputs)
        for layer in range(self.num_hidden_layers):
            h = self.layers_inner[layer](h, offset, cache[layer], mask="causal")
        if logits_to_keep is not None:
            if logits_to_keep <= 0:
                raise ValueError("logits_to_keep must be positive")
            h = h[:, -logits_to_keep:, :]
        h = self.norm(h)
        if self.w_lm_head is not None:
            return quantized_linear(h, self.w_lm_head)
        return self.embedding.as_linear(h)
