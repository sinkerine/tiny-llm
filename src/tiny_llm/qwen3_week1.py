import torch
from .basics import linear, silu
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from typing import Any
from .embedding import Embedding


class Qwen3MultiHeadAttention:
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        max_seq_len: int = 32768,
        theta: int = 1000000,
        rms_norm_eps: float = 1e-5,
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
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        B, L, _ = x.shape
        q = linear(x, self.wq)
        k = linear(x, self.wk)
        v = linear(x, self.wv)
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B, L, self.num_kv_heads, self.head_dim)
        v = v.reshape(B, L, self.num_kv_heads, self.head_dim)
        q = self.q_norm(q)
        k = self.k_norm(k)
        q = self.rope(q, slice(0, L))
        k = self.rope(k, slice(0, L))
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        attention_mask = mask if isinstance(mask, torch.Tensor) else None
        is_causal = isinstance(mask, str) and mask == "causal"
        out = torch.nn.functional.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attention_mask,
            is_causal=is_causal,
            dropout_p=0.0,
            enable_gqa=True,
        )
        out = (
            out.transpose(1, 2)
            .reshape(B, L, self.num_heads * self.head_dim)
            .to(dtype=x.dtype)
        )
        return linear(out, self.wo)


class Qwen3MLP:
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
    ):
        self.w_gate = w_gate
        self.w_up = w_up
        self.w_down = w_down

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        gate = silu(linear(x, self.w_gate))
        up = linear(x, self.w_up)
        up_gated = gate * up
        return linear(up_gated, self.w_down)


class Qwen3TransformerBlock:
    def __init__(
        self,
        num_attention_heads: int,
        num_kv_heads: int,
        hidden_size: int,
        head_dim: int,
        intermediate_size: int,
        rms_norm_eps: float,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        q_norm: torch.Tensor,
        k_norm: torch.Tensor,
        w_gate: torch.Tensor,
        w_up: torch.Tensor,
        w_down: torch.Tensor,
        w_input_layernorm: torch.Tensor,
        w_post_attention_layernorm: torch.Tensor,
        max_seq_len: int = 32768,
        theta: int = 1000000,
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
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x, mask)
        x = x + residual
        post_residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = x + post_residual
        return x


class Qwen3ModelWeek1:
    def __init__(self, hf_model: Any):
        config = hf_model.config
        self.embed_tokens = Embedding(
            config.vocab_size, config.hidden_size, hf_model.model.embed_tokens.weight
        )
        self.norm = RMSNorm(
            config.hidden_size, hf_model.model.norm.weight, config.rms_norm_eps
        )
        self.layers = []
        for layer in hf_model.model.layers:
            self.layers.append(
                Qwen3TransformerBlock(
                    config.num_attention_heads,
                    config.num_key_value_heads,
                    config.hidden_size,
                    config.head_dim,
                    config.intermediate_size,
                    config.rms_norm_eps,
                    layer.self_attn.q_proj.weight,
                    layer.self_attn.k_proj.weight,
                    layer.self_attn.v_proj.weight,
                    layer.self_attn.o_proj.weight,
                    layer.self_attn.q_norm.weight,
                    layer.self_attn.k_norm.weight,
                    layer.mlp.gate_proj.weight,
                    layer.mlp.up_proj.weight,
                    layer.mlp.down_proj.weight,
                    layer.input_layernorm.weight,
                    layer.post_attention_layernorm.weight,
                    config.max_position_embeddings,
                    config.rope_theta,
                )
            )
        self.tie_word_embeddings = config.tie_word_embeddings
        self.lm_head_weight = hf_model.lm_head.weight

    def __call__(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed_tokens(inputs)
        for layer in self.layers:
            x = layer(x, mask="causal")
        x = self.norm(x)
        if self.tie_word_embeddings:
            x = self.embed_tokens.as_linear(x)
        else:
            x = linear(x, self.lm_head_weight)
        return x
