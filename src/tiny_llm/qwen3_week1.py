import torch
from .basics import linear, silu
from .attention import scaled_dot_product_attention_grouped
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
        self.q_norm = q_norm
        self.k_norm = k_norm
        self.rope = RoPE(head_dim, max_seq_len, theta)

    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        B,L,_ = x.shape
        q = linear(x, self.wq)
        k = linear(x, self.wk)
        v = linear(x, self.wv)
        q = q.reshape(B, L, self.num_heads, self.head_dim)
        k = k.reshape(B,L,self.num_kv_heads, self.head_dim)
        v = v.reshape(B,L,self.num_kv_heads, self.head_dim)
        q = torch.nn.functional.rms_norm(q,(self.head_dim,), self.q_norm, self.rms_norm_eps)
        k = torch.nn.functional.rms_norm(k,(self.head_dim,), self.k_norm, self.rms_norm_eps)
        q = self.rope(q, slice(0, L))
        k = self.rope(k, slice(0, L))
        q = q.transpose(1,2).float()
        k = k.transpose(1,2).float()
        v = v.transpose(1,2).float()
        out = scaled_dot_product_attention_grouped(q,k,v,mask=mask)
        out = out.transpose(1,2).reshape(B,L,self.num_heads * self.head_dim).to(dtype=x.dtype)
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
        pass

    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | str | None = None,
    ) -> torch.Tensor:
        pass


class Qwen3ModelWeek1:
    def __init__(self, hf_model: Any):
        pass

    def __call__(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        pass
