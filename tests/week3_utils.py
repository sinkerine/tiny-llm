"""Download-free CUDA fixtures and cached HF model helpers for Week 3."""

import os
from pathlib import Path

import huggingface_hub
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

from .week2_utils import CUDA, assert_close, packed_fixture, tiny_model

HF_CACHE_DIR = Path(
    os.environ.get("HF_HUB_CACHE", str(Path.home() / ".cache" / "tiny-llm"))
)


def tiny_qwen3_model(
    num_hidden_layers: int = 1,
    *,
    head_dim: int = 128,
    num_attention_heads: int = 2,
    num_key_value_heads: int = 1,
    tie_word_embeddings: bool = True,
    seed: int = 0,
    device: str = "cuda",
):
    torch.manual_seed(seed)
    hidden_size = max(128, num_attention_heads * head_dim)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=hidden_size,
        intermediate_size=hidden_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        head_dim=head_dim,
        max_position_embeddings=256,
        rope_theta=10000,
        tie_word_embeddings=tie_word_embeddings,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).to(device=device, dtype=torch.bfloat16).eval()


def tiny_qwen3_moe_model(
    *,
    num_hidden_layers: int = 3,
    num_experts: int = 4,
    num_experts_per_tok: int = 2,
    decoder_sparse_step: int = 1,
    mlp_only_layers: list[int] | None = None,
    norm_topk_prob: bool = True,
    seed: int = 0,
    device: str = "cuda",
):
    torch.manual_seed(seed)
    config = Qwen3MoeConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=128,
        moe_intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=256,
        rope_theta=10000,
        num_experts=num_experts,
        num_experts_per_tok=num_experts_per_tok,
        decoder_sparse_step=decoder_sparse_step,
        mlp_only_layers=mlp_only_layers or [],
        norm_topk_prob=norm_topk_prob,
        tie_word_embeddings=True,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return Qwen3MoeForCausalLM(config).to(device=device, dtype=torch.bfloat16).eval()


def hf_model_exists(repo_id: str) -> bool:
    try:
        cached = huggingface_hub.try_to_load_from_cache(
            repo_id=repo_id,
            filename="config.json",
            cache_dir=str(HF_CACHE_DIR),
        )
        return isinstance(cached, str)
    except Exception:
        return False


def load_hf_qwen3(repo_id: str, device: str = "cuda"):
    tokenizer = AutoTokenizer.from_pretrained(
        repo_id, cache_dir=str(HF_CACHE_DIR), local_files_only=True
    )
    model = (
        AutoModelForCausalLM.from_pretrained(
            repo_id,
            dtype=torch.bfloat16,
            cache_dir=str(HF_CACHE_DIR),
            local_files_only=True,
        )
        .to(device=device)
        .eval()
    )
    return model, tokenizer


__all__ = [
    "CUDA",
    "HF_CACHE_DIR",
    "assert_close",
    "hf_model_exists",
    "load_hf_qwen3",
    "packed_fixture",
    "tiny_model",
    "tiny_qwen3_model",
    "tiny_qwen3_moe_model",
]
