"""Download-free CUDA fixtures shared by learner and reference checkpoints."""

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM
from tiny_llm_cuda.packing import pack_weights

CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")


def tiny_model(seed=0, device="cuda", tied=True):
    torch.manual_seed(seed)
    config = Qwen3Config(
        vocab_size=137,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=128,
        max_position_embeddings=128,
        rope_theta=10000,
        tie_word_embeddings=tied,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).to(device=device, dtype=torch.bfloat16).eval()


def packed_fixture(rows=10, outputs=97, width=256):
    torch.manual_seed(2026)
    x = torch.randn(rows, width, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(outputs, width, device="cuda", dtype=torch.bfloat16)
    packed, scales, biases = pack_weights(weight)
    shifts = torch.arange(0, 32, 4, device="cuda")
    q = ((packed.long()[..., None] >> shifts) & 15).flatten(-2).float()
    dense = (
        q * scales.repeat_interleave(128, -1).float()
        + biases.repeat_interleave(128, -1).float()
    ).bfloat16()
    return x, packed, scales, biases, dense


def assert_close(actual, expected, atol=0.08, rtol=0.03):
    assert actual is not None, "implement the learner-owned checkpoint"
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
