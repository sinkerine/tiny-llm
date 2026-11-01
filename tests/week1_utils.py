import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA unavailable"
        ),
    ),
]
DTYPES = [torch.float32, torch.float16, torch.bfloat16]


def assert_close(actual, expected):
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    assert actual.device == expected.device
    tolerance = {
        torch.float32: (2e-5, 2e-6),
        torch.float16: (0.05, 0.003),
        torch.bfloat16: (0.08, 0.03),
    }
    rtol, atol = tolerance[actual.dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def tiny_model(device, dtype, *, tied=True):
    config = Qwen3Config(
        vocab_size=43,
        hidden_size=24,
        intermediate_size=48,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        rope_theta=10000,
        tie_word_embeddings=tied,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    return Qwen3ForCausalLM(config).to(device=device, dtype=dtype).eval()


def attention_kwargs(model, index=0):
    layer = model.model.layers[index].self_attn
    config = model.config
    return dict(
        hidden_size=config.hidden_size,
        num_heads=config.num_attention_heads,
        num_kv_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        wq=layer.q_proj.weight,
        wk=layer.k_proj.weight,
        wv=layer.v_proj.weight,
        wo=layer.o_proj.weight,
        q_norm=layer.q_norm.weight,
        k_norm=layer.k_norm.weight,
        max_seq_len=config.max_position_embeddings,
        theta=config.rope_theta,
        rms_norm_eps=config.rms_norm_eps,
    )


def block_kwargs(model):
    layer = model.model.layers[0]
    kwargs = attention_kwargs(model)
    kwargs["num_attention_heads"] = kwargs.pop("num_heads")
    kwargs.update(
        intermediate_size=model.config.intermediate_size,
        w_gate=layer.mlp.gate_proj.weight,
        w_up=layer.mlp.up_proj.weight,
        w_down=layer.mlp.down_proj.weight,
        w_input_layernorm=layer.input_layernorm.weight,
        w_post_attention_layernorm=layer.post_attention_layernorm.weight,
    )
    return kwargs
