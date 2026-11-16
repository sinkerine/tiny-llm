"""CUDA translations of upstream reference-only scheduling regressions."""

import pytest
import torch
from tiny_llm_ref.qwen3_week2 import Qwen3ModelWeek2
from tiny_llm_ref.quantize import quantized_matmul
from tiny_llm_ref.week2_kernels import (
    FastRMSNorm,
    FastRoPE,
    _qk_norm_rope4,
    qk_norm_rope,
    decode_attention_online,
    scaled_dot_product_attention,
)
from .week2_utils import CUDA, tiny_model, packed_fixture, assert_close

pytestmark = CUDA


def fixture(shape, phase):
    from math import prod

    return (
        (torch.arange(prod(shape), device="cuda").float() * 0.017 + phase)
        .sin()
        .reshape(shape)
        .bfloat16()
    )


@pytest.mark.parametrize("context", [256, 257, 8192])
def test_online_decode_matches_readable_with_padded_cache(context):
    q = fixture((1, 8, 1, 128), 0.1)
    k = fixture((1, 2, context + 17, 128), 0.7)[:, :, :context]
    v = fixture((1, 2, context + 17, 128), 1.3)[:, :, :context]
    mask = torch.where(torch.arange(context, device="cuda") % 7 == 0, -torch.inf, 0.0)
    assert not k.is_contiguous()
    actual = decode_attention_online(q, k, v, 128**-0.5, mask)
    expected = scaled_dot_product_attention(q, k, v, 128**-0.5, mask)
    assert_close(actual, expected, atol=0.02, rtol=0.02)


def test_online_decode_fully_masked_partitions_are_finite_zero():
    q = fixture((1, 4, 1, 128), 0.1)
    k = fixture((1, 1, 257, 128), 0.7)
    v = fixture(k.shape, 1.3)
    out = decode_attention_online(
        q, k, v, 128**-0.5, torch.full((257,), -torch.inf, device="cuda")
    )
    assert torch.isfinite(out).all() and (out == 0).all()


@pytest.mark.parametrize("heads", [8, 32])
@pytest.mark.parametrize("materialized", [False, True])
def test_qk_norm_rope_eager_fallback_and_explicit_fusion(heads, materialized):
    x = fixture((1, 1, heads, 128), 0.1)
    weight = fixture((128,), 0.7)
    norm = FastRMSNorm(128, weight)
    rope = FastRoPE(128, 32768, 1000000)
    normalized = norm(x)
    if materialized:
        torch.cuda.synchronize()
        normalized = normalized.clone()
    expected = rope(normalized, 8192)
    assert_close(_qk_norm_rope4(normalized, rope, 8192), expected, atol=0.02, rtol=0.02)
    assert_close(qk_norm_rope(x, norm, rope, 8192), expected, atol=0.02, rtol=0.02)


@pytest.mark.parametrize("rows,outputs", [(32, 64), (33, 65)])
def test_quantized_prefill_full_and_partial_tiles_match_torch(rows, outputs):
    x, p, s, b, dense = packed_fixture(rows, outputs)
    out = quantized_matmul(s, b, 128, 4, x, p, True, use_simdgroup=True)
    assert_close(out, torch.nn.functional.linear(x, dense), atol=0.25, rtol=0.01)


@torch.inference_mode()
def test_selected_default_and_teaching_checkpoint_keep_cached_decode_outputs():
    hf = tiny_model()
    default = Qwen3ModelWeek2(hf)
    teaching = Qwen3ModelWeek2(hf, checkpoint="tiled-prefill")
    a = default.create_kv_cache(capacity=13)
    b = teaching.create_kv_cache(capacity=13)
    tokens = torch.arange(1, 11, device="cuda")[None]
    for offset in (0, 10, 11, 12):
        assert_close(
            default(tokens, offset, a, logits_to_keep=1),
            teaching(tokens, offset, b, logits_to_keep=1),
            atol=0.1,
            rtol=0.05,
        )
        tokens = torch.tensor([[13]], device="cuda")
    assert (
        default.layers_inner[0].self_attn.attention_dispatch_counts[
            "native_online_decode"
        ]
        == 3
    )
    assert not teaching.layers_inner[0].self_attn.use_native_online_decode
    assert (
        not default.use_decode_submit18
    )  # CUDA launches eagerly; MLX queue flushing is inapplicable.


@torch.inference_mode()
def test_selected_shape_guard_reaches_fused_qk_model_path():
    from transformers import Qwen3ForCausalLM

    config = tiny_model().config
    config.num_attention_heads = 32
    config.num_key_value_heads = 8
    config.num_hidden_layers = 1
    torch.manual_seed(7)
    hf = Qwen3ForCausalLM(config).cuda().bfloat16().eval()
    optimized = Qwen3ModelWeek2(hf)
    control = Qwen3ModelWeek2(hf, use_fused_qk_norm_rope4=False)
    a, b = optimized.create_kv_cache(capacity=3), control.create_kv_cache(capacity=3)
    prompt = torch.tensor([[1, 2]], device="cuda")
    assert_close(optimized(prompt, 0, a), control(prompt, 0, b), atol=0.1, rtol=0.05)
    token = torch.tensor([[3]], device="cuda")
    assert_close(optimized(token, 2, a), control(token, 2, b), atol=0.1, rtol=0.05)
    assert (
        optimized.layers_inner[0].self_attn.attention_dispatch_counts[
            "fused_qk_norm_rope4"
        ]
        == 1
    )
    assert (
        control.layers_inner[0].self_attn.attention_dispatch_counts[
            "fused_qk_norm_rope4"
        ]
        == 0
    )


def test_reference_controls_can_be_disabled_and_reject_invalid_values():
    hf = tiny_model()
    m = Qwen3ModelWeek2(
        hf, use_native_online_decode=False, use_fused_qk_norm_rope4=False
    )
    assert not m.layers_inner[0].self_attn.use_native_online_decode
    for key in (
        "use_native_online_decode",
        "use_fused_qk_norm_rope4",
        "use_decode_submit18",
    ):
        with pytest.raises(ValueError):
            Qwen3ModelWeek2(hf, **{key: 1})
    with pytest.raises(ValueError, match="eager"):
        Qwen3ModelWeek2(hf, use_decode_submit18=True)
