"""Week 2 Day 5: tiled GQA, mask tails, fully masked rows and integration."""

import pytest
import torch
from .tiny_llm_base import Qwen3ModelWeek2, dense_prefill_attention_mma
from .week2_utils import CUDA, tiny_model, assert_close

pytestmark = CUDA


@pytest.mark.parametrize("L,S", [(9, 9), (17, 35), (33, 33)])
@pytest.mark.parametrize("mask_kind", ["none", "causal", "additive", "fully-masked"])
def test_task_1_online_softmax_gqa_partial_tiles(L, S, mask_kind):
    torch.manual_seed(2026)
    q = torch.randn(2, 4, L, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(2, 2, S, 128, device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    mask = None
    expected_mask = None
    if mask_kind == "causal":
        mask = "causal"
        expected_mask = (
            torch.arange(S, device="cuda")[None]
            <= torch.arange(L, device="cuda")[:, None] + S - L
        )
    elif mask_kind in ("additive", "fully-masked"):
        mask = torch.zeros(2, 1, L, S, device="cuda")
        mask[..., -3:] = -torch.inf
        if mask_kind == "fully-masked":
            mask[:, :, 0, :] = -torch.inf
        expected_mask = mask
    actual = dense_prefill_attention_mma(q, k, v, 128**-0.5, mask)
    expected = torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=expected_mask, enable_gqa=True
    )
    assert_close(actual, expected, atol=0.02, rtol=0.03)
    assert torch.isfinite(actual).all()
    if mask_kind == "fully-masked":
        assert (actual[:, :, 0] == 0).all()


@pytest.mark.parametrize("checkpoint", ["tiled-prefill", "selected"])
@torch.inference_mode()
def test_task_3_selected_model_and_short_query_fallback(checkpoint):
    hf = tiny_model()
    model = Qwen3ModelWeek2(hf, checkpoint=checkpoint)
    control = Qwen3ModelWeek2(hf, checkpoint="swiglu")
    tokens = torch.arange(1, 11, device="cuda")[None]
    a = model.create_kv_cache(capacity=11)
    b = control.create_kv_cache(capacity=11)
    assert_close(model(tokens, 0, a), control(tokens, 0, b), atol=0.1, rtol=0.05)
    assert_close(
        model(tokens[:, :1], 10, a), control(tokens[:, :1], 10, b), atol=0.1, rtol=0.05
    )
    for layer in model.layers_inner:
        assert layer.self_attn.attention_dispatch_counts["tiled_prefill"] == 1
        assert layer.self_attn.attention_dispatch_counts["tiled_prefill_fallback"] == 1
