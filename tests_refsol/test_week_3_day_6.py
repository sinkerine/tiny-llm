"""Optional Week 3 Day 6 MoE tests."""

from types import SimpleNamespace

import torch

from tiny_llm_cuda.packing import pack_weights
from .tiny_llm_base import (
    Moe,
    QuantizedWeights,
    dequantize_weights,
    dispatch_model,
    grouped_expert_linear,
    route_topk,
)
from .week3_utils import CUDA, assert_close, tiny_qwen3_moe_model

pytestmark = CUDA


def _quantized_3d(weight: torch.Tensor) -> QuantizedWeights:
    packed, scales, biases = pack_weights(weight, group_size=128, bits=4)
    return QuantizedWeights(scales=scales, biases=biases, group_size=128, bits=4, weight=packed)


def test_task_1_grouped_expert_linear():
    torch.manual_seed(1)
    scale = 0.25
    x = torch.randn((2, 3, 128), device="cuda", dtype=torch.bfloat16) * scale
    w_experts = torch.randn((3, 64, 128), device="cuda", dtype=torch.bfloat16) * scale
    expert_ids = torch.tensor([[2, 0, 1], [1, 2, 0]], device="cuda", dtype=torch.int64)

    qw = _quantized_3d(w_experts)
    out = grouped_expert_linear(x, qw, expert_ids)

    dense = dequantize_weights(qw.weight, qw.scales, qw.biases, 128, 4)
    expected = torch.empty((2, 3, 64), device="cuda", dtype=torch.bfloat16)
    for b in range(2):
        for s in range(3):
            e = int(expert_ids[b, s].item())
            expected[b, s] = torch.nn.functional.linear(x[b, s], dense[e])

    assert out.shape == (2, 3, 64)
    assert_close(out, expected, atol=2e-2, rtol=3e-2)


def test_task_2_router_topk():
    torch.manual_seed(2)
    scale = 0.25
    x = torch.randn((2, 2, 128), device="cuda", dtype=torch.bfloat16) * scale
    w_router = torch.randn((4, 128), device="cuda", dtype=torch.bfloat16) * scale
    qw = QuantizedWeights.from_layer(SimpleNamespace(weight=w_router))

    router_probs, expert_ids, expert_scores = route_topk(x, qw, top_k=2)
    _, _, normalized_scores = route_topk(x, qw, top_k=2, norm_topk_prob=True)

    dense_router = dequantize_weights(qw.weight, qw.scales, qw.biases, 128, 4)
    expected_logits = torch.nn.functional.linear(x, dense_router)
    expected_probs = torch.softmax(expected_logits.float(), dim=-1).to(torch.bfloat16)
    expected_scores, expected_ids = torch.topk(expected_probs, k=2, dim=-1)
    expected_norm = (
        expected_scores.float() / expected_scores.float().sum(dim=-1, keepdim=True)
    ).to(torch.bfloat16)

    assert router_probs.shape == (2, 2, 4)
    assert expert_ids.shape == (2, 2, 2)
    assert expert_scores.shape == (2, 2, 2)
    assert torch.sort(expert_ids, dim=-1).values.tolist() == torch.sort(
        expected_ids, dim=-1
    ).values.tolist()
    assert_close(router_probs, expected_probs, atol=2e-2, rtol=3e-2)
    assert_close(expert_scores, expected_scores, atol=2e-2, rtol=3e-2)
    assert_close(normalized_scores, expected_norm, atol=2e-2, rtol=3e-2)


def test_task_3_moe():
    torch.manual_seed(3)
    scale = 0.25
    x = torch.randn((2, 3, 128), device="cuda", dtype=torch.bfloat16) * scale
    w_router = QuantizedWeights.from_layer(
        SimpleNamespace(weight=torch.randn((3, 128), device="cuda", dtype=torch.bfloat16) * scale)
    )
    w_gate = _quantized_3d(torch.randn((3, 128, 128), device="cuda", dtype=torch.bfloat16) * scale)
    w_up = _quantized_3d(torch.randn((3, 128, 128), device="cuda", dtype=torch.bfloat16) * scale)
    w_down = _quantized_3d(torch.randn((3, 128, 128), device="cuda", dtype=torch.bfloat16) * scale)

    moe = Moe(
        w_router=w_router,
        w_gate=w_gate,
        w_up=w_up,
        w_down=w_down,
        num_experts_per_tok=2,
        norm_topk_prob=True,
    )
    out = moe(x)

    # Reference dense computation using dequantized weights
    r_w = dequantize_weights(w_router.weight, w_router.scales, w_router.biases, 128, 4)
    g_w = dequantize_weights(w_gate.weight, w_gate.scales, w_gate.biases, 128, 4)
    u_w = dequantize_weights(w_up.weight, w_up.scales, w_up.biases, 128, 4)
    d_w = dequantize_weights(w_down.weight, w_down.scales, w_down.biases, 128, 4)
    probs = torch.softmax(torch.nn.functional.linear(x, r_w).float(), dim=-1)
    scores, ids = torch.topk(probs, k=2, dim=-1)
    scores = (scores / scores.sum(dim=-1, keepdim=True)).to(torch.bfloat16)
    expected = torch.zeros_like(x)
    for b in range(2):
        for s in range(3):
            acc = torch.zeros(128, device="cuda", dtype=torch.bfloat16)
            for k in range(2):
                e = int(ids[b, s, k].item())
                g = torch.nn.functional.linear(x[b, s], g_w[e])
                u = torch.nn.functional.linear(x[b, s], u_w[e])
                h = torch.nn.functional.silu(g.float()).to(torch.bfloat16) * u
                acc = acc + scores[b, s, k] * torch.nn.functional.linear(h, d_w[e])
            expected[b, s] = acc

    assert out.shape == x.shape
    assert_close(out, expected, atol=3e-2, rtol=5e-2)


@torch.inference_mode()
def test_task_4_model_dispatches_dense_and_sparse_moe_layers():
    hf_model = tiny_qwen3_moe_model(
        num_hidden_layers=2,
        num_experts=3,
        num_experts_per_tok=2,
        decoder_sparse_step=2,
        mlp_only_layers=[0],
        norm_topk_prob=True,
        seed=6,
    )
    model = dispatch_model(
        "qwen3-30b-a3b",
        hf_model,
        week=3,
        page_size=16,
        enable_paged_attention=False,
        use_torch_quantized_linear=True,
    )
    assert not isinstance(model.layers_inner[0].mlp, Moe)
    assert isinstance(model.layers_inner[1].mlp, Moe)

    tokens = torch.tensor([[1, 5, 7]], dtype=torch.long, device="cuda")
    actual = model(tokens, 0, model.create_kv_cache())
    assert actual.shape == (1, 3, 128)
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
