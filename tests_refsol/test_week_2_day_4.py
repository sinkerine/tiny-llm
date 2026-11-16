"""Week 2 Day 4: fused model primitives and cumulative checkpoints."""

import pytest
import torch
from .tiny_llm_base import Qwen3ModelWeek2, FastRMSNorm, FastRoPE, swiglu
from .week2_utils import CUDA, tiny_model, assert_close

pytestmark = CUDA


@pytest.mark.parametrize("width", [128, 1024, 4096, 4352])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_task_1_rms_norm(width, dtype):
    x = torch.randn(3, width, device="cuda", dtype=dtype)
    w = torch.randn(width, device="cuda", dtype=dtype)
    norm = FastRMSNorm(width, w)
    expected = (
        x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-5)
    ).to(dtype) * w
    assert_close(
        norm(x), expected, atol=0.03 if dtype == torch.bfloat16 else 0.003, rtol=0.01
    )
    key = "register_cached" if width <= 4096 else "fixed_width_fallback"
    assert norm.dispatch_counts[key] == 1


@pytest.mark.parametrize("traditional", [False, True])
@pytest.mark.parametrize("offsets", [0, [3, 7]])
def test_task_2_rope_per_batch_offsets(traditional, offsets):
    x = torch.randn(2, 5, 3, 128, device="cuda", dtype=torch.bfloat16)
    rope = FastRoPE(128, 128, 10000, traditional)
    off = torch.as_tensor(offsets, device="cuda").reshape(-1, 1)
    angle = (off + torch.arange(5, device="cuda"))[:, :, None] * 10000 ** (
        -torch.arange(64, device="cuda").float() / 64
    )
    co, si = angle.cos().bfloat16()[:, :, None], angle.sin().bfloat16()[:, :, None]
    u, v = (x[..., ::2], x[..., 1::2]) if traditional else x.chunk(2, -1)
    real, imag = u * co - v * si, u * si + v * co
    expected = (
        torch.stack((real, imag), -1).flatten(-2)
        if traditional
        else torch.cat((real, imag), -1)
    )
    assert_close(rope(x, offsets), expected, atol=0.03, rtol=0.02)


def test_task_3_swiglu_matches_stable_equation():
    g = torch.linspace(-100, 100, 4096, device="cuda").bfloat16()
    u = torch.randn_like(g)
    assert_close(
        swiglu(g, u),
        torch.nn.functional.silu(g.float()).bfloat16() * u,
        atol=0.03,
        rtol=0.02,
    )


@pytest.mark.parametrize("checkpoint", ["rmsnorm", "rope", "swiglu"])
@torch.inference_mode()
def test_task_4_cumulative_model(checkpoint):
    hf = tiny_model()
    model = Qwen3ModelWeek2(hf, checkpoint=checkpoint)
    control = Qwen3ModelWeek2(hf, checkpoint="simd-matmul")
    tokens = torch.arange(1, 11, device="cuda")[None]
    a = model.create_kv_cache(capacity=11)
    b = control.create_kv_cache(capacity=11)
    assert_close(model(tokens, 0, a), control(tokens, 0, b), atol=0.15, rtol=0.05)
    assert_close(
        model(tokens[:, :1], 10, a), control(tokens[:, :1], 10, b), atol=0.15, rtol=0.05
    )
