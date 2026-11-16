"""Week 2 Day 3: cooperative BF16 WMMA prefill with packed weights."""

import pytest
import torch
from .tiny_llm_base import Qwen3ModelWeek2, quantized_matmul, quantized_matmul_vanilla
from .week2_utils import CUDA, packed_fixture, tiny_model, assert_close

pytestmark = CUDA


@pytest.mark.parametrize("rows,outputs", [(10, 97), (17, 33), (32, 64)])
def test_task_2_wmma_handles_partial_tiles(rows, outputs):
    x, p, s, b, dense = packed_fixture(rows, outputs)
    tiled = quantized_matmul(s, b, 128, 4, x, p, True, use_simdgroup=True)
    readable = quantized_matmul_vanilla(s, b, 128, 4, x, p, True)
    assert_close(tiled, readable, atol=0.25, rtol=0.02)
    assert_close(tiled, torch.nn.functional.linear(x, dense), atol=0.25, rtol=0.02)


@pytest.mark.parametrize("seed", [0, 4])
@torch.inference_mode()
def test_task_3_wmma_model_prefill_and_decode(seed):
    hf = tiny_model(seed)
    tiled = Qwen3ModelWeek2(hf, checkpoint="simd-matmul")
    control = Qwen3ModelWeek2(hf, checkpoint="quantized-matvec")
    a = tiled.create_kv_cache(capacity=11)
    b = control.create_kv_cache(capacity=11)
    tokens = torch.arange(1, 11, device="cuda")[None]
    assert_close(tiled(tokens, 0, a), control(tokens, 0, b), atol=0.1, rtol=0.05)
    assert_close(
        tiled(tokens[:, :1], 10, a), control(tokens[:, :1], 10, b), atol=0.1, rtol=0.05
    )
