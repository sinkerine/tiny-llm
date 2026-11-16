"""Week 2 Day 2: selected rows, packed scalar/warp kernels and model wiring."""

import pytest
import torch
from .tiny_llm_base import (
    Qwen3ModelWeek2,
    QuantizedEmbedding,
    QuantizedWeights,
    dequantize_weights,
    quantized_matmul_vanilla,
    quantized_matvec_custom,
)
from .week2_utils import CUDA, packed_fixture, tiny_model, assert_close

pytestmark = CUDA


def test_task_1_dequantize_and_embedding_selected_rows():
    x, packed, s, b, dense = packed_fixture(outputs=7)
    assert_close(dequantize_weights(packed, s, b, 128, 4), dense, 0, 0)
    embedding = QuantizedEmbedding(7, 256, QuantizedWeights(s, b, 128, 4, packed))
    ids = torch.tensor([[1, 4]], device="cuda")
    assert_close(embedding(ids), dense[ids], 0, 0)


@pytest.mark.parametrize(
    "rows,outputs,width", [(1, 5, 128), (8, 65, 256), (1, 97, 2560)]
)
@pytest.mark.parametrize("kind", ["scalar", "warp"])
def test_task_3_packed_projection_partial_tiles(rows, outputs, width, kind):
    x, p, s, b, dense = packed_fixture(rows, outputs, width)
    op = quantized_matvec_custom if kind == "warp" else quantized_matmul_vanilla
    actual = op(s, b, 128, 4, x, p, True)
    assert_close(actual, torch.nn.functional.linear(x, dense), atol=0.5, rtol=0.03)


def test_task_3_invalid_wrapper_inputs():
    x, p, s, b, _ = packed_fixture(rows=10)
    with pytest.raises(ValueError):
        quantized_matvec_custom(s, b, 128, 4, x, p, True)
    for bits, transpose in [(8, True), (4, False)]:
        with pytest.raises(ValueError):
            quantized_matmul_vanilla(s, b, 128, bits, x, p, transpose)


def test_task_3_uses_current_cuda_stream():
    x, p, s, b, dense = packed_fixture(rows=1, outputs=17)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.cuda._sleep(1000000)
        changed = x + 1
        actual = quantized_matvec_custom(s, b, 128, 4, changed, p, True)
        expected = torch.nn.functional.linear(changed, dense)
    stream.synchronize()
    assert_close(actual, expected, atol=0.25, rtol=0.03)


@pytest.mark.parametrize("tied", [True, False])
@torch.inference_mode()
def test_task_4_model_matches_dense_quantized_control(tied):
    hf = tiny_model(tied=tied)
    model = Qwen3ModelWeek2(hf, checkpoint="quantized-matvec")
    control = Qwen3ModelWeek2(
        hf, checkpoint="quantized-matvec", use_torch_quantized_linear=True
    )
    assert isinstance(model.layers_inner[0].self_attn.wq, QuantizedWeights)
    a = model.create_kv_cache(capacity=4)
    b = control.create_kv_cache(capacity=4)
    tokens = torch.tensor([[1, 2, 3]], device="cuda")
    assert_close(model(tokens, 0, a), control(tokens, 0, b), atol=0.1, rtol=0.05)
    assert_close(
        model(tokens[:, :1], 3, a), control(tokens[:, :1], 3, b), atol=0.1, rtol=0.05
    )
