import pytest
import torch
import torch.nn.functional as F
from .tiny_llm_base import (
    softmax,
    linear,
    scaled_dot_product_attention_simple,
    SimpleMultiHeadAttention,
)
from .week1_utils import DEVICES, DTYPES, assert_close


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_task_1_softmax(device, dtype):
    x = torch.randn(3, 7, device=device, dtype=dtype) * 100
    assert_close(softmax(x, axis=-1), F.softmax(x, dim=-1))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [(), (2,), (2, 3), (2, 1, 3)])
@pytest.mark.parametrize("masked", [False, True])
def test_task_1_simple_attention(device, dtype, batch, masked):
    q, k, v = [torch.rand(*batch, 4, 6, device=device, dtype=dtype) for _ in range(3)]
    mask = torch.randn(4, 4, device=device, dtype=dtype) if masked else None
    assert_close(
        scaled_dot_product_attention_simple(q, k, v, scale=0.4, mask=mask),
        F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=0.4),
    )
    assert_close(
        scaled_dot_product_attention_simple(q, k, v),
        F.scaled_dot_product_attention(q, k, v),
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_task_2_linear(device, dtype):
    x = torch.randn(2, 3, 7, device=device, dtype=dtype)
    w = torch.randn(5, 7, device=device, dtype=dtype)
    b = torch.randn(5, device=device, dtype=dtype)
    assert_close(linear(x, w, b), F.linear(x, w, b))
    assert_close(linear(x, w), F.linear(x, w))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_task_2_simple_multi_head_attention(device, dtype):
    layer = torch.nn.MultiheadAttention(
        12, 3, bias=False, batch_first=True, device=device, dtype=dtype
    )
    wq, wk, wv = layer.in_proj_weight.detach().chunk(3)
    q, k, v = [torch.randn(2, 5, 12, device=device, dtype=dtype) for _ in range(3)]
    mask = torch.randn(5, 5, device=device, dtype=dtype)
    actual = SimpleMultiHeadAttention(
        12, 3, wq, wk, wv, layer.out_proj.weight.detach()
    )(q, k, v, mask)
    expected, _ = layer(q, k, v, attn_mask=mask, need_weights=False)
    assert_close(actual, expected)
