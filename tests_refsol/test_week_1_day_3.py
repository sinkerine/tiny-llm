import pytest
import torch
import torch.nn.functional as F
from .tiny_llm_base import (
    scaled_dot_product_attention_grouped,
    causal_mask,
    qwen3_week1,
)
from .week1_utils import DEVICES, DTYPES, assert_close, tiny_model, attention_kwargs


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("batch", [(), (2,), (2, 3), (2, 1, 3)])
@pytest.mark.parametrize("mask_kind", ["none", "additive", "causal"])
def test_task_1_and_2_grouped_attention(device, dtype, batch, mask_kind):
    q = torch.rand(*batch, 6, 3, 8, device=device, dtype=dtype)
    k, v = [torch.rand(*batch, 2, 7, 8, device=device, dtype=dtype) for _ in range(2)]
    mask = (
        torch.randn(*batch, 6, 3, 7, device=device, dtype=dtype)
        if mask_kind == "additive"
        else None
    )
    oracle_mask = mask
    if mask_kind == "causal":
        oracle_mask = (
            torch.arange(7, device=device)[None, :]
            <= torch.arange(4, 7, device=device)[:, None]
        )
        mask = "causal"
    expected = F.scaled_dot_product_attention(
        q,
        k.repeat_interleave(3, dim=-3),
        v.repeat_interleave(3, dim=-3),
        attn_mask=oracle_mask,
        scale=0.4,
    )
    assert_close(
        scaled_dot_product_attention_grouped(q, k, v, scale=0.4, mask=mask), expected
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_task_2_causal_mask(dtype):
    expected = torch.tensor([[0, 0, 0, -torch.inf], [0, 0, 0, 0]], dtype=dtype)
    torch.testing.assert_close(causal_mask(2, 4, dtype).cpu(), expected)


def test_task_1_invalid_heads_and_causal_domain():
    with pytest.raises((AssertionError, ValueError)):
        scaled_dot_product_attention_grouped(
            torch.ones(5, 3, 4), torch.ones(2, 3, 4), torch.ones(2, 3, 4)
        )
    with pytest.raises(ValueError):
        scaled_dot_product_attention_grouped(
            torch.ones(2, 5, 4), torch.ones(1, 3, 4), torch.ones(1, 3, 4), mask="causal"
        )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@torch.inference_mode()
def test_task_3_qwen_attention(device, dtype):
    model = tiny_model(device, dtype)
    layer = model.model.layers[0].self_attn
    x = torch.randn(2, 5, model.config.hidden_size, device=device, dtype=dtype)
    pos = torch.arange(5, device=device)[None].expand(2, -1)
    embeddings = model.model.rotary_emb(x, pos)
    mask = torch.randn(2, 4, 5, 5, device=device, dtype=dtype)
    expected, _ = layer(x, position_embeddings=embeddings, attention_mask=mask)
    actual = qwen3_week1.Qwen3MultiHeadAttention(**attention_kwargs(model))(
        x, mask=mask
    )
    assert_close(actual, expected)
