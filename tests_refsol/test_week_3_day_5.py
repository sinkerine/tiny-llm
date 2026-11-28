"""Week 3 Day 5 paged-FlashAttention public-behavior tests."""

import pytest
import torch

from .tiny_llm_base import (
    Qwen3ModelWeek3,
    paged_attention,
    scaled_dot_product_attention_grouped,
)
from .week3_utils import CUDA, assert_close, tiny_qwen3_model

pytestmark = CUDA


def _assert_paged_case(
    *,
    query_length: int,
    context_length: int,
    page_size: int,
    physical_orders: list[list[int]],
    head_dim: int = 128,
    scale: float | None = None,
    mask: str | None = "causal",
) -> None:
    torch.manual_seed(query_length + context_length + head_dim)
    batch_size = len(physical_orders)
    num_kv_heads = 2
    num_heads = 4

    query = torch.randn(
        (batch_size, num_heads, query_length, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    dense_key = torch.randn(
        (batch_size, num_kv_heads, context_length, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )
    dense_value = torch.randn(
        (batch_size, num_kv_heads, context_length, head_dim),
        device="cuda",
        dtype=torch.bfloat16,
    )

    num_physical_pages = max(page for order in physical_orders for page in order) + 2
    key_pages = torch.full(
        (num_physical_pages, num_kv_heads, page_size, head_dim),
        29.0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    value_pages = torch.full(
        (num_physical_pages, num_kv_heads, page_size, head_dim),
        -23.0,
        device="cuda",
        dtype=torch.bfloat16,
    )
    for batch, physical_order in enumerate(physical_orders):
        for logical_page, physical_page in enumerate(physical_order):
            start = logical_page * page_size
            stop = min(start + page_size, context_length)
            width = stop - start
            key_pages[physical_page, :, :width, :] = dense_key[batch, :, start:stop, :]
            value_pages[physical_page, :, :width, :] = dense_value[
                batch, :, start:stop, :
            ]

    expected = scaled_dot_product_attention_grouped(
        query, dense_key, dense_value, scale=scale, mask=mask
    )
    actual = paged_attention(
        query,
        key_pages,
        value_pages,
        torch.tensor(physical_orders, dtype=torch.int32, device="cuda"),
        torch.tensor([context_length] * batch_size, dtype=torch.int32, device="cuda"),
        page_size,
        scale=scale,
        mask=mask,
    )

    assert actual.shape == query.shape
    assert actual.dtype == torch.bfloat16
    assert_close(actual, expected, atol=2e-2, rtol=3e-2)


@pytest.mark.parametrize(
    ("query_length", "context_length", "page_size", "physical_orders"),
    [
        (9, 9, 16, [[2]]),
        (65, 97, 16, [[5, 1, 6, 0, 4, 2, 3]]),
    ],
    ids=["one-page", "partial-query-and-tail"],
)
def test_long_prefill_matches_dense_attention_across_paged_boundaries(
    query_length: int,
    context_length: int,
    page_size: int,
    physical_orders: list[list[int]],
) -> None:
    _assert_paged_case(
        query_length=query_length,
        context_length=context_length,
        page_size=page_size,
        physical_orders=physical_orders,
    )


def test_batched_gqa_long_prefill_honors_explicit_scale() -> None:
    _assert_paged_case(
        query_length=17,
        context_length=41,
        page_size=16,
        physical_orders=[[4, 0, 5], [2, 6, 1]],
        scale=0.125,
        mask=None,
    )


@pytest.mark.parametrize(
    ("query_length", "context_length", "head_dim", "mask"),
    [(1, 17, 128, "causal"), (9, 17, 32, None)],
    ids=["decode", "generic-bfloat16-prefill"],
)
def test_day5_preserves_decode_and_generic_bfloat16_fallback(
    query_length: int,
    context_length: int,
    head_dim: int,
    mask: str | None,
) -> None:
    _assert_paged_case(
        query_length=query_length,
        context_length=context_length,
        page_size=16,
        physical_orders=[[1, 0]],
        head_dim=head_dim,
        scale=0.25,
        mask=mask,
    )


@torch.inference_mode()
def test_week3_attention_module_runs_long_prefill_through_paged_storage() -> None:
    hf_model = tiny_qwen3_model(seed=7)
    paged_model = Qwen3ModelWeek3(hf_model, page_size=16, enable_paged_attention=True)
    dense_model = Qwen3ModelWeek3(hf_model, page_size=16, enable_paged_attention=False)
    inputs = torch.tensor(
        [[1, 5, 7, 3, 9, 11, 13, 15, 17]], dtype=torch.long, device="cuda"
    )

    paged_output = paged_model(inputs, 0, paged_model.create_kv_cache())
    dense_output = dense_model(inputs, 0, dense_model.create_kv_cache())

    assert paged_output.shape == dense_output.shape == (1, 9, 128)
    assert paged_output.dtype == dense_output.dtype == torch.bfloat16
    assert_close(paged_output, dense_output, atol=5e-2, rtol=5e-2)
