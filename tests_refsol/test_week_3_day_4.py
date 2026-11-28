"""Week 3 Day 4 paged-attention runtime tests."""

import pytest
import torch

from .tiny_llm_base import (
    BatchingKvCache,
    Qwen3ModelWeek2,
    Qwen3ModelWeek3,
    TinyKvPagedCache,
    TinyKvPagedPool,
    paged_attention,
    scaled_dot_product_attention_grouped,
)
from .week3_utils import CUDA, assert_close, tiny_qwen3_model

pytestmark = CUDA


def _random_chunk(length: int, num_heads: int = 2, head_dim: int = 4):
    key = torch.randn((1, num_heads, length, head_dim), device="cuda", dtype=torch.float32)
    value = torch.randn((1, num_heads, length, head_dim), device="cuda", dtype=torch.float32)
    return key, value


@pytest.mark.parametrize(
    ("query_length", "context_length", "page_size", "physical_order"),
    [
        (9, 13, 8, [2, 0]),
        (17, 35, 16, [3, 0, 2]),
    ],
    ids=["nine-token", "multi-page"],
)
def test_task_2_bfloat16_long_prefill_matches_dense_attention(
    query_length, context_length, page_size, physical_order
):
    torch.manual_seed(query_length)
    num_kv_heads = 2
    num_heads = 4
    head_dim = 128

    query = torch.randn(
        (1, num_heads, query_length, head_dim), device="cuda", dtype=torch.bfloat16
    )
    dense_key = torch.randn(
        (1, num_kv_heads, context_length, head_dim), device="cuda", dtype=torch.bfloat16
    )
    dense_value = torch.randn(
        (1, num_kv_heads, context_length, head_dim), device="cuda", dtype=torch.bfloat16
    )
    num_pages = max(physical_order) + 2
    key_pages = torch.full(
        (num_pages, num_kv_heads, page_size, head_dim), 29.0, device="cuda", dtype=torch.bfloat16
    )
    value_pages = torch.full(
        (num_pages, num_kv_heads, page_size, head_dim), -23.0, device="cuda", dtype=torch.bfloat16
    )
    for logical_page, phys_page in enumerate(physical_order):
        start = logical_page * page_size
        stop = min(start + page_size, context_length)
        width = stop - start
        key_pages[phys_page, :, :width, :] = dense_key[0, :, start:stop, :]
        value_pages[phys_page, :, :width, :] = dense_value[0, :, start:stop, :]

    expected = scaled_dot_product_attention_grouped(
        query, dense_key, dense_value, mask="causal"
    )
    actual = paged_attention(
        query,
        key_pages,
        value_pages,
        torch.tensor([physical_order], dtype=torch.int32, device="cuda"),
        torch.tensor([context_length], dtype=torch.int32, device="cuda"),
        page_size,
        mask="causal",
    )
    assert_close(actual, expected, atol=2e-2, rtol=3e-2)


def test_task_1_batch_metadata_preserves_slots_and_logical_lengths():
    pool = TinyKvPagedPool(page_size=4)
    c0 = TinyKvPagedCache(pool)
    c2 = TinyKvPagedCache(pool)
    c0.update_and_fetch(*_random_chunk(5))
    c2.update_and_fetch(*_random_chunk(2))

    batch = BatchingKvCache(max_active_requests=3)
    batch.add_request(c0, 0)
    batch.add_request(c2, 2)

    step_k = torch.randn((3, 2, 1, 4), device="cuda", dtype=torch.float32)
    step_v = torch.randn((3, 2, 1, 4), device="cuda", dtype=torch.float32)
    meta = batch.update_and_fetch_paged(step_k, step_v, mask_length=1, mask="causal")
    assert meta.context_lens.tolist() == [6, 0, 3]
    assert meta.block_table.shape == (3, 2)
    assert meta.block_table[1].tolist() == [-1, -1]
    assert meta.block_table[2, 1].item() == -1
    c0.release()
    c2.release()


def test_task_1_paged_attention_matches_dense_attention():
    torch.manual_seed(4)
    pool = TinyKvPagedPool(page_size=4)
    cache = TinyKvPagedCache(pool)
    k, v = _random_chunk(9, num_heads=2, head_dim=16)
    meta = cache.update_and_fetch_paged(k, v, mask="causal")
    q = torch.randn((1, 4, 3, 16), device="cuda", dtype=torch.float32)
    expected = scaled_dot_product_attention_grouped(q, k, v, mask="causal")
    actual = paged_attention(
        q,
        meta.key_pages,
        meta.value_pages,
        meta.block_table,
        meta.context_lens,
        meta.page_size,
        mask="causal",
    )
    assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    cache.release()


def test_task_2_batched_paged_attention_matches_dense_attention():
    torch.manual_seed(5)
    pool = TinyKvPagedPool(page_size=4)
    c0 = TinyKvPagedCache(pool)
    c2 = TinyKvPagedCache(pool)
    k0, v0 = _random_chunk(6, num_heads=2, head_dim=16)
    k2, v2 = _random_chunk(3, num_heads=2, head_dim=16)
    c0.update_and_fetch(k0, v0)
    c2.update_and_fetch(k2, v2)

    batch = BatchingKvCache(max_active_requests=3)
    batch.add_request(c0, 0)
    batch.add_request(c2, 2)
    step_k = torch.randn((3, 2, 1, 16), device="cuda", dtype=torch.float32)
    step_v = torch.randn((3, 2, 1, 16), device="cuda", dtype=torch.float32)
    meta = batch.update_and_fetch_paged(step_k, step_v, mask_length=1, mask="causal")

    q = torch.randn((3, 4, 1, 16), device="cuda", dtype=torch.float32)
    actual = paged_attention(
        q,
        meta.key_pages,
        meta.value_pages,
        meta.block_table,
        meta.context_lens,
        meta.page_size,
        mask="causal",
    )
    exp0 = scaled_dot_product_attention_grouped(
        q[0:1], torch.cat([k0, step_k[0:1]], dim=2), torch.cat([v0, step_v[0:1]], dim=2), mask="causal"
    )
    exp2 = scaled_dot_product_attention_grouped(
        q[2:3], torch.cat([k2, step_k[2:3]], dim=2), torch.cat([v2, step_v[2:3]], dim=2), mask="causal"
    )
    assert_close(actual[0:1], exp0, atol=1e-4, rtol=1e-4)
    assert (actual[1] == 0).all()
    assert_close(actual[2:3], exp2, atol=1e-4, rtol=1e-4)
    c0.release()
    c2.release()


def test_paged_attention_preserves_bfloat16_for_decode():
    torch.manual_seed(6)
    q = torch.randn((1, 4, 1, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.randn((1, 2, 11, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.randn((1, 2, 11, 128), device="cuda", dtype=torch.bfloat16)
    pool = TinyKvPagedPool(page_size=4)
    cache = TinyKvPagedCache(pool)
    meta = cache.update_and_fetch_paged(k, v, mask="causal")
    actual = paged_attention(
        q, meta.key_pages, meta.value_pages, meta.block_table, meta.context_lens, meta.page_size, mask="causal"
    )
    expected = scaled_dot_product_attention_grouped(q, k, v, mask="causal")
    assert actual.dtype == torch.bfloat16
    assert_close(actual, expected, atol=2e-2, rtol=3e-2)
    cache.release()


@pytest.mark.parametrize(
    ("block_table", "context_lens", "page_size"),
    [
        ([[0, -1]], [-1], 32),
        ([[0, 0]], [64], 32),
        ([[0, 5]], [64], 32),
        ([[0, 1]], [16], 32),
        ([[0]], [64], 32),
    ],
)
def test_paged_attention_rejects_invalid_live_metadata_before_dispatch(
    block_table, context_lens, page_size
):
    q = torch.zeros((1, 4, 1, 16), device="cuda", dtype=torch.float32)
    kp = torch.zeros((3, 2, page_size, 16), device="cuda", dtype=torch.float32)
    vp = torch.zeros((3, 2, page_size, 16), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError):
        paged_attention(
            q,
            kp,
            vp,
            torch.tensor(block_table, dtype=torch.int32, device="cuda"),
            torch.tensor(context_lens, dtype=torch.int32, device="cuda"),
            page_size,
            mask="causal",
        )


@pytest.mark.parametrize(
    ("query_length", "context_len"),
    [(2, 1), (9, 8), (65, 64)],
    ids=["direct", "flash-9", "flash-65"],
)
def test_paged_attention_rejects_active_context_shorter_than_query(
    query_length, context_len
):
    head_dim = 4 if query_length == 2 else 128
    dtype = torch.float32 if query_length == 2 else torch.bfloat16
    q = torch.zeros((1, 4, query_length, head_dim), device="cuda", dtype=dtype)
    kp = torch.zeros((3, 2, 32, head_dim), device="cuda", dtype=dtype)
    vp = torch.zeros((3, 2, 32, head_dim), device="cuda", dtype=dtype)
    live_pages = (context_len + 31) // 32
    block_table = [[*range(live_pages), *([-1] * (3 - live_pages))]]
    with pytest.raises(ValueError):
        paged_attention(
            q,
            kp,
            vp,
            torch.tensor(block_table, dtype=torch.int32, device="cuda"),
            torch.tensor([context_len], dtype=torch.int32, device="cuda"),
            32,
            mask="causal",
        )


@torch.inference_mode()
def test_task_3_incremental_decode_matches_week2_with_paged_attention():
    hf_model = tiny_qwen3_model(seed=0)
    week2_model = Qwen3ModelWeek2(hf_model)
    week3_model = Qwen3ModelWeek3(hf_model, page_size=4, enable_paged_attention=True)
    inputs = torch.tensor([[1, 5, 7, 3, 9, 11]], dtype=torch.long, device="cuda")
    week2_cache = week2_model.create_kv_cache()
    week3_cache = week3_model.create_kv_cache()

    for offset in range(inputs.shape[1]):
        token = inputs[:, offset : offset + 1]
        week2_out = week2_model(token, offset, week2_cache).float()
        week3_out = week3_model(token, offset, week3_cache).float()
        week2_out = week2_out - torch.logsumexp(week2_out, dim=-1, keepdim=True)
        week3_out = week3_out - torch.logsumexp(week3_out, dim=-1, keepdim=True)
        torch.testing.assert_close(week3_out, week2_out, rtol=0.05, atol=0.5)
