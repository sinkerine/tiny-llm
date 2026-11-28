"""Week 3 Day 3 paged-KV storage tests."""

import pytest
import torch

if __package__ == "tests_refsol":
    from tiny_llm_ref import cuda_ops as paged_ops
else:
    from tiny_llm import cuda_ops as paged_ops

from .tiny_llm_base import (
    BatchingKvCache,
    Qwen3ModelWeek2,
    Qwen3ModelWeek3,
    TinyKvFullCache,
    TinyKvPagedCache,
    TinyKvPagedPool,
)
from .week3_utils import CUDA, assert_close, tiny_qwen3_model

pytestmark = CUDA


def _random_chunk(
    length: int,
    num_heads: int = 2,
    head_dim: int = 4,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.randn((1, num_heads, length, head_dim), device="cuda", dtype=dtype)
    value = torch.randn((1, num_heads, length, head_dim), device="cuda", dtype=dtype)
    return key, value


def _logical_contents(cache: TinyKvPagedCache):
    if cache.offset == 0:
        return None
    return cache.gather_dense()


def _assert_logical_contents(cache: TinyKvPagedCache, expected):
    actual = _logical_contents(cache)
    if expected is None:
        assert actual is None
        return
    assert actual is not None
    assert_close(actual[0], expected[0], atol=1e-5, rtol=1e-5)
    assert_close(actual[1], expected[1], atol=1e-5, rtol=1e-5)


def test_task_1_paged_cache_matches_full_cache():
    torch.manual_seed(0)
    pool = TinyKvPagedPool(page_size=4)
    paged = TinyKvPagedCache(pool)
    full = TinyKvFullCache()

    for length in (3, 1, 5, 2):
        key, value = _random_chunk(length)
        pk, pv, poff, _ = paged.update_and_fetch(key, value)
        fk, fv, foff, _ = full.update_and_fetch(key, value)
        assert poff == foff
        assert_close(pk, fk, atol=1e-5, rtol=1e-5)
        assert_close(pv, fv, atol=1e-5, rtol=1e-5)
    paged.release()
    assert pool.num_free_pages == pool.num_pages


def test_task_1_paged_pool_reuses_freed_capacity():
    pool = TinyKvPagedPool(page_size=4)
    req1 = TinyKvPagedCache(pool)
    req1.update_and_fetch(*_random_chunk(7))
    first_pages = list(req1.page_ids)
    assert len(first_pages) == 2
    req1.release()
    assert pool.num_free_pages == 2

    req2 = TinyKvPagedCache(pool)
    req2.update_and_fetch(*_random_chunk(6))
    assert set(req2.page_ids) == set(first_pages)
    assert pool.reused_page_allocations == 2
    assert pool.num_pages == 2
    req2.release()


def test_task_1_rejects_incompatible_dtype_without_mutating_cache():
    pool = TinyKvPagedPool(page_size=4)
    cache = TinyKvPagedCache(pool)
    k0, v0 = _random_chunk(3, dtype=torch.float32)
    cache.update_and_fetch(k0, v0)
    k1, v1 = _random_chunk(2, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        cache.update_and_fetch(k1, v1)
    assert cache.offset == 3
    _assert_logical_contents(cache, (k0, v0))
    cache.release()


def test_task_1_rejects_shape_mismatch_without_mutating_cache():
    pool = TinyKvPagedPool(page_size=4)
    cache = TinyKvPagedCache(pool)
    k0, v0 = _random_chunk(3, num_heads=2, head_dim=4)
    cache.update_and_fetch(k0, v0)
    k1, v1 = _random_chunk(2, num_heads=3, head_dim=4)
    with pytest.raises(ValueError):
        cache.update_and_fetch(k1, v1)
    assert cache.offset == 3
    _assert_logical_contents(cache, (k0, v0))
    cache.release()


def test_task_1_rolls_back_a_multi_page_append(monkeypatch):
    pool = TinyKvPagedPool(page_size=4)
    cache = TinyKvPagedCache(pool)
    k0, v0 = _random_chunk(3)
    cache.update_and_fetch(k0, v0)

    orig_write = pool.write_page_slice
    calls = {"n": 0}

    def failing_write(page_id, start, key, value):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("injected page write failure")
        return orig_write(page_id, start, key, value)

    monkeypatch.setattr(pool, "write_page_slice", failing_write)
    k1, v1 = _random_chunk(6)
    with pytest.raises(RuntimeError):
        cache.update_and_fetch(k1, v1)

    assert cache.offset == 3
    assert len(cache.page_ids) == 1
    assert pool.used_page_ids == {cache.page_ids[0]}
    _assert_logical_contents(cache, (k0, v0))
    cache.release()


def test_task_1_mixed_pools_fail_before_any_batch_row_mutates():
    pool_a = TinyKvPagedPool(page_size=4)
    pool_b = TinyKvPagedPool(page_size=4)
    c0 = TinyKvPagedCache(pool_a)
    c1 = TinyKvPagedCache(pool_b)
    k0, v0 = _random_chunk(2)
    k1, v1 = _random_chunk(2)
    c0.update_and_fetch(k0, v0)
    c1.update_and_fetch(k1, v1)

    batch = BatchingKvCache(max_active_requests=2)
    batch.add_request(c0, 0)
    batch.add_request(c1, 1)
    step_k = torch.randn((2, 2, 1, 4), device="cuda", dtype=torch.float32)
    step_v = torch.randn((2, 2, 1, 4), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError):
        batch.update_and_fetch_paged(step_k, step_v, mask_length=1)

    assert c0.offset == 2
    assert c1.offset == 2
    c0.release()
    c1.release()


def test_task_1_batch_append_is_transactional(monkeypatch):
    pool = TinyKvPagedPool(page_size=4)
    c0 = TinyKvPagedCache(pool)
    c1 = TinyKvPagedCache(pool)
    k0, v0 = _random_chunk(2)
    k1, v1 = _random_chunk(2)
    c0.update_and_fetch(k0, v0)
    c1.update_and_fetch(k1, v1)

    batch = BatchingKvCache(max_active_requests=2)
    batch.add_request(c0, 0)
    batch.add_request(c1, 1)

    def fail_second(*args, **kwargs):
        raise RuntimeError("row 1 failed")

    monkeypatch.setattr(c1, "update_and_fetch_paged", fail_second)
    step_k = torch.randn((2, 2, 1, 4), device="cuda", dtype=torch.float32)
    step_v = torch.randn((2, 2, 1, 4), device="cuda", dtype=torch.float32)
    with pytest.raises(RuntimeError):
        batch.update_and_fetch_paged(step_k, step_v, mask_length=1)

    assert c0.offset == 2
    assert c1.offset == 2
    _assert_logical_contents(c0, (k0, v0))
    c0.release()
    c1.release()


def test_task_1_paged_pool_grows_geometrically_and_preserves_data():
    torch.manual_seed(1)
    pool = TinyKvPagedPool(page_size=2)
    cache = TinyKvPagedCache(pool)
    chunks_k, chunks_v = [], []
    for _ in range(9):
        k, v = _random_chunk(2)
        chunks_k.append(k)
        chunks_v.append(v)
        cache.update_and_fetch(k, v)
    assert pool.num_pages == 9
    assert pool.capacity >= 9
    assert pool.storage_growths <= 4
    expected = (torch.cat(chunks_k, dim=2), torch.cat(chunks_v, dim=2))
    _assert_logical_contents(cache, expected)
    cache.release()


def test_task_1_paged_pool_reset_removes_warmup_capacity():
    pool = TinyKvPagedPool(page_size=4)
    cache = TinyKvPagedCache(pool)
    cache.update_and_fetch(*_random_chunk(9))
    with pytest.raises(ValueError):
        pool.reset()
    cache.release()
    pool.reset()
    assert pool.capacity == 0
    assert pool.num_pages == 0
    assert pool.storage_nbytes == 0


def test_task_1_block_table_tracks_logical_pages():
    pool = TinyKvPagedPool(page_size=4)
    cache = TinyKvPagedCache(pool)
    cache.update_and_fetch(*_random_chunk(9))
    bt = cache.block_table(max_pages=5)
    assert bt.shape == (1, 5)
    assert bt.dtype == torch.int32
    assert bt[0, :3].tolist() == cache.page_ids
    assert bt[0, 3:].tolist() == [-1, -1]
    cache.release()


def test_task_1_materialize_keeps_logical_contents_available():
    pool = TinyKvPagedPool(page_size=4)
    cache = TinyKvPagedCache(pool)
    k, v = _random_chunk(5)
    cache.update_and_fetch(k, v)
    cache.materialize()
    _assert_logical_contents(cache, (k, v))
    cache.release()


def test_task_1_paged_cache_rewind_matches_full_cache():
    torch.manual_seed(2)
    pool = TinyKvPagedPool(page_size=4)
    paged = TinyKvPagedCache(pool)
    full = TinyKvFullCache()
    k, v = _random_chunk(10)
    paged.update_and_fetch(k, v)
    full.update_and_fetch(k, v)
    paged.rewind(5)
    full.rewind(5)
    assert paged.offset == full.offset == 5
    assert paged.num_pages == 2
    assert pool.num_free_pages == 1
    _assert_logical_contents(paged, full.key_values)
    paged.release()


def test_task_1_paged_cache_rewind_then_append_matches_full_cache():
    torch.manual_seed(3)
    pool = TinyKvPagedPool(page_size=4)
    paged = TinyKvPagedCache(pool)
    full = TinyKvFullCache()
    k0, v0 = _random_chunk(9)
    paged.update_and_fetch(k0, v0)
    full.update_and_fetch(k0, v0)
    paged.rewind(3)
    full.rewind(3)
    k1, v1 = _random_chunk(5)
    paged.update_and_fetch(k1, v1)
    full.update_and_fetch(k1, v1)
    assert paged.offset == full.offset == 11
    _assert_logical_contents(paged, full.key_values)
    paged.release()


@pytest.mark.parametrize("cache_kind", ["full", "paged"])
def test_task_1_rewind_zero_and_full_length_are_symmetric(cache_kind):
    pool = TinyKvPagedPool(page_size=4)
    cache = TinyKvFullCache() if cache_kind == "full" else TinyKvPagedCache(pool)
    k, v = _random_chunk(6)
    cache.update_and_fetch(k, v)
    cache.rewind(0)
    assert cache.offset == 6
    cache.rewind(6)
    assert cache.offset == 0
    assert cache.key_values is None


@pytest.mark.parametrize("cache_kind", ["full", "paged"])
@pytest.mark.parametrize("rewind_length", [-1, 7, True])
def test_task_1_invalid_rewind_fails_without_mutating_cache(cache_kind, rewind_length):
    pool = TinyKvPagedPool(page_size=4)
    cache = TinyKvFullCache() if cache_kind == "full" else TinyKvPagedCache(pool)
    k, v = _random_chunk(6)
    cache.update_and_fetch(k, v)
    with pytest.raises(ValueError):
        cache.rewind(rewind_length)
    assert cache.offset == 6
    cache.release()


def test_task_1_request_caches_keep_independent_metadata():
    pool = TinyKvPagedPool(page_size=4)
    c0 = TinyKvPagedCache(pool)
    c1 = TinyKvPagedCache(pool)
    k0, v0 = _random_chunk(5)
    k1, v1 = _random_chunk(3)
    c0.update_and_fetch(k0, v0)
    c1.update_and_fetch(k1, v1)
    assert set(c0.page_ids).isdisjoint(c1.page_ids)
    assert c0.offset == 5
    assert c1.offset == 3
    _assert_logical_contents(c0, (k0, v0))
    _assert_logical_contents(c1, (k1, v1))
    c0.release()
    c1.release()


def test_task_2_batch_removal_releases_capacity_for_next_request():
    pool = TinyKvPagedPool(page_size=4)
    batch = BatchingKvCache(max_active_requests=2)
    c0 = TinyKvPagedCache(pool)
    c0.update_and_fetch(*_random_chunk(6))
    batch.add_request(c0, 0)
    batch.remove_request(0)
    assert pool.num_free_pages == 2
    c1 = TinyKvPagedCache(pool)
    c1.update_and_fetch(*_random_chunk(6))
    assert pool.num_pages == 2
    assert pool.reused_page_allocations == 2
    c1.release()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_task_2_paged_cache_update_writes_only_the_requested_slice(dtype):
    pages = torch.full((3, 2, 4, 8), -7.0, device="cuda", dtype=dtype)
    chunk = torch.arange(2 * 2 * 8, device="cuda", dtype=torch.float32).reshape(
        1, 2, 2, 8
    ).to(dtype)
    updated = paged_ops.paged_cache_update(pages, chunk, 1, 1)
    assert_close(updated[1, :, 1:3, :], chunk[0], atol=1e-5, rtol=1e-5)
    assert (updated[0] == -7.0).all()
    assert (updated[2] == -7.0).all()
    assert (updated[1, :, 0, :] == -7.0).all()
    assert (updated[1, :, 3, :] == -7.0).all()


@pytest.mark.parametrize(
    ("page_id", "start"),
    [(-1, 0), (3, 0), (0, -1), (0, 3)],
)
def test_task_2_paged_cache_update_rejects_invalid_inputs(page_id, start):
    pages = torch.zeros((3, 2, 4, 8), device="cuda", dtype=torch.float32)
    chunk = torch.ones((1, 2, 2, 8), device="cuda", dtype=torch.float32)
    with pytest.raises(ValueError):
        paged_ops.paged_cache_update(pages, chunk, page_id, start)


@pytest.mark.parametrize("tie_word_embeddings", [True, False])
@torch.inference_mode()
def test_task_3_week3_full_prompt_matches_week2(tie_word_embeddings: bool):
    inputs = torch.tensor([[1, 5, 7, 3, 9, 11]], dtype=torch.long, device="cuda")
    for seed in (0, 11):
        hf_model = tiny_qwen3_model(tie_word_embeddings=tie_word_embeddings, seed=seed)
        week2_model = Qwen3ModelWeek2(hf_model)
        week3_model = Qwen3ModelWeek3(
            hf_model, page_size=4, enable_paged_attention=False
        )
        week2_out = week2_model(inputs, 0, week2_model.create_kv_cache()).float()
        week3_out = week3_model(inputs, 0, week3_model.create_kv_cache()).float()
        week2_out = week2_out - torch.logsumexp(week2_out, dim=-1, keepdim=True)
        week3_out = week3_out - torch.logsumexp(week3_out, dim=-1, keepdim=True)
        torch.testing.assert_close(week3_out, week2_out, rtol=0.05, atol=0.5)


@torch.inference_mode()
def test_task_3_incremental_decode_attention_cache_matches_week2():
    hf_model = tiny_qwen3_model(seed=0)
    week2_model = Qwen3ModelWeek2(hf_model)
    week3_model = Qwen3ModelWeek3(hf_model, page_size=4, enable_paged_attention=False)
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
