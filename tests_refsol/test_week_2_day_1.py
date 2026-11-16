"""Week 2 Day 1: append, bounded capacity, cached decode and serving."""

import pytest
import torch
from .tiny_llm_base import Qwen3ModelWeek2, TinyKvFullCache, cached_generate
from .week2_utils import CUDA, tiny_model, assert_close


@pytest.mark.parametrize("capacity", [None, 5])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=CUDA)])
def test_task_1_full_cache_appends_chunks(capacity, device):
    cache = TinyKvFullCache(capacity=capacity)
    a = torch.arange(12, device=device, dtype=torch.float32).reshape(1, 1, 3, 4)
    cache.update_and_fetch(a, a + 10, mask="causal")
    k, v, offset, mask = cache.update_and_fetch(
        a[:, :, :2] + 20, a[:, :, :2] + 30, mask="causal"
    )
    assert offset == 5 and mask == "causal"
    assert_close(k, torch.cat((a, a[:, :, :2] + 20), 2), 0, 0)
    assert_close(v, torch.cat((a + 10, a[:, :, :2] + 30), 2), 0, 0)
    assert k.shape[-2] == 5
    if capacity is not None:
        assert cache.physical_growth_copy_bytes == 0
        assert cache.slice_write_bytes == (k.numel() + v.numel()) * k.element_size()


def test_capacity_rewind_overflow_and_reset():
    cache = TinyKvFullCache(capacity=3)
    a = torch.arange(4).float().reshape(1, 1, 2, 2)
    cache.update_and_fetch(a, a + 10)
    cache.rewind(1)
    k, v, offset, _ = cache.update_and_fetch(a + 20, a + 30)
    assert offset == 3
    assert_close(k, torch.cat((a[:, :, :1], a + 20), 2), 0, 0)
    with pytest.raises(ValueError, match="capacity"):
        cache.update_and_fetch(a, a)
    assert cache.offset == 3
    cache.rewind(1)
    k, _, offset, _ = cache.update_and_fetch(a[:, :, :1] + 40, a[:, :, :1] + 50)
    assert offset == 3 and k[0, 0, -1, 0] == 40
    cache.reset()
    assert cache.offset == 0
    k, _, offset, _ = cache.update_and_fetch(a[:, :, :1], a[:, :, :1])
    assert offset == 1 and k.shape[-2] == 1


@CUDA
@pytest.mark.parametrize("checkpoint", ["kv-cache", "capacity-cache"])
@torch.inference_mode()
def test_tasks_2_and_3_cached_decode_matches_full_and_hf(checkpoint):
    hf = tiny_model()
    model = Qwen3ModelWeek2(hf, checkpoint=checkpoint)
    tokens = torch.tensor([[1, 2, 3]], device="cuda")
    cache = model.create_kv_cache(
        capacity=3 if checkpoint == "capacity-cache" else None
    )
    model(tokens[:, :2], 0, cache)
    decoded = model(tokens[:, 2:], 2, cache)
    full = model(tokens, 0, model.create_kv_cache(capacity=3))
    assert_close(decoded, full[:, -1:], atol=0.03, rtol=0.03)
    assert_close(full, hf(tokens).logits, atol=0.08, rtol=0.05)
    with pytest.raises(ValueError, match="offset"):
        model(tokens[:, :1], 0, cache)
    if checkpoint == "capacity-cache":
        with pytest.raises(ValueError, match="capacity"):
            model(tokens[:, :1], 3, cache)


class Tokenizer:
    eos_token_id = 0

    def encode(self, prompt, **kwargs):
        return [1, 2] if prompt else []

    def decode(self, tokens, **kwargs):
        return "".join("A" if int(t) == 3 else "" for t in tokens)


def test_task_3_generation_sends_only_new_tokens_and_releases(capsys):
    class Cache:
        released = False

        def release(self):
            self.released = True

    class Model:
        device = "cpu"
        use_bounded_kv_capacity = True

        def create_kv_cache(self, capacity=None):
            assert capacity == 5
            self.cache = [Cache()]
            return self.cache

        def __call__(self, tokens, offset, cache, logits_to_keep=None):
            calls.append((tokens.clone(), offset, logits_to_keep))
            logits = torch.full((1, 1, 4), -100.0)
            logits[0, 0, 3 if len(calls) < 3 else 0] = 100
            return logits

    calls = []
    model = Model()
    cached_generate(model, Tokenizer(), "prompt", max_tokens=3)
    assert capsys.readouterr().out.strip() == "AA"
    assert [x[0].tolist() for x in calls] == [[[1, 2]], [[3]], [[3]]]
    assert [x[1] for x in calls] == [0, 2, 3]
    assert all(x[2] == 1 for x in calls)
    assert model.cache[0].released
