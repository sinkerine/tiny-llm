"""Week 3 Day 1 continuous-batching tests."""

from pathlib import Path

import numpy as np
import pytest
import torch

if Path(__file__).parent.name == "tests_refsol":
    import tiny_llm_ref.batch as batch_runtime
    import tiny_llm_ref.models as model_dispatch
else:
    import tiny_llm.batch as batch_runtime
    import tiny_llm.models as model_dispatch

from tiny_llm_cuda.packing import pack_weights
from .tiny_llm_base import (
    BatchingKvCache,
    FastRoPE,
    QuantizedWeights,
    Qwen3ModelWeek2,
    RoPE,
    TinyKvFullCache,
    causal_mask,
    dequantize_weights,
    quantized_linear,
)
from .week3_utils import (
    CUDA,
    assert_close,
    hf_model_exists,
    load_hf_qwen3,
    tiny_qwen3_model,
)

pytestmark = CUDA


def _rope_reference(x: torch.Tensor, offsets: list[int], base: int, traditional: bool):
    B, L, H, D = x.shape
    ref = RoPE(D, max(offsets) + L + 1, base, traditional=traditional)
    rows = []
    for b, off in enumerate(offsets):
        rows.append(ref(x[b : b + 1], slice(off, off + L)))
    return torch.cat(rows, dim=0)


@pytest.mark.parametrize("traditional", [False, True], ids=["default", "traditional"])
def test_task_1_rope_multiple_offsets(traditional: bool):
    torch.manual_seed(0)
    B, L, H, D = 16, 9, 8, 4
    MAX_SEQ_LEN = 14
    BASE = 10000
    user_layer = FastRoPE(D, MAX_SEQ_LEN, BASE, traditional=traditional)
    for _ in range(10):
        x = torch.rand((B, L, H, D), device="cuda", dtype=torch.bfloat16)
        offsets = np.random.randint(0, MAX_SEQ_LEN - L, size=B).tolist()
        ref = _rope_reference(x, offsets, BASE, traditional)
        out = user_layer(x, offsets)
        assert_close(out, ref, atol=2e-2, rtol=2e-2)


def test_task_1_rectangular_causal_mask_keeps_the_prefix_visible():
    mask = causal_mask(L=2, S=5, dtype=torch.float32)
    expected = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, -torch.inf],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(mask, expected)


def test_task_2_batching_kv_cache():
    cache = BatchingKvCache(max_active_requests=3)

    slot0 = TinyKvFullCache()
    slot0.update_and_fetch(
        torch.tensor([[[[10.0]]]], device="cuda", dtype=torch.float32),
        torch.tensor([[[[110.0]]]], device="cuda", dtype=torch.float32),
    )

    slot2 = TinyKvFullCache()
    slot2.update_and_fetch(
        torch.tensor([[[[20.0], [21.0]]]], device="cuda", dtype=torch.float32),
        torch.tensor([[[[120.0], [121.0]]]], device="cuda", dtype=torch.float32),
    )

    cache.add_request(slot0, 0)
    cache.add_request(slot2, 2)

    keys = torch.tensor(
        [
            [[[12.0], [13.0]]],
            [[[0.0], [0.0]]],
            [[[22.0], [23.0]]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    values = torch.tensor(
        [
            [[[112.0], [113.0]]],
            [[[0.0], [0.0]]],
            [[[122.0], [123.0]]],
        ],
        device="cuda",
        dtype=torch.float32,
    )

    batched_keys, batched_values, seq_len, mask = cache.update_and_fetch(
        keys, values, mask_length=2
    )

    expected_keys = torch.tensor(
        [
            [[[0.0], [10.0], [12.0], [13.0]]],
            [[[0.0], [0.0], [0.0], [0.0]]],
            [[[20.0], [21.0], [22.0], [23.0]]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    expected_values = torch.tensor(
        [
            [[[0.0], [110.0], [112.0], [113.0]]],
            [[[0.0], [0.0], [0.0], [0.0]]],
            [[[120.0], [121.0], [122.0], [123.0]]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    expected_mask = torch.tensor(
        [
            [[[-torch.inf, 0.0, 0.0, -torch.inf], [-torch.inf, 0.0, 0.0, 0.0]]],
            [
                [
                    [-torch.inf, -torch.inf, -torch.inf, -torch.inf],
                    [-torch.inf, -torch.inf, -torch.inf, -torch.inf],
                ]
            ],
            [[[0.0, 0.0, 0.0, -torch.inf], [0.0, 0.0, 0.0, 0.0]]],
        ],
        device="cuda",
        dtype=torch.float32,
    ).reshape(3, 1, 2, 4)

    assert seq_len is None
    assert_close(batched_keys, expected_keys, atol=1e-6, rtol=1e-5)
    assert_close(batched_values, expected_values, atol=1e-6, rtol=1e-5)
    assert_close(mask, expected_mask, atol=1e-6, rtol=1e-5)

    cache.remove_request(0)
    replacement = TinyKvFullCache()
    replacement.update_and_fetch(
        torch.tensor([[[[30.0]]]], device="cuda", dtype=torch.float32),
        torch.tensor([[[[130.0]]]], device="cuda", dtype=torch.float32),
    )
    cache.add_request(replacement, 0)

    reused_keys, reused_values, seq_len, reused_mask = cache.update_and_fetch(
        torch.tensor(
            [[[[31.0]]], [[[0.0]]], [[[24.0]]]], device="cuda", dtype=torch.float32
        ),
        torch.tensor(
            [[[[131.0]]], [[[0.0]]], [[[124.0]]]], device="cuda", dtype=torch.float32
        ),
        mask_length=1,
    )
    expected_reused_keys = torch.tensor(
        [
            [[[0.0], [0.0], [0.0], [30.0], [31.0]]],
            [[[0.0], [0.0], [0.0], [0.0], [0.0]]],
            [[[20.0], [21.0], [22.0], [23.0], [24.0]]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    expected_reused_values = torch.tensor(
        [
            [[[0.0], [0.0], [0.0], [130.0], [131.0]]],
            [[[0.0], [0.0], [0.0], [0.0], [0.0]]],
            [[[120.0], [121.0], [122.0], [123.0], [124.0]]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    expected_reused_mask = torch.tensor(
        [
            [[[-torch.inf, -torch.inf, -torch.inf, 0.0, 0.0]]],
            [[[-torch.inf, -torch.inf, -torch.inf, -torch.inf, -torch.inf]]],
            [[[0.0, 0.0, 0.0, 0.0, 0.0]]],
        ],
        device="cuda",
        dtype=torch.float32,
    )
    assert seq_len is None
    assert_close(reused_keys, expected_reused_keys, atol=1e-6, rtol=1e-5)
    assert_close(reused_values, expected_reused_values, atol=1e-6, rtol=1e-5)
    assert_close(reused_mask, expected_reused_mask, atol=1e-6, rtol=1e-5)


@torch.inference_mode()
def helper_test_task_3(model_name: str, seq_len: int, iters: int = 1):
    requests = 4
    max_seq_len = seq_len

    hf_model, tokenizer = load_hf_qwen3(model_name)
    model = model_dispatch.dispatch_week3_batch_model(model_name, hf_model)
    for _ in range(iters):
        cache = [
            BatchingKvCache(requests, max_seq_len)
            for _ in range(model.num_hidden_layers)
        ]
        staggered_start = [seq_len * i // requests for i in range(requests)]
        inputs = (
            torch.arange(requests * seq_len, dtype=torch.long, device="cuda").reshape(
                requests, seq_len
            )
            % tokenizer.vocab_size
        )
        ref_outputs = torch.cat(
            [
                model(inputs[r : r + 1], 0, model.create_kv_cache())
                for r in range(requests)
            ],
            dim=0,
        )
        for offset in range(seq_len + staggered_start[-1]):
            seq_idx = [offset - start for start in staggered_start]
            for request_id, sidx in enumerate(seq_idx):
                if sidx == 0:
                    for c in cache:
                        c.add_request(TinyKvFullCache(), request_id)
                elif sidx == seq_len:
                    for c in cache:
                        c.remove_request(request_id)

            next_tokens = []
            next_offsets = []
            for request_id, sidx in enumerate(seq_idx):
                if 0 <= sidx < seq_len:
                    next_tokens.append(int(inputs[request_id, sidx].item()))
                    next_offsets.append(sidx)
                else:
                    next_tokens.append(0)
                    next_offsets.append(0)

            user_out = model(
                inputs=torch.tensor(
                    next_tokens, dtype=torch.long, device="cuda"
                ).reshape(-1, 1),
                offset=torch.tensor(next_offsets, dtype=torch.long, device="cuda"),
                cache=cache,
            )

            for request_id, sidx in enumerate(seq_idx):
                if 0 <= sidx < seq_len:
                    user_out_r = user_out[request_id, 0, :].float()
                    ref_out_r = ref_outputs[request_id, sidx, :].float()
                    user_out_r = user_out_r - torch.logsumexp(user_out_r, dim=-1, keepdim=True)
                    ref_out_r = ref_out_r - torch.logsumexp(ref_out_r, dim=-1, keepdim=True)
                    torch.testing.assert_close(user_out_r, ref_out_r, rtol=0.15, atol=2.5)


@pytest.mark.skipif(
    not hf_model_exists("Qwen/Qwen3-0.6B"), reason="Qwen/Qwen3-0.6B not cached"
)
def test_task_3_qwen3_0_6b():
    helper_test_task_3("Qwen/Qwen3-0.6B", seq_len=3)


@pytest.mark.skipif(
    not hf_model_exists("Qwen/Qwen3-1.7B"), reason="Qwen/Qwen3-1.7B not cached"
)
def test_task_3_qwen3_1_7b():
    helper_test_task_3("Qwen/Qwen3-1.7B", seq_len=3)


@pytest.mark.skipif(
    not hf_model_exists("Qwen/Qwen3-4B"), reason="Qwen/Qwen3-4B not cached"
)
def test_task_3_qwen3_4b():
    helper_test_task_3("Qwen/Qwen3-4B", seq_len=3)


def _tiny_batch_forward(model):
    batch_size = 3
    active_slots = (0, 2)
    cache = [
        BatchingKvCache(batch_size, max_seq_len=8)
        for _ in range(model.num_hidden_layers)
    ]
    for slot in active_slots:
        request_cache = model.create_kv_cache()
        for layer_cache, batch_cache in zip(request_cache, cache):
            batch_cache.add_request(layer_cache, slot)

    inputs = torch.tensor([[5], [0], [7]], dtype=torch.long, device="cuda")
    offsets = torch.tensor([0, 0, 3], dtype=torch.long, device="cuda")
    return model(inputs=inputs, offset=offsets, cache=cache, logits_to_keep=1)


@torch.inference_mode()
def test_task_3_week3_batch_factory_returns_numerically_valid_model():
    hf_model = tiny_qwen3_model(seed=7)
    factory_model = model_dispatch.dispatch_week3_batch_model("qwen3-0.6b", hf_model)
    equivalent_model = Qwen3ModelWeek2(hf_model)

    factory_logits = _tiny_batch_forward(factory_model)
    expected_logits = _tiny_batch_forward(equivalent_model)
    active = torch.tensor([0, 2], dtype=torch.long, device="cuda")

    assert factory_logits.shape == (3, 1, 128)
    assert factory_logits.dtype == torch.bfloat16
    assert torch.isfinite(factory_logits[active]).all()
    assert_close(factory_logits[active], expected_logits[active], atol=0.1, rtol=0.05)


def test_task_3_torch_projection_selector_matches_public_equation_with_bias():
    torch.manual_seed(11)
    dense_weight = torch.randn((4, 128), device="cuda", dtype=torch.bfloat16)
    packed, scales, biases = pack_weights(dense_weight, group_size=128, bits=4)
    weights = QuantizedWeights(
        scales=scales,
        biases=biases,
        group_size=128,
        bits=4,
        weight=packed,
        use_torch_quantized_linear=True,
    )
    x = torch.randn((2, 128), device="cuda", dtype=torch.bfloat16)
    output_bias = torch.tensor([0.5, -1.0, 1.5, -2.0], device="cuda", dtype=torch.bfloat16)
    expected = (
        torch.nn.functional.linear(
            x, dequantize_weights(packed, scales, biases, 128, 4)
        )
        + output_bias
    )
    actual = quantized_linear(x, weights, bias=output_bias)
    assert_close(actual, expected, atol=2e-2, rtol=2e-2)


class SchedulerFakeDetokenizer:
    def __init__(self, _):
        self.text = ""

    def add_token(self, token):
        self.text += str(token)


class SchedulerFakeTokenizer:
    eos_token_id = 99
    _tokenizer = object()
    detokenizer = SchedulerFakeDetokenizer(_tokenizer)

    def encode(self, prompt, add_special_tokens=False):
        assert not add_special_tokens
        return {
            "long": [10],
            "short": [20],
            "later": [30],
            "immediate-eos": [40],
            "two-token-prompt": [50, 51],
        }[prompt]


class SchedulerFakeModel:
    num_hidden_layers = 1
    device = torch.device("cpu")

    def create_kv_cache(self):
        return [TinyKvFullCache()]

    def __call__(self, inputs, offsets, cache, logits_to_keep=1):
        assert logits_to_keep == 1
        batch_cache = isinstance(cache[0], BatchingKvCache)
        sequence_length = 1 if batch_cache else inputs.shape[1]
        keys = (inputs[:, None, -sequence_length:, None] + 1).to(torch.float32)
        prompt_totals = None
        if batch_cache:
            cache[0].update_and_fetch(keys, keys, mask_length=sequence_length)
        else:
            accumulated_keys, _, _, _ = cache[0].update_and_fetch(keys, keys)
            prompt_totals = accumulated_keys.sum(dim=(1, 2, 3)).tolist()

        next_by_token = {
            0: 99,
            1: 2,
            2: 3,
            3: 4,
            4: 99,
            5: 99,
            6: 99,
            7: 99,
            10: 1,
            20: 5,
            30: 6,
            40: 99,
            50: 51,
            51: 8,
        }
        logits = torch.zeros((inputs.shape[0], 1, 100), dtype=torch.float32)
        for row, token in enumerate(inputs[:, -1].tolist()):
            next_token = next_by_token[token]
            if prompt_totals is not None and token == 51 and prompt_totals[row] == 103:
                next_token = 7
            logits[row, 0, next_token] = 1
        return logits


def make_stage_compatible_request(model, prompt, *, prefill_max_step, prompt_idx):
    try:
        return (
            batch_runtime.Request(
                model,
                SchedulerFakeTokenizer(),
                prompt,
                prefill_max_step=prefill_max_step,
                prompt_idx=prompt_idx,
                max_seq_len=8,
            ),
            True,
        )
    except TypeError:
        return (
            batch_runtime.Request(
                model,
                SchedulerFakeTokenizer(),
                prompt,
                prefill_max_step=prefill_max_step,
                prompt_idx=prompt_idx,
            ),
            False,
        )


def test_task_4_request_prefills_the_complete_prompt_in_one_call():
    request, cumulative_day_2 = make_stage_compatible_request(
        SchedulerFakeModel(),
        "two-token-prompt",
        prefill_max_step=1,
        prompt_idx=7,
    )

    request.try_prefill()
    if cumulative_day_2:
        while not request.is_prefill_done:
            request.try_prefill()

    assert request.is_prefill_done
    assert request.offset == 2
    assert request.next_token == 7
    assert request.prompt_idx == 7


def test_task_4_batch_generate_handles_immediate_eos(monkeypatch):
    monkeypatch.setattr(batch_runtime, "_print_progress", lambda *args: None)
    result = batch_runtime.batch_generate(
        SchedulerFakeModel(),
        SchedulerFakeTokenizer(),
        ["immediate-eos"],
        max_seq_len=2,
        batch_size=1,
        prefill_step=2,
    )
    assert result == [(0, "")]


@pytest.mark.parametrize(
    ("prompt", "max_seq_len", "expected"),
    [
        ("long", 1, [(0, "")]),
        ("long", 2, [(0, "1")]),
    ],
)
def test_task_4_batch_generate_stops_before_crossing_max_seq_len(
    monkeypatch, prompt, max_seq_len, expected
):
    monkeypatch.setattr(batch_runtime, "_print_progress", lambda *args: None)
    result = batch_runtime.batch_generate(
        SchedulerFakeModel(),
        SchedulerFakeTokenizer(),
        [prompt],
        max_seq_len=max_seq_len,
        batch_size=1,
        prefill_step=max_seq_len,
    )
    assert result == expected


def test_task_4_batch_generate_rejects_an_oversized_prompt(monkeypatch):
    monkeypatch.setattr(batch_runtime, "_print_progress", lambda *args: None)

    class FailOnModelWork:
        num_hidden_layers = 1
        device = torch.device("cpu")

        def create_kv_cache(self):
            pytest.fail("cache creation happened before oversized rejection")

        def __call__(self, *args, **kwargs):
            pytest.fail("model forward happened before oversized rejection")

    model = FailOnModelWork()
    with pytest.raises(ValueError):
        batch_runtime.Request(
            model,
            SchedulerFakeTokenizer(),
            "two-token-prompt",
            prefill_max_step=1,
            max_seq_len=1,
        )


def test_task_4_batch_generate_reuses_capacity_and_returns_completion_order(
    monkeypatch,
):
    monkeypatch.setattr(batch_runtime, "_print_progress", lambda *args: None)
    model = SchedulerFakeModel()
    result = batch_runtime.batch_generate(
        model,
        SchedulerFakeTokenizer(),
        ["long", "short", "later"],
        batch_size=2,
    )
    assert result == [(1, "5"), (2, "6"), (0, "1234")]


def test_task_4_single_slot_keeps_the_last_waiting_request(monkeypatch):
    monkeypatch.setattr(batch_runtime, "_print_progress", lambda *args: None)
    result = batch_runtime.batch_generate(
        SchedulerFakeModel(),
        SchedulerFakeTokenizer(),
        ["long", "later"],
        batch_size=1,
    )
    assert result == [(0, "1234"), (1, "6")]
