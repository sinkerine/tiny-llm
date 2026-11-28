# 🚧 Week 3 Day 1: Continuous Batching

You begin with the completed
[Week 2 Day 5 single-request model](./week2-05-tiled-prefill-attention.md):
multi-offset RoPE and causal masking already have stable interfaces, and each
request can own a dense KV cache. The Day 1 starter leaves four learner-owned
slices behind those interfaces:

- dense batch assembly and masking in `BatchingKvCache`;
- `torch_quantized_linear` plus the per-weight `use_torch_quantized_linear`
  selector and the explicit `dispatch_week3_batch_model` factory;
- selector propagation through `Qwen3ModelWeek2`; and
- `Request.try_prefill` plus the request-admission/decode loop in
  `batch_generate`.

Complete and test those slices in order. The resulting **continuous batch**
keeps several active requests on the device and replaces each request as soon
as it finishes. When enabled, only quantized projections cross the optional
PyTorch projection seam; normalization, activation, RoPE, cache state,
attention, and scheduling remain course-owned.

So far, each generation loop has processed only one request. That may not
provide enough work to use the device efficiently, so Day 1 decodes several
requests in each model call.

A static batch could select five prompts and run them together until every
request finishes. However, generated sequences have different lengths. If four
requests finish quickly while the fifth continues, most of the batch remains
idle and queued requests cannot start.

Continuous batching instead sets a maximum number of active decode requests.
When one finishes, the scheduler assigns its batch slot and KV-cache entry to a
waiting request. This keeps the decode batch populated whenever work is queued.

The scheduler must also interleave prefill and decode work. We will use a simple
policy: advance one pending prefill, then decode one token for every active
request.

```python
while requests_in_queue_or_in_progress:
    if prefill_request is not None:
        prefill_request.try_prefill()  # Day 1 processes the complete prompt
        if prefill_request.ready:
            if kv_cache.try_add(prefill_request):
                prefill_request = next(requests)
    if active_requests:
        tokens = decode(model, kv_cache)
        for request, token in zip(active_requests, tokens):
            request.append(token)
```

A complete prompt is admitted in one call on Day 1. This makes the scheduling
policy easy to inspect and exposes an important limitation: one long prefill
can delay every active request's next decode step. Day 2 will add a bounded
prefill budget to solve that fairness problem.

## Task 1: Reuse RoPE and Causal Masking for Batched Requests

```
src/tiny_llm/week2_kernels.py::FastRoPE  (reuse unchanged)
src/tiny_llm/attention.py::causal_mask   (reuse unchanged)
```

Continuous batching requires one RoPE offset per batch element and a causal
mask whose query and source lengths may differ. Verify those two Week 2
interfaces before adding the scheduler so the serving layer can use one model
contract for every request position.

Verify multi-offset RoPE and the rectangular causal mask with:

```bash
pdm run test --week 3 --day 1 -- -k task_1
```

## Task 2: Batch KV Cache

```
src/tiny_llm/kv_cache.py::BatchingKvCache
```

`BatchingKvCache` holds one request cache per decode slot. Because requests may
have different sequence lengths, it must combine their keys and values into
dense tensors and construct a matching `B x 1 x L x S` mask.

```
S = max(S_i across active requests)
L = mask_length (input parameter)
request_keys: H, S_i, D
request_values: H, S_i, D
batched_keys: B, H, S, D
batched_values: B, H, S, D
mask: B, 1, L, S
```

Right-align each active request in the common `S` dimension. The leading
positions remain zero and masked out. Inactive slots remain fully masked.

```python
keys_i, values_i = request_cache[i]
batched_keys[i, :, (S - S_i):S, :] = keys_i
batched_values[i, :, (S - S_i):S, :] = values_i
mask[i, :, 0:L, (S - S_i):S] = causal_mask(L, S_i)
```

You can verify your solution by running:

```bash
pdm run test --week 3 --day 1 -- -k task_2
```

## Task 3: Add the Week 3 Projection Seam

```
src/tiny_llm/quantize.py::torch_quantized_linear
src/tiny_llm/qwen3_week2.py::Qwen3ModelWeek2.__init__
src/tiny_llm/models.py::dispatch_week3_batch_model
```

Week 2 ends with course-owned packed W4 CUDA projections and tiled dense
attention. The [Day 3 projection](./week2-03-simd-matrix-prefill.md) exposes
its loader and tensor-core WMMA operations. Week 3 also exposes a readable
PyTorch projection control so you can verify or isolate serving mechanisms
independently of the custom W4 kernels.

Add a per-weight `use_torch_quantized_linear` selector on `QuantizedWeights`
whose default remains `False`, preserving every Week 2 checkpoint. When
selected, `quantized_linear` should call `torch_quantized_linear(x, w, bias)`,
which materializes the dense weight matrix with `dequantize_weights` and calls
`torch.nn.functional.linear` (adding the optional bias). Propagate
`use_torch_quantized_linear` through `Qwen3ModelWeek2.__init__`, and implement
`dispatch_week3_batch_model` as the explicit construction seam that builds the
dense-cache Week 2 model (`dispatch_model(model_name, hf_model, week=2)`).

Call this batch-ready model with several requests, one offset per batch
element, and the mask returned by `BatchingKvCache`. Exercise requests joining
and leaving at different positions. When `use_torch_quantized_linear=True`,
only quantized projections cross the PyTorch projection seam; normalization,
RoPE, activation, attention, cache state, and scheduling remain in your
solution. The model remains request-agnostic; slot ownership and lifecycle
belong to the cache and scheduler.

Verify the projection seam and its batch-model factory with:

```bash
pdm run test --week 3 --day 1 -- -k task_3
```

## Task 4: Batch Generate

```
src/tiny_llm/batch.py
```

First implement `Request.try_prefill` by prefilling the complete prompt in one
call. The visible `prefill_max_step` input is reserved for Day 2; on Day 1, give
this call a budget that covers the complete remaining prompt. Then complete the
scheduler in `batch_generate`. Each request passes through three states:

- **Pending.** `pending_prefill_request` holds at most one request. When its
  prefill finishes, it may already be complete: its first generated token is
  EOS, or it has reached `max_seq_len`. Record its result and clear
  `pending_prefill_request` without giving it a decode slot. Otherwise find an idle slot `i`, add the request's cache for each
  layer to slot `i` of that layer's `BatchingKvCache` with `add_request`, place
  the request in `decode_requests[i]`, and clear `pending_prefill_request`. If
  every slot is busy, leave it pending. The supplied loop keeps running while a
  request is pending, so it is admitted as soon as a slot frees.
- **Decoding.** Every iteration, collect the next token and offset of each
  active slot, using a placeholder token and offset 0 for idle slots; the batch
  mask already hides them.
- **Complete.** When a slot's request reaches EOS or `max_seq_len`, call
  `remove_request(i)` on every layer's cache, record its result, and set
  `decode_requests[i]` back to `None` so the next pending request can use it.

A prompt longer than `max_seq_len` must fail before model or cache work, and a
generated token that would cross the limit is not emitted.

Results are returned in **completion order**, because short requests can leave
the batch before earlier long requests. Each result's `prompt_idx` maps it back
to the original input position; do not reorder completed results into input
order.

Use the supplied scheduler checkpoint for full prefill, admission and slot
reuse, immediate EOS, maximum-length termination, completion-order results,
and their original `prompt_idx` values:

```bash
pdm run test --week 3 --day 1 -- -k task_4
```

Then run the scheduler against a real Hugging Face checkpoint on CUDA (once
`Qwen3ModelWeek3` is wired on Day 3, `--loader week3` runs `batch_generate`
directly; the older `batch-main` and `bench-chunked-prefill` scripts remain
historical MLX/Metal tools):

```bash
pdm run main --loader week3 --solution tiny_llm \
  --model Qwen/Qwen3-0.6B --max-seq-len 64 --batch-size 1
```

The table later in Day 2 uses the historical task #367 MLX/Metal trace from
source `18aec850`, not a result reproduced on this PyTorch/CUDA checkout.

{{#include copyright.md}}
