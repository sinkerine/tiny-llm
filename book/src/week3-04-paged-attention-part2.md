# 🚧 Week 3 Day 4: Direct Paged Attention

> 🚧 This chapter is under review and may change.

In this chapter, you will make **direct paged attention** handle decode and
prefill in both float32 and BF16. The scheduler passes request-local block
tables and context lengths to the operator, which reads K/V from the shared
layer pool without gathering a dense batch first. One correct page-walking
kernel may serve every query shape; separate decode and prefill kernels are an
optimization choice. Day 5 replaces the supported BF16 long-prefill hot case
with a tiled implementation.

> **Prerequisite:** Complete Week 3 Day 3's paged storage and
> [Week 2 Day 5's online-softmax attention](./week2-05-tiled-prefill-attention.md).
> The new concept here is translating logical K/V
> positions through a block table. Tiled FlashAttention comes only after this
> direct path works.

## Paged KV Cache vs Paged Attention

These two ideas are related, but they are not the same:

1. **Paged KV cache**
   KV is stored in fixed-size pages.
2. **Paged attention**
   The attention path reads KV directly from those pages via metadata such as a page table.

You can implement the first one without the second one, but the real serving payoff comes when both are present.

## The Metadata a Paged Runtime Needs

Once KV is paged, dense `B x H x S x D` tensors are no longer the natural runtime representation. Instead, the runtime should prepare metadata like:

```plain
block_table:  [B, max_pages_per_request]
context_lens: [B]
```

For the current layer being executed:

- `block_table[b, i]` gives the page id for request `b`'s current-layer logical page `i`
- `context_lens[b]` gives the valid token count for request `b`

This is the bridge between the scheduler and the attention kernel.

A production runtime often also carries write-side metadata such as `slot_mapping`.
For this chapter, we keep the write side inside the cache and focus on the read-side
metadata needed by attention.

## Why `block_table` Matters

Suppose one layer cache for request A has:

```plain
page_ids = [12, 5, 3]
context_len = 10
page_size = 4
```

Then the logical sequence positions map to physical storage like this:

```plain
logical 0..3  -> page 12
logical 4..7  -> page 5
logical 8..9  -> page 3
```

The attention runtime does not need a fully gathered dense tensor if it already knows:

- which current-layer page each logical block lives in,
- how long the context is,
- and where the current query positions are.

That is exactly what `block_table` and `context_lens` encode.

## The Paged Attention API

At this point, the runtime should grow a new attention entry point:

```python
paged_attention(
    query,
    key_pages,
    value_pages,
    block_table,
    context_lens,
    page_size,
    scale=None,
    mask="causal",
)
```

With shapes like:

```plain
query:          B, H_q, L, D
key_pages[i]:   1, H_kv, page_size, D
value_pages[i]: 1, H_kv, page_size, D
block_table:    B, max_pages
context_lens:   B
```

The source length is no longer represented by one contiguous tensor dimension.
The operator reconstructs it logically from the page table.

In this chapter, `paged_attention` should read pages directly from a GPU
kernel. The runtime contract is now: model code and batching code pass pages
plus metadata, and the attention kernel walks that metadata without first
rebuilding dense K/V.

## Prefill Metadata

During prefill, a chunk may span multiple pages. The runtime needs to know:

- which current-layer pages already existed,
- which new pages were allocated,
- how many valid tokens are in the tail page,
- how to map incoming K/V rows into page storage.

In this teaching implementation, the cache still owns the write-side bookkeeping.
The attention path only needs the block table after the write is done.

## Decode Metadata

During decode, each active request typically writes one token.

The runtime should be able to:

1. append the token's K/V to the current tail page,
2. allocate a new page only if the tail page is full,
3. update the current layer cache's `context_len`,
4. run attention over the full logical context using `block_table`

This is the point where decode stops paying the repeated dense-repack cost from Day 1.

## Establish the Direct Page-Walking Boundary

First make every supported query shape correct through the same direct paged
boundary. You may reuse one kernel for decode and prefill. If you split the
workloads, a single tile shape is unlikely to keep the GPU busy for both a
one-token query and a long prompt. Use these design rules:

1. Preserve the Week 2 BF16 model boundary and reuse its internal accumulation
   policy unchanged.
2. For short queries, expose parallelism across the cached context or head
   dimension. Do not reserve most of a thread block for query rows that do not
   exist.
3. For prefill, begin with a direct page-walking schedule whose address
   calculation is easy to validate.
4. Use the readable PyTorch equation (`scaled_dot_product_attention_grouped`)
   and the dense Week 2 attention kernel in your solution as correctness
   oracles for the new page-walking schedule.

The CUDA reference uses a single warp-based direct page-walking kernel
(`paged_attention_direct_kernel`, one 32-thread warp per `(batch, head, query_row)`)
for all generic and decode shapes on Day 4, and adds a tiled BF16 `D == 128`
long-prefill kernel (`paged_attention_tiled_bf16_d128_kernel`) on Day 5:

| Shape | Schedule in CUDA reference | Work decomposition |
|---|---|---|
| Day 4 (all shapes) and Day 5 fallback (`L <= 8`, non-BF16, or `D != 128`) | Direct warp paged attention | One 32-thread warp per `(batch, head, query_row)` walks logical K/V positions through `block_table` with online softmax in FP32. |
| Day 5 hot case (`L > 8`, BF16, `D == 128`) | Tiled paged FlashAttention | Shared-memory BQ4/BK16 tile staging across logical page boundaries with 4 warps (128 threads) per block. |

If you make a shape decision, put it at the extension boundary rather than
converting inputs or falling back to dense attention in Python. Keep the
model-facing `paged_attention` API unchanged. Public Day 4 tests grade only its
shape, dtype, validation, page addressing, masking, and numerical behavior;
they do not require this split or any kernel name.

## How This Maps to `tiny-llm`

### `src/tiny_llm/attention.py`

Add a new function:

```python
def paged_attention(...):
    ...
```

In your solution, make it a correctness-first page-walking CUDA operation
with online softmax. It may use one kernel or several internal schedules:

1. use `block_table[b]` to find the physical pages for request `b`,
2. use `context_lens[b]` to ignore unused tail capacity,
3. visit K/V in small tiles instead of materializing dense K/V,
4. merge each tile into the output with online softmax.

The important change from dense attention is the K/V address calculation. Dense attention can
advance through dense K/V by pointer arithmetic. Week 3 must translate each
logical key position through `block_table` first:

```plain
logical key position -> logical page -> physical page id -> slot in page
```

After that lookup, the online-softmax update is the same recurrence as Week 2
Day 5. Keep the page-walking schedule simple enough that block-table and
tail-page boundary errors are visible. Day 5 will tile its inner matrix work
while preserving this address calculation.

For performance, one-token decode benefits from a different work decomposition.
A multi-row prefill tile would leave almost every query row idle, so short
queries work well with a warp-per-query schedule that keeps the query in
registers and reduces dot products across lanes with warp shuffles. Splitting
decode and long prefill is an optimization, not a public correctness
requirement.

The page pool should therefore expose contiguous physical storage:

```plain
key_pages:   P, H_kv, page_size, D
value_pages: P, H_kv, page_size, D
```

A Python list of page tensors is convenient for teaching the allocator, but a
GPU kernel needs a single buffer so `page_id` can be turned into an address.

### `src/tiny_llm/qwen3_week3.py`

The attention module should call the paged runtime directly:

```python
metadata = cache.update_and_fetch_paged(...)
x = paged_attention(...)
```

Week 3 cache handles are expected to provide paged metadata. If a dense cache is
passed to the Week 3 model, that is a programming error rather than a signal to
silently fall back to dense attention.

### `src/tiny_llm/batch.py`

The scheduler does not build the metadata itself. A prefill call passes the
request's own layer caches, and each `TinyKvPagedCache.update_and_fetch_paged`
(from Day 3) returns that one request's metadata. A batched decode call passes
the per-layer `BatchingKvCache`, whose `update_and_fetch_paged` returns the
padded `block_table` and `context_lens` for every slot. The scheduler owns the
request lifecycle around both: which slot each request occupies, and releasing
every layer cache's pages when a request ends.

This is where continuous batching and paged attention finally connect. On Day 1, batching worked by repacking tensors. Here, batching should work by reusing page tables and updating only the new slots.

## Implementation Order

Use this implementation order:

1. paged storage
2. `block_table` / `context_lens` plumbing
3. correctness-first page-walking GPU attention
4. model and batch dispatch

Each step has a direct correctness check before the next abstraction is added.

## What Must Hold, and What Breaks If It Doesn't

These are the invariants worth checking in tests:

1. **`context_len` equals the number of written logical token positions.** If it
   is too small, attention skips written K/V; if it is too large, attention
   reads unwritten tail slots. Either case makes paged output diverge from the
   dense baseline.
2. **`block_table` reconstructs the same logical K/V order as the dense
   baseline.** A wrong mapping can pair a query with the wrong token's K/V and
   change the output even when every page contains valid data. Reordering
   complete pages can change a causal-prefix result because it changes which
   K/V pairs each query can see. By contrast, one-token decode over all
   positions in complete pages is permutation-invariant to their order when
   each K/V pair moves together.
3. **The allocator gives each page to only one live cache handle unless sharing
   is explicit.** If two live handles alias a page, a write for one request
   overwrites K/V that the other request can still attend to.
4. **Releasing a request returns every page owned by every layer cache exactly
   once.** Missing a page leaks pool capacity; returning one twice raises the
   pool's already-free error instead of completing cleanup.
5. **Decode allocates a new page only when the tail page overflows.** Allocating
   earlier strands writable tail slots and inflates the used-page count. This
   course pool grows instead of reporting exhaustion, so that waste can force
   backing storage to grow and copy earlier, increasing memory pressure.

<a id="task-1-add-batch-metadata"></a>

## Task 1: Check the Batch Metadata

```
src/tiny_llm/paged_kv_cache.py
src/tiny_llm/kv_cache.py
```

Day 4's attention reads the `block_table` and `context_lens` you already built
on [Day 3](./week3-03-paged-attention-part1.md#batching-requests-with-pages):
`TinyKvPagedCache.block_table`, `context_lens`, `paged_metadata` and
`update_and_fetch_paged`, plus `BatchingKvCache.update_and_fetch_paged`. Task 1
adds no new methods and no scheduler change. The metadata reaches attention
through the model's cache call, which Task 3 wires up. Before writing the
kernel, make sure those arrays hold exactly what it will read:

- one row per decode slot, including idle ones;
- an active row lists that request's page ids in logical order, padded with
  `-1` up to the longest active request's page count;
- an idle slot has `context_lens` 0 and a row of all `-1`;
- each `context_lens` entry counts the request's written tokens, including the
  chunk just appended.

Your Day 3 metadata check still applies. It builds two request caches, appends
through `BatchingKvCache.update_and_fetch_paged`, and checks their
`context_lens` and page-to-token mapping without calling `paged_attention`:

```bash
pdm run test --week 3 --day 3 -- -k task_3_model_request_caches_keep_independent_metadata
```

That is a prerequisite check on your Day 3 code. Both of its rows are active,
so it does not cover the boundary Day 4's kernel depends on. This Day 4 check
does: two active slots around an idle one, rows of different lengths, and the
written length after a one-token append. It also checks the page tensors, page
size and mask that `update_and_fetch_paged` returns. It does not call
`paged_attention`, so run it before you write the kernel:

```bash
pdm run test --week 3 --day 4 -- -k task_1_batch_metadata
```

Day 4's other Task 1 test compares `paged_attention` with dense attention, so
it turns green with Task 2:

```bash
pdm run test --week 3 --day 4 -- -k task_1
```

## Task 2: Define `paged_attention`

```
src/tiny_llm/attention.py
src/cuda/student.cu
src/cuda/bindings.cpp     (already wired)
src/tiny_llm/cuda_ops.py  (already wired)
```

Modify the stable starter boundary:

- `paged_attention` in `src/tiny_llm/attention.py`, which validates the paged
  metadata and delegates to `cuda_ops.paged_attention`;
- `paged_attention_cuda` and its CUDA kernel(s) in `src/cuda/student.cu`
  (already bound in `src/cuda/bindings.cpp` and wrapped in
  `src/tiny_llm/cuda_ops.py`). The starter stub remains fail-closed until you
  replace it.

Add a paged attention interface whose inputs come from the paged runtime rather
than a dense reconstructed `S` dimension. Preserve the Week 2 precision
contract without adding a new model dtype or conversion at the serving layer.

Walk every request's block table while keeping online-softmax state:

```python
running_max = max(previous_max, page_max)
running_sum = previous_sum * exp(previous_max - running_max) + page_sum
output = previous_output * exp(previous_max - running_max) + page_output
```

After all visible pages are consumed, divide `output` by `running_sum`.
This is the key idea that lets the kernel avoid materializing dense K/V while
still producing the same result as dense attention.

In the CUDA reference, one correctness-first kernel
(`paged_attention_direct_kernel<T>`) serves both float32 and BF16 for all query
lengths: launch one 32-thread warp per `(batch, head, query_row)`, accumulate
dot products across the 32 lanes with warp shuffles (`__shfl_xor_sync`), keep
the online-softmax `(m, l)` state and output accumulator in FP32, and translate
each logical key index `s` through `block_table[b, s / page_size]` and
`s % page_size`. Guard the device with `c10::cuda::CUDAGuard`, launch on
`at::cuda::getCurrentCUDAStream()`, and check with
`C10_CUDA_KERNEL_LAUNCH_CHECK()`. Favor inspectable page ownership over the
final tiled performance schedule; Day 5 is where the supported BF16
long-prefill region receives a dedicated tiled implementation.

Compare small deterministic fixtures with the readable PyTorch equation
(`scaled_dot_product_attention_grouped`) and the dense Week 2 attention path
before tuning the page-walking schedule.

Rebuild the extension, then use the BF16 long-prefill checkpoint as the first
feedback loop for this task:

```bash
pdm run build-ext
pdm run test --week 3 --day 4 -- -k task_2_bfloat16_long_prefill
```

The focused checkpoint constructs valid page metadata directly. It checks a
nine-token prefill and a longer multi-page prefill with noncontiguous physical
pages, poisoned unused tail slots, and causal prefix masking. It does not
require model dispatch, continuous batching, or the Day 5 kernel. Kernel names
and implementation structure are not part of the test contract; only the public
numerical, metadata, and dtype behavior is.

`paged_attention` must also reject bad metadata with `ValueError` before it
dispatches any kernel, because a kernel given a wrong page id reads another
request's K/V or memory outside the pool. Check that:

- `page_size` is positive;
- every `context_lens` entry is nonnegative and fits in its row:
  `ceil(context_len / page_size)` must not exceed the table width;
- each of those live entries is a page id inside the page storage, and no live
  page id appears twice anywhere in the table;
- every entry after the live ones is exactly `-1`;
- an active row (`context_len > 0`) holds at least as many tokens as the query
  has rows, `L`.

With the batch metadata from Task 1, run the remaining Task 2 checks and the
metadata validation:

```bash
pdm run test --week 3 --day 4 -- -k task_2
pdm run test --week 3 --day 4 -- -k rejects
```

Rerun `-k task_1` from Task 1 now as well; it needs this operator.

Day 4 must work without a future tiled/MMA kernel. A dedicated prefill kernel
is optional here: reusing the direct warp kernel for long queries is valid when
it preserves the public behavior. Day 5 may replace only the internal supported
BF16 long-query region while preserving the same API, generic fallback, and
page-table semantics.

### Your solution's boundary

The walkthrough keeps page translation and online softmax in a course-owned
CUDA operation so you can inspect them. The public checkpoint does not grade
an internal helper, tile, symbol, or library choice: a behaviorally equivalent
implementation or external low-level library is valid. To preserve the
chapter's systems outcome, the completed operator still consumes page storage
and its block table directly rather than rebuilding dense K/V in Python.
PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`) remains a
useful correctness oracle and performance baseline.

Both prefill and decode read page storage through this interface. Do not add a
dense-only special case: Day 5 optimizes this same paged contract.

## Task 3: Dispatch from the Model

```
src/tiny_llm/qwen3_week3.py
```

Modify `Qwen3MultiHeadAttention.__call__`, `Qwen3ModelWeek3.__init__`, and
`Qwen3ModelWeek3.__call__` to select the paged path when
`enable_paged_attention=True` (the default).

Update the model so it routes to paged attention when the cache provides paged
runtime metadata.

Append K/V to the page pool and pass its metadata to attention for every query
shape. Neither choice changes cache dtype or gathers a dense K/V tensor.

This creates the Day 4 routing policy:

```plain
every query shape -> correct direct page-walking attention
```

Day 5 replaces the long-query schedule with paged FlashAttention without
changing this model-facing policy. `enable_paged_attention=False` is a Day 3
dense-gather teaching ablation, not the completed serving path.

This test decodes incrementally through the Week 3 model and compares each
step with the Week 2 dense model:

```bash
pdm run test --week 3 --day 4 -- -k task_3
```

## Task 4: Connect It to Continuous Batching

```
src/tiny_llm/batch.py
```

Modify `Request.try_prefill`, `Request.decode_done`, `_step`, and
`batch_generate`. Request cleanup must call `TinyKvPagedCache.release` for
every layer cache.

Update request admission, slot reuse, and request removal so that:

- finished requests free their pages,
- in this teaching implementation, that means freeing pages from every layer cache,
- new requests allocate from the corresponding layer pool,
- active decode steps reuse page metadata instead of rebuilding dense K/V.

Day 4's tests do not exercise this scheduler wiring. To see it end to end,
run the real model through your paged scheduler on CUDA:

```bash
pdm run main --loader week3 --solution tiny_llm \
  --model Qwen/Qwen3-0.6B --max-seq-len 64 --batch-size 1
```

After this chapter, the serving stack has the right structure for a real high-throughput runtime: paging is no longer just a storage trick, but part of the execution model itself.

## Measure the Direct Page Walk

The goal of this lab is to decide when direct page traversal is useful. Paged
attention is not automatically a faster attention operator: it trades regular,
contiguous K/V access for flexible allocation and removes the dense repack that
would otherwise happen before attention. Your measurements must include both
sides of that trade.

Record three operator baselines on the same machine:

1. the dense Week 2 attention path in your solution, including any required
   K/V gather,
2. your direct paged-attention path,
3. PyTorch SDPA (`torch.nn.functional.scaled_dot_product_attention`) as a
   production-library baseline.

Use the same Qwen3-4B decode shape for all three paths with explicit
`torch.cuda.synchronize()` around timed iterations. The dense control must
include its required page-to-dense gather; the direct path reads the same page
metadata; the PyTorch SDPA row measures its fused attention operator on the
already gathered tensor. (The older `bench-week3-attention` script remains a
historical MLX/Metal benchmark.)

The table below records historical task #367 MLX/Metal results from source
`18aec8503929d80c986324578068ecac2463c2ac`: medians of four balanced
fresh-process medians, with 60 synchronized calls after five warmups per
process. A current CUDA run will produce different values; it need not
reproduce those numbers.

| Context | Dense + gather | Direct paged | MLX fused |
|---:|---:|---:|---:|
| 128 | 201.26 us | 228.58 us | 188.79 us |
| 1,024 | 468.39 us | 299.14 us | 250.04 us |

In that predecessor trace, direct traversal was 13.6% slower than
dense-plus-gather at 128 tokens, but 36.1% faster at 1,024 tokens. MLX was
faster at both shapes. The recorded BF16 outputs matched the readable dense
equation within 0.00439453125 at
`S=128` and 0.001953125 at `S=1,024`. This operator benchmark contains no model
projection, so it isolates the attention paths directly.

### Checkpoint 1: Establish a Correct Direct Path

Implement the simplest page-walking kernel first. Verify that it:

- reads K/V through `block_table` without constructing a dense K/V tensor,
- ignores unused slots in the final page,
- matches dense attention for several page boundaries and context lengths,
- supports grouped-query attention when `H_q != H_kv`.

The correctness schedule may be slower than dense attention. At this
checkpoint, the useful result is a trustworthy baseline and a working runtime
interface.

### Checkpoint 2: Design the Decode Schedule

Optimize for the one-token decode shape instead of treating page traversal as a
serial loop. Work through these changes one at a time and benchmark after each
one:

1. assign the lanes of a warp to adjacent elements of a head so K/V loads
   can be coalesced,
2. load a page-table entry once and reuse it for all positions in that page,
3. keep the query and online-softmax state in registers across page tiles,
4. combine partial dot products with warp shuffle reductions (`__shfl_xor_sync`)
   instead of shared memory and repeated barriers,
5. specialize the `L = 1` decode case so it does not carry prefill control flow,
6. prepare the batch's page metadata once per scheduler step rather than once
   per layer or attention head.

Also benchmark the write path separately. A fast page-reading kernel cannot
recover time lost to a functional whole-cache update before every layer.

For each change, explain which cost it targets: memory traffic, synchronization,
address calculation, or dispatch overhead. Keep a change only when the measured
result supports the explanation.

Optimize the paged path in your solution for the Qwen head dimension of 128,
but keep one grouped-query schedule rather than duplicating the online-softmax
recurrence for individual GQA ratios. This keeps the relationship among page
traversal, head mapping, and reduction visible in one kernel.

### Checkpoint 3: Evaluate the Serving System

Operator latency alone does not capture the purpose of paging. Run an
end-to-end workload with requests entering and leaving the batch, then report:

- time per decode step and aggregate tokens per second,
- peak KV-cache memory and the number of live requests admitted,
- bytes or time spent gathering and repacking K/V,
- paged-attention latency relative to the dense Week 2 path in your solution
  and PyTorch SDPA.

Keep the dense path as a teaching ablation so you can measure when contiguous
attention is faster. The completed serving route stays paged: it eliminates
repacks, reuses pages across scheduler steps, and leaves more measured KV
headroom in the fixed-batch trace. Proving that it admits more concurrent
requests requires a memory-capped admission sweep. Day 5 optimizes its
long-prefill schedule rather than routing around the page-table contract.

(The older `bench-serving-progression` runner remains a historical MLX/Metal
script.) The numbers that follow are from historical task #367 on predecessor
MLX/Metal source `18aec8503929d80c986324578068ecac2463c2ac`; they do not
describe today's PyTorch/CUDA checkout. The direct path's
four-process medians were 672.68 prefill tok/s, 46.36 output tok/s, 105.01
decode tok/s, and 0.618 requests/s. Its synchronized decode calls took
28.97/36.78/63.04 ms at median/p95/max; the completion gaps, which include
intervening scheduler and prefill work, were 30.16/222.18/239.49 ms.

Those are historical cumulative system results, not an isolated Day 4 kernel
speedup. The ledger at
`benchmark_results/task367-final-main/task367-final-main-benchmark-ledger.md`
records the fixed trace, balanced process order, and denominator boundary.

```bash
pdm run test --week 3 --day 4
```

{{#include copyright.md}}
