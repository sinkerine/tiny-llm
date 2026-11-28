# 🚧 Week 3 Day 5: Paged FlashAttention

> 🚧 This chapter is under review and may change.

In this chapter, you will replace the correctness-first implementation
for the supported BF16 long-prefill hot case with paged FlashAttention. The
operator still translates logical K/V positions through `block_table`, but it
now stages page-backed tiles on chip and combines them with online softmax.
Short decode and shapes outside the optimized region keep their correct Day 4
fallbacks.

Day 5 starts differently from an ordinary correctness checkpoint: the public
behavior tests may already pass with your Day 4 implementation. That is
intentional. Your work is to make the optimized path real and reachable while
preserving the same observable behavior. The manual control-flow trace at the
end of the chapter is the completion feedback for that performance change; it
is guidance, not a hidden grading requirement.

This is a required chapter. Week 2's dense tiled kernel teaches the
online-softmax update, but the serving model's K/V source is now the page
pool. Reuse that arithmetic while translating each K/V tile through the
block table; the dense loading schedule cannot read paged state directly.

## Prerequisites

This chapter combines four prerequisites:

- [Week 2 Day 5](./week2-05-tiled-prefill-attention.md) introduced the
  online-softmax recurrence for tiled dense prefill attention.
- [Week 2 Day 3](./week2-03-simd-matrix-prefill.md) introduced tensor-core
  WMMA tiling for quantized projection prefill.
- Week 3 Day 3 introduced physical pages and block tables.
- Week 3 Day 4 introduced direct page-walking attention and the decode
  schedule.

No new model dtype is introduced here. Preserve the Week 2 precision contract
at the `paged_attention` boundary.

## Why Optimize the Paged Path

A conventional attention expression materializes a score matrix with shape
`L × S`. A page-walking implementation can avoid gathering K/V and still make
that intermediate too large. Paged FlashAttention does both:

1. it resolves each K/V tile through `block_table` instead of gathering a dense
   cache;
2. it keeps only a query tile, one K/V tile, and online-softmax state on chip;
3. it writes the normalized output once after all visible pages are consumed.

The algorithm remains numerically equivalent attention under the course's BF16
rounding tolerance. Only the order of loads and reductions changes.

## Keep the Day 4 Interface

Do not add a second model-facing operator. Continue to call:

```python
paged_attention(
    query,
    key_pages,
    value_pages,
    block_table,
    context_lens,
    page_size,
    scale=scale,
    mask="causal",
)
```

The reference solution puts this shape dispatch inside `paged_attention_cuda`
in `src/cuda/student.cu`:

| Query shape | Schedule |
|---|---|
| `L <= 8` | Keep the Day 4 direct warp paged-attention kernel. |
| `L > 8`, BF16, `D == 128` | Use the tiled paged FlashAttention kernel (`paged_attention_tiled_bf16_d128_kernel`). |
| Every other supported shape | Keep the Day 4 direct page-walking fallback. |

The completed Week 3 model therefore has one paged-attention contract and an
optimized BF16 long-prefill region. The public tests do not grade a kernel
name, dispatch threshold, tile shape, private route flag, helper, or source
layout. A behaviorally equivalent implementation, including one built with an
external low-level library, is valid.

## Task 1: Tile Queries and Paged K/V

The CUDA reference walkthrough adds `paged_attention_tiled_bf16_d128_kernel` in
`src/cuda/student.cu`. Keep your Day 4 `paged_attention_direct_kernel`
unchanged; it remains the short-query and generic control.

In the CUDA reference schedule (matching Week 2 Day 5's dense tiled kernel),
each thread block launches 4 warps (128 threads) to cover a `BQ = 4` query
block — one warp per query row — and stages `BK = 16` logical K/V positions
per iteration into shared memory. (You may choose a different tile size or
cooperative loader as long as the operator retains direct page-table semantics
and equivalent public behavior.)

For every logical key row in a tile:

```plain
logical_position = tile_start + row
logical_page     = logical_position / page_size
slot             = logical_position % page_size
physical_page    = block_table[batch, logical_page]
address          = pages[physical_page, kv_head, slot, :]
```

Resolve the physical page while staging the K/V tile into shared memory. Once
staged, the inner dot-product and value-accumulation loops do not need to know
whether two adjacent logical rows came from adjacent physical pages.

Tail cases are required. A query block, K/V tile, final page, or context may be
partially full, and physical page ids need not be consecutive.

## Task 2: Compute Tiled Online Softmax

Continue modifying `paged_attention_tiled_bf16_d128_kernel` in
`src/cuda/student.cu`. This task fills the tiled online-softmax body; it does
not add another public function.

For each query row in the tile, maintain one running maximum, one running sum,
and an unnormalized FP32 output accumulator across its warp. For each staged
K/V tile:

1. compute `Q @ Kᵀ` in FP32 across the warp lanes (reducing with
   `__shfl_xor_sync`);
2. apply scale and causal bounds;
3. merge the tile maximum into the running maximum;
4. rescale the previous sum and output accumulator;
5. compute exponentials (`expf` or `exp2f` with a `log2(e)`-scaled score) for
   the current scores and update the running sum;
6. multiply the tile probabilities by the staged `V` tile and update the output
   accumulator.

After the final visible tile, divide each output row by its running sum and
store it as BF16.

The causal offset is `context_len - L`. A key at logical position `s` is visible
to query row `l` when:

```plain
s <= l + context_len - L
```

Skip a whole K/V tile when its first key is beyond the last visible key for the
query block. This is both a correctness rule and an important causal-prefill
optimization.

## Task 3: Validate the Page Boundary

Complete the long-query selection in `paged_attention_cuda` in
`src/cuda/student.cu`, then test `paged_attention_tiled_bf16_d128_kernel`
against the Day 4 kernel. Keep the pybind11 binding in `src/cuda/bindings.cpp`,
`cuda_ops.paged_attention`, and the Python `paged_attention` signature
unchanged.

Use the GPU-debugging ladder from Week 2 Day 3:

1. compare Day 4 page-walking attention with the readable PyTorch equation
   (`scaled_dot_product_attention_grouped`);
2. compare paged FlashAttention with the Day 4 path;
3. trace the model-to-kernel route, then design a matched operator benchmark
   before making a speed claim.

The behavior fixtures cover:

- a context contained in one page;
- a partial query block and context tail with non-consecutive physical pages;
- poisoned unused pages and tail slots, which must not affect the result;
- batched GQA where multiple query heads map to one K/V head;
- causal and non-causal calls, including an explicit scale;
- short decode and generic BF16 head-dimension fallbacks;
- a model-level long-prefill comparison through the public Week 3 path;
- output shape and dtype, plus numerical agreement with dense attention.

These are public input/output checks. They do not inspect allocator choices,
page identifiers chosen by an implementation, kernel names, or routing state.

Check every launch with `C10_CUDA_KERNEL_LAUNCH_CHECK()`, and call
`torch.cuda.synchronize()` when debugging or timing so asynchronous CUDA errors
are reported at the responsible call.

The new kernel and dispatch live in the extension, so rebuild it before every
test run; otherwise the tests still exercise your previous Day 4 build:

```bash
pdm run build-ext
pdm run test --week 3 --day 5
```

If this command is green before you begin Day 5, you have confirmed that Day
4's correctness boundary is intact. Continue with the optimized implementation
and use the trace below to confirm that the supported hot case no longer takes
the scalar fallback.

## Task 4: Integrate and Measure

Verify the existing dispatch in `Qwen3MultiHeadAttention.__call__` and the
shape selection inside `paged_attention_cuda` in `src/cuda/student.cu`. Task 4
adds no new extension function.

The Week 3 model should use the tiled paged path automatically for supported
long prefills (`L > 8`, BF16, `D == 128`). Short queries continue through the
direct warp paged-attention schedule. Neither path gathers a dense K/V tensor.
Week 3 reuses your Week 2 packed W4 CUDA projections (with the optional
`use_torch_quantized_linear` PyTorch control), while its cache, paged
attention, batching, and scheduling remain course-owned.

The historical task #367 continuous-serving trace below was recorded on the
predecessor MLX/Metal checkout (`18aec8503929d80c986324578068ecac2463c2ac`).
It is useful system context, but it is neither a CUDA measurement nor a matched
direct-versus-tiled Day 5 experiment. (The older `bench-serving-progression`
and `bench-course-progression` scripts remain historical MLX/Metal tools.) Do
not claim a Day 5 speedup from these rows. A separate CUDA benchmark should
hold inputs, page tables, precision, warmup, `torch.cuda.synchronize()`, and
every non-attention mechanism fixed while changing only the direct-versus-tiled
schedule.

FlashAttention is expected to matter more as prefill grows. Treat that as a
hypothesis until a matched operator benchmark measures it. It should not
replace the Day 4 decode schedule: a one-token query has no query-tile reuse.

On that predecessor M4 Pro MLX/Metal trace, all three course rows shared the
same projection seam:

<div class="table-wrapper">
<table>
  <thead>
    <tr><th scope="col">Storage / attention path</th><th scope="col">Prefill tok/s</th><th scope="col">Output tok/s</th><th scope="col">Decode tok/s</th><th scope="col">Requests/s</th><th scope="col">Peak KV</th><th scope="col">Avoidable KV copy</th></tr>
  </thead>
  <tbody>
    <tr><th scope="row">Dense growth and reconstruction</th><td>711.18</td><td>35.23</td><td>57.59</td><td>0.469</td><td>1,096 MiB</td><td>209,532 MiB</td></tr>
    <tr><th scope="row">Paged storage + dense gather</th><td>725.46</td><td>41.64</td><td>78.53</td><td>0.555</td><td>not a total peak</td><td>103,445 MiB</td></tr>
    <tr><th scope="row">Direct paged attention</th><td>672.68</td><td>46.36</td><td>105.01</td><td>0.618</td><td>576 MiB</td><td>504 MiB</td></tr>
  </tbody>
</table>
</div>

Relative to dense serving in that predecessor trace, direct paging was 5.4%
lower on prefill, 31.6% higher on output/request throughput, 82.3% higher on
decode, and 47.4% lower on
measured peak KV storage, and 99.76% lower on avoidable logical copy volume.
The compatibility row's page-pool counter excludes its temporary dense staging
allocation, so it is not a total peak. These are cumulative Week 3 system
results; they do not isolate the Day 5 prefill schedule from paging, direct
decode, allocation, or scheduling, and they do not credit the MLX projection
seam to paged attention.

The existing 8K static sweep below is another historical MLX/Metal cumulative
diagnostic from task #367 on source `18aec8503929d80c986324578068ecac2463c2ac`.
It does not isolate direct versus tiled paged attention, measure request
turnover, page reuse, or capacity, and does not describe today's PyTorch/CUDA
checkout. The [performance appendix](./appendix-performance.md) records the
historical serving and long-context measurements. Long-context decode remains a
Day 4 direct-kernel workload; do not credit a prefill schedule with a decode
gain.

<div class="table-wrapper">
<table>
  <thead>
    <tr><th scope="col">8K static checkpoint</th><th scope="col">Prefill tok/s</th><th scope="col">Decode tok/s</th></tr>
  </thead>
  <tbody>
    <tr><th scope="row">Week 2 course-owned projections</th><td>323.96</td><td>17.73</td></tr>
    <tr><th scope="row">Week 3 seam + course paged path</th><td>463.69</td><td>27.42</td></tr>
    <tr><th scope="row">Full MLX</th><td>639.73</td><td>28.37</td></tr>
  </tbody>
</table>
</div>

In that predecessor source, the cumulative Week 3 prefill row was 43.1% faster
than the former Week 2 Split-K row and reached 72.5% of full MLX at this shape.
The 43.1% denominator is not today's five-day `selected` checkpoint; no
current Week 2-versus-3 percentage follows from this table. It remains a
historical static diagnostic: it does not measure request turnover, page reuse,
admission capacity, the
projection seam, or the Day 5 schedule causally. Its decode row is the Day 4
vector schedule, not evidence for the tiled prefill kernel. Full method and raw
samples are in
`benchmark_results/task367-final-main/task367-final-main-benchmark-ledger.md`.

> **Trace the implementation, not its names.** Copy the prompt below into GPT
> or Claude after your behavior tests pass. This is learner guidance and is not
> part of automated grading.
>
> ```text
> Trace the actual control flow for Week 3 long-prefill paged attention. First determine whether the implementation stays in this repository or crosses into an external low-level library. For a repository implementation, start at the Python model call and follow native dispatch into CUDA (`src/cuda/bindings.cpp` and `src/cuda/student.cu`). For an external implementation, trace from the Python model call to the library boundary and record equivalent boundary evidence, including the call site, arguments, and selected backend. Determine whether BF16 queries with L > 8 and D == 128 reach the learner's optimized tiled or FlashAttention-equivalent implementation rather than the correctness fallback. Verify that short decode and unsupported or generic shapes retain correct fallback behavior. Cite file and line evidence for repository code and equivalent boundary evidence for external code, flag dead or unreachable paths, judge control flow rather than function or kernel names, and report findings only—do not modify files.
> ```

{{#include copyright.md}}
