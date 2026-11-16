# Week 2 Day 5: Tiled Dense Prefill Attention

Complete [Day 4](./week2-04-fused-model-kernels.md). Today adds tiled
online-softmax attention to the cached, packed CUDA model. Keep `swiglu` as
the pre-edit product control. The new checkpoints are `tiled-prefill` and
`selected`. The learner's `selected` checkpoint names the final teaching feature
set. Reference `selected` also enables guarded decode/fusion optimizations;
it is a separate control rather than another learner task.

## Task 1: Make Tiled Attention Match the Readable Result

Implement `dense_prefill_attention_mma` in `src/tiny_llm/week2_kernels.py` and
`attention_cuda` in `src/cuda/student.cu`. The historical wrapper name is
retained; it does not prescribe CUDA WMMA for attention. The CUDA reference
uses warp dot-product reductions and shared-memory tiles, separate from Day
3's WMMA projection kernel.

Inputs are BF16 Q `[B,Hq,L,128]`, K/V `[B,Hkv,S,128]`, with Hq divisible by
Hkv and L >= 9. Return the same shape/dtype/device as Q. GQA maps query head
h to KV head `h // (Hq/Hkv)` without physically repeating the K/V arrays.

The reference schedule is BQ4/BK16: four warps own four query rows, while
a block stages 16 key/value rows in shared memory. Each lane owns four of
the 128 output coordinates. Reuse the staged tile across query rows and
iterate over the context. This is the CUDA teaching schedule; historical
Metal BQ32/BK16 instructions do not describe this kernel.

Maintain a running maximum m, exponential sum l, and weighted value vector
o for each query. When incorporating score s:

```text
new_m = max(m, s)
alpha = exp(m - new_m)
p = exp(s - new_m)
l = alpha*l + p
o = alpha*o + p*value
m = new_m
```

Initialize m to negative infinity, l/o to zero. Skip masked scores before
rescaling so `-inf - -inf` cannot produce NaNs. Divide o by l when l > 0,
otherwise return zero. Accumulate scores, maxima, sums and outputs in FP32,
then store BF16. Never materialize the full `[B,Hq,L,S]` score matrix.

Guard query and key tails, but keep all block threads participating in
shared-memory barriers. An invalid query row must not return early while
other warps still need a barrier.

```bash
nix develop --command pdm run build-ext
nix develop --command pdm run test --week 2 --day 5 -- -k online_softmax -q
```

## Task 2: Preserve Masks and the Readable Fallback

Causal masking is bottom-right aligned: key position s is allowed for query
i when `s <= S-L+i`. Additive masks broadcast to `[B,Hq,L,S]`; the wrapper
converts them to contiguous FP32. Fully masked rows return zeros. Reject
unsupported strings, incompatible GQA shapes and wrong dtypes before a
native launch.

Keep PyTorch scaled-dot-product attention for short-query decode, non-BF16
inputs, or head dimensions other than 128 in the teaching `tiled-prefill`
checkpoint. Reference `selected` uses native online decode for eligible
single-token BF16/D128 queries up to 32768 context tokens, splitting contexts
over 256 tokens into stable softmax partitions. The CUDA kernel reads
capacity-strided K/V directly and shares long-context K/V tiles for GQA4. `DENSE_PREFILL_MIN_QUERY=9`
selects the tiled route; the direct wrapper rejects shorter queries.
The operator tests cover L=9/17/33, rectangular contexts, grouped heads,
additive masks and fully masked rows.

## Task 3: Run `tiled-prefill` in the Model

Only attention changes relative to `swiglu`. Route eligible prefill through
your native operator, while preserving cached decode and all earlier packed
and fused features. Track `tiled_prefill`, `tiled_prefill_fallback`, and
`readable` dispatch counts so tests can verify the intended route actually
ran instead of merely accepting numerically similar output.

```bash
nix develop --command pdm run test --week 2 --day 5 -- -q
nix develop --command pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint tiled-prefill --model Qwen/Qwen3-0.6B --max-tokens 32
```

## Finish the Cumulative `selected` Model

Run the final named checkpoint on the same cached model:

```bash
nix develop --command pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint selected --model Qwen/Qwen3-0.6B --max-tokens 32
```

Reference runs use `--solution tiny_llm_ref`. They verify the port and provide
controls; they do not complete student exercises. No checkpoint imports MLX,
compiles Metal or requires Apple tooling on this route.

## Measure the Matched Product

```bash
nix develop --command pdm run bench-week2-cuda --solution tiny_llm \
  --checkpoint swiglu --checkpoint tiled-prefill --checkpoint selected \
  --prefill-tokens 128 --decode-tokens 16 --prefill-logits all --repeats 3 \
  --profile /tmp/week2-day5.json
```

Report prefill and decode separately with model, GPU, shapes, logit mode,
warmups and repetitions. This educational tiled kernel may be slower than
PyTorch's optimized SDPA. Online softmax bounds intermediate score storage;
it does not guarantee lower latency. Historical Apple benchmark results and
Metal capture commands are not measurements of this CUDA checkout.

Week 2's active CUDA route ends here. Week 3 remains the historical MLX/Metal
route and requires a separate backend port before tutoring on Linux/CUDA.

## Profile Before and After

Follow [Find the Next Bottleneck](./week2-profile-guided.md): save the
`swiglu` baseline before editing, state a mechanism, run correctness
checks, then compare `tiled-prefill` under the same request workload.
Retain regressions and mark keep/reject/inconclusive; an operator share is
not a predicted request speedup.

```bash
nix develop --command pdm run profile-week2-kernels --solution tiny_llm \
  --model Qwen/Qwen3-0.6B --case swiglu:decode:128 \
  --case tiled-prefill:decode:128 --warmup 4 --iterations 12 \
  --json-output tiled-prefill-decode.replay.json
```

For prefill changes, repeat both cases with `prefill` instead of `decode`.
The CUDA request benchmark also reports peak allocated tensor bytes and full
request wall time. Preserve their recorded scopes and source/workload identity.

{{#include copyright.md}}
