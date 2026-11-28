# 🚧 Week 3: Build a Mini vLLM

> 🚧 This overview and chapters carrying the same marker are under review and
> may change.

Week 3 turns the optimized single-request model into a multi-request serving
engine. Students add scheduling, request-owned cache state, shared page pools,
and the runtime metadata needed to read noncontiguous K/V directly. The final
model uses one page-aware attention interface. A correct direct page-walking
implementation may serve every query shape; the completed reference adds a
tiled schedule for the supported BF16 long-prefill hot case.

The [Week 2 `selected` model](./week2-05-tiled-prefill-attention.md) completes
the single-request route with tiled dense attention and course-owned packed W4
CUDA projections. On this PyTorch/CUDA branch, Week 3 reuses that same runtime
and native extension (`src/cuda/bindings.cpp`, `src/cuda/student.cu`), while
exposing an optional per-weight `use_torch_quantized_linear` selector
(`torch_quantized_linear`, which dequantizes W4 weights and calls
`torch.nn.functional.linear`) as a readable PyTorch projection control.
Normalization, activation, RoPE, cache state, attention, paging, and scheduling
all remain course-owned.


## What We’ll Cover

- Continuous batching and request-slot reuse
- Chunked prefill and scheduler fairness
- Paged KV storage and page-walking attention
- Paged FlashAttention for long prefill
- Optional Day 6 Mixture-of-Experts model support
- Optional Day 7 speculative decoding over rewindable caches

Day 1 introduces that projection seam and batches independent request states. Day 2
splits long prefills so they cannot monopolize the scheduler. Day 3 replaces a
growing dense cache with fixed-size pages while retaining a dense-gather
compatibility path. Day 4 removes that gather by teaching attention to walk the
page table directly with a correctness-first implementation; one kernel may
serve all query shapes. Day 5 then replaces the supported BF16 long-prefill hot
case with a tiled schedule. Page translation is therefore introduced before
it is optimized.

These five days form the required path in your solution. The final model in
your solution runs paged FlashAttention for supported BF16 long prefill and
retains correct direct fallbacks for short queries and generic shapes. Every
schedule reads the same page pool through the same block-table interface;
none rebuilds dense K/V.

Paged attention is not an automatic single-request latency win. The historical
task #367 trace, measured on source
`18aec8503929d80c986324578068ecac2463c2ac` before the five-day Week 2
revamp, showed lower KV storage, page reuse, incremental growth, and batching;
page-table indirection can make one request slower. Those figures do not
describe the current Week 2 `selected` baseline or establish an
admission-capacity gain without a memory-capped sweep. Each chapter ends with a
focused measurement you can run on this checkout. The
[performance appendix](./appendix-performance.md) preserves the older matched
chapter-by-chapter results with their source boundary.

Optional Day 6 adds MoE model support independently of the cache and scheduler.
Optional Day 7 then adds speculative decoding, whose rejection path needs a
precise cache rewind operation and whose multi-token verification needs the
page-aware long-query path. Neither extension is required to complete Week 3.

{{#include copyright.md}}
