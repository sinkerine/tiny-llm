# Week 2: Find the Next Bottleneck

Read this after [Day 1](./week2-01-kv-cache.md), then return to it before
each later optimization. This is the PyTorch/CUDA form of the upstream
profiling-led route. A correct kernel can still slow a request down; keep
both the baseline and the result instead of assuming the next checkpoint wins.

## The Loop

1. Save a benchmark of the current checkpoint before editing.
2. Profile that checkpoint on the same model and request shape.
3. State a mechanism: which bytes, operations or launch overhead should change?
4. Make one learner-owned change.
5. Run focused correctness checks, then repeat the matched request benchmark.
6. Record keep/reject/inconclusive and the next experiment, even for a regression.

## Four Views of One Request

| View | CUDA tool | Answers | Does not establish |
|---|---|---|---|
| Request clock | `bench-week2-cuda` | Prefill/decode latency, full request wall time, peak allocated tensor bytes | Which operator caused the change |
| Public operator replay | `profile-week2-kernels` | Synchronized medians and shares for fixed operator groups | Actual model wiring or request latency |
| CPU/GPU timeline | PyTorch profiler trace via `--profile`, optionally Nsight Systems | Launch gaps, GPU work and CPU preparation on one timeline | Source-line hardware counters |
| Kernel/counter/source investigation | Optional NVIDIA Nsight Compute | Kernel limits and source evidence when compiled with appropriate metadata | A whole-request improvement |

Start with the request clock, narrow the investigation, then finish with the
request clock. PyTorch CUDA executes eagerly but launches asynchronously;
Python overhead and gaps between small kernels can dominate decode.

## View 1: The Request Clock

Use cached Hugging Face weights and the same input/output-work counts:

```bash
nix develop --command pdm run bench-week2-cuda --solution tiny_llm \
  --checkpoint kv-cache --checkpoint capacity-cache \
  --prefill-tokens 128 --decode-tokens 16 --prefill-logits all --repeats 3 \
  > week2-capacity-baseline.jsonl
```

This CUDA runner warms each checkpoint, creates a fresh cache per request,
and synchronizes phase boundaries. Its `decode-tokens` value is the number
of one-token decode calls after prefill. Compare fixed work, not EOS-dependent
response lengths. Run variants in both orders and repeat when differences
are near host/GPU noise. The runner is an in-process teaching measurement;
it does not claim upstream's separate-process isolation or balanced ordering.

`prefill_ms` and `decode_ms` have narrow phase clocks. `request_wall_ms`
includes request cache setup, phase execution, counters and cache release.
Allocator reset/query and the instrumentation synchronizations run outside
that full-request clock. The peak fields are:

- `request_peak_cuda_allocated_bytes`: one peak per measured request.
- `peak_cuda_allocated_bytes`: maximum of those peaks.
- `peak_cuda_scope`: the measurement definition, including resident model tensors.

The peak is PyTorch allocator **allocated** bytes after resetting peak stats
immediately before each measured request, after loading and warmup. It includes
the source Hugging Face model and active course-model tensors retained by this
runner. It is not a baseline-subtracted increment, reserved allocator bytes,
CUDA driver/context allocations, or physical GPU usage. Missing older fields
mean missing measurements, never zero. Do not compare CUDA bytes with historical
MLX allocator bytes as if the scopes were identical.

Use `--prefill-logits all` for matched work with Week 1. `last` selects only
Week 2's final prefill hidden position before vocabulary projection and is a
separate serving optimization; the runner rejects a Week 1/last comparison.

## View 2: Public Operator Replay

The supplied adapter profiles public operators without constructing or
inspecting your unfinished learner model. It loads source Hugging Face
weights, generates independent deterministic operands, and uses the selected
solution's projection, pointwise, attention and cache interfaces.

```bash
nix develop --command pdm run profile-week2-kernels --solution tiny_llm \
  --model Qwen/Qwen3-0.6B --case capacity-cache:decode:128 \
  --warmup 4 --iterations 12 --cache-capacity 257 \
  --json-output week2-capacity-decode.replay.json
```

Cases use `CHECKPOINT:PHASE:CONTEXT`. Decode has one query and a seeded
prefix of context−1 tokens. Prefill has context queries and an empty prefix.
Offset preparation stays inside the pointwise timing, once per sample. Each
cache sample prepares its prefix outside timing, appends inside timing,
checks logical shape/offset/counters, and releases all caches even on errors.
Groups rotate through measurement order rather than always measuring the same
one first.

Read the per-group median and its share of `replay_total_us`. That total is
the sum of synchronized group medians, **not** the latency of a real model
step. The actual model can have different launch gaps and scheduling. A
projection replay that gets twice as fast does not establish a request speedup.

The schema records model shape, effective features, exact workload identity,
source commit/dirty status, software and GPU. Compare only matching workloads.
A reference `selected` row reports `coverage.complete=false` with a reason:
its extra reference-only decode/fusion scheduling cannot be fully attributed
by the public teaching-operator replay. It has no fabricated zero total.
Use `tiled-prefill` for complete public attribution, and a real model timeline
for `selected`. Do not silently replace `selected` with a teaching checkpoint.

Optional `--decision-output` writes a separate decision record. Supply
`--baseline-checkpoint`, `--candidate-checkpoint`, `--dominant-category`,
`--hypothesis`, `--observed-effect`, `--decision`, and `--next-experiment`.
Both replay rows must have complete coverage and identical source, solution,
model, workload, host and software identities. Output paths must be distinct
and must not already exist.

## View 3: The System Timeline

```bash
nix develop --command pdm run bench-week2-cuda --solution tiny_llm_ref \
  --checkpoint tiled-prefill --checkpoint selected --prefill-tokens 128 \
  --decode-tokens 16 --prefill-logits all --repeats 3 \
  --profile /tmp/week2-timeline.json
```

This writes per-checkpoint Chrome traces. Examine CPU launch activity and CUDA
kernels together. Treat profiler runs separately from the uninstrumented
request numbers. Optional Nsight Systems offers another timeline view when
installed; it is not a required course dependency.

## View 4: Kernel Counters and Source Lines

Use NVIDIA Nsight Compute if you need source/counter evidence, following its
driver permissions and compiler line-information requirements. PyTorch Chrome
traces alone do not contain sampled source-line hardware costs. Do not label
a sum of trace event durations as source-line samples or an exact memory-bandwidth
measurement. Keep raw trace/counter artifacts and record their tool versions.

Upstream's private Apple capture decoder and its tests remain historical
optional tooling (`source-costs-week2-metal`). They do not run in this CUDA
route and their sampled Metal costs cannot be translated into measured CUDA
costs. The upstream [Apple case study](https://github.com/skyzh/tiny-llm/blob/7d2e3b7/book/src/week2-profile-guided.md)
is background methodology, not a performance claim about this checkout.

## Choose the Next Change

Day 2 investigates weight traffic and packed W4. Day 3 investigates repeated
weight reads across prefill rows. Day 4 investigates small pointwise passes
and launches. Day 5 investigates attention score storage. The profile chooses
which mechanism matters on your hardware; the checkpoint order preserves the
prerequisites for implementing and testing it. Save the result, including
regressions, before moving to the next day.

{{#include copyright.md}}
