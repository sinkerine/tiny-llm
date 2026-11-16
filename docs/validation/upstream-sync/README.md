# Upstream sync with CUDA translations

Synced upstream `skyzh/tiny-llm` main through `7d2e3b7` (18 commits after
`17a8cc6`). Verified on kazuki, 2026-11-16 (PST, UTC−08:00), with the local
RTX 5070 Ti, PyTorch 2.10.0+cu128, the repository Nix shell and cached
Qwen/Qwen3-0.6B. No model downloads were performed.

The pre-sync Week 2 CUDA work was uncommitted. It was preserved in Git stash
`e6e78ca506b432edb767bec4967c9ac51fc92a64` and backup ref
`refs/backup/cuda-pre-upstream-sync`, restored during the merge, then integrated
with upstream changes. The merge includes that existing port. Student Week 1
implementations are retained, and Week 2 exercise bodies remain unfinished.

## Logic translated to CUDA

| Upstream change | CUDA form |
|---|---|
| Online/split single-token attention | BF16/D128 native online softmax; 256-token partitions; stable FP32 merge; capacity-strided K/V reads without full-prefix contiguous copies |
| Long-context GQA4 | Four query warps share staged K/V tiles at context >=8192 |
| Deferred Q/K norm–RoPE graph rewrite | Explicit fused CUDA operation on raw activations for guarded [1,1,32,128]/[1,1,8,128] Q/K; already-materialized inputs keep ordinary RoPE semantics |
| Aligned packed-matrix schedule | Compile-time full-tile WMMA path, with the guarded partial-tile control retained |
| MLX submit-at-layer-18 flush | Inapplicable to eager CUDA launches; no layer-18 synchronization added. The MLX-specific True override is rejected explicitly. |
| Public profiling adapter | Same opaque public-operator contract with PyTorch/CUDA runtime bindings, group rotation, cache cleanup, workload identity and decision records |
| Per-request peak allocation | PyTorch allocator peak allocated bytes, reset after warmup before each measured request; separately scoped full-request wall time |
| Python-only Week 1 imports | Native CUDA loading remains deferred; subprocess check requires no MLX imports or native compilation |
| Week 1 test/document clarifications | PyTorch equivalents retained; MLX-only tolerance changes do not relax CUDA tests |
| Week 3 fixes and general docs | Upstream changes retained; Week 3 remains historical MLX/Metal and is outside CUDA runtime validation |

Reference `selected` enables the new decode/fusion controls. Teaching
`tiled-prefill` keeps its original readable decode behavior. New reference
optimizations are not additional learner tasks. The 4B-shaped fusion route
is exercised with a tiny synthetic Hugging Face configuration; no 4B model
integration run was performed.

## Validation

Final combined run: **739 passed, 4 skipped, 17 subtests passed**. The skips
are optional larger-model checks in the student/reference Week 1 suites.
The required cached 0.6B integrations passed. Syntax/undefined-name lint and
`git diff --check` passed. Learner and reference extensions both built, and
the learner binding smoke check passed without filling exercise kernels.

New reference coverage includes padded contexts 256/257/8192, fully masked
partitions, eager materialization fallback, explicit norm/RoPE fusion,
aligned and partial matrix tiles, selected-vs-teaching cached outputs,
actual fused model dispatch, control overrides and rejected MLX-only flushing.
New metric coverage checks per-request CUDA peaks and unmatched Week 1/last-logit
comparisons. Upstream profile lifecycle, output protection, source-cost helper
and progression checks were retained or translated where CUDA-specific.

The final public CLI printed `CUDA sync works.` using reference `selected`.
Actual public replay ran `capacity-cache`, `quantized-matvec` and
`tiled-prefill` with cached Qwen3-0.6B. Reference `selected` correctly reported
incomplete replay coverage instead of fabricating attribution for private
scheduling. A fixed 16-token prefill / 2-step decode smoke comparison recorded
request latency and allocated peaks for capacity, tiled and selected models.
These small, two-repeat, in-process timings are execution evidence, not a
controlled speedup claim. Peak scopes include resident source HF and active
course-model tensors, not incremental or physical device memory.

Evidence:

- [Combined regression](regression.log)
- [Public replay report](public-replay.json)
- [Request/allocator smoke report](request-smoke.jsonl)
- [Generation smoke](generation.log)

Reports were captured while the merge worktree was dirty; the replay metadata
records that state. Historical Apple private decoder/capture tools remain
optional (`*-metal` entrypoints) and were not run on CUDA. Weeks 3–4 and the
historical MLX extension-interface suite are outside this runtime gate.

## Reproduce

```bash
nix develop --command pdm run build-ext-ref
nix develop --command pdm run build-ext
nix develop --command pdm run build-ext-test
nix develop --command python -m pytest \
  tests_refsol/test_week_1_day_*.py tests_refsol/test_week_2_day_*.py \
  tests_refsol/test_week_2_reference_runtime.py tests/test_week_1_day_*.py \
  benches/test_week2_profile_adapter.py benches/test_profile_week2_kernels.py \
  benches/test_week1_native_import.py benches/test_bench_request_metrics.py \
  benches/test_week2_source_costs.py benches/test_bench_course_progression.py \
  benches/test_week2_cuda_metrics.py -q --tb=short
```

Active profiling instructions are in
[Find the Next Bottleneck](../../../book/src/week2-profile-guided.md).
