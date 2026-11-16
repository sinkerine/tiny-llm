# Week 2 Day 3: Tensor-Core Matrix Prefill

Complete [Day 2](./week2-02-quantize-model.md) first. Keep its packed W4
format, scalar control, and warp matvec. Today replaces only the larger
matrix-shaped prefill path with a cooperative CUDA tensor-core kernel.

The public checkpoint remains `simd-matmul` for continuity. On CUDA,
`use_simdgroup_matmul=True` selects WMMA rather than an Apple SIMD-group
instruction. This legacy flag does not imply a Metal dependency.

## Keep the Pre-Edit Control

Measure `quantized-matvec` before editing. Keep model, input length, generated
length, prefill-logit mode and repetitions fixed for your later comparison:

```bash
nix develop --command pdm run bench-week2-cuda --solution tiny_llm \
  --checkpoint quantized-matvec --prefill-tokens 128 --decode-tokens 16 \
  --prefill-logits all --repeats 3
```

## Task 1: Load and Multiply a W4 Tile

Work in `src/cuda/student.cu`, mode 2 of the `w4` binding. Compute
`C[M,N] = A[M,K] @ W[N,K].T` without constructing the full W matrix.

Use CUDA WMMA BF16 fragments with FP32 accumulators. The reference schedule
uses one warp per 16x16 output tile and reduction slices of 16:

1. Cooperatively stage a 16x16 activation tile in shared memory, row-major.
2. Unpack the corresponding W4 weights, apply scale/bias, round to BF16,
   and stage a 16x16 column-major B tile representing `W.T`.
3. Synchronize the warp, load WMMA fragments and multiply-accumulate.
4. Repeat over K, then stage the FP32 accumulator for guarded output stores.

For a 10x97 output, the last column tile has one valid column and every row
tile has a partial tail. Zero-fill invalid loads and guard final stores.
Tests also exercise 17x33 and 32x64. Match native fragment alignment and
shared-memory layout requirements; an arithmetic equation does not specify
a valid tensor-core memory layout.

Use a device supporting BF16 tensor cores (compute capability 8.0 or newer).
The supplied Nix environment and build select the visible GPU architecture;
see [CUDA setup](../../CUDA.md) for overrides and compiler compatibility.

## Task 2: Select the Matrix Schedule

Keep `quantized_matmul(..., use_simdgroup=True)` wired to mode 2. The model's
`quantized_linear` still selects Day 2's warp matvec when `M <= 8`; larger
matrices use WMMA only when requested. Preserve the scalar control as a
separate callable operator. The native operator must not silently redirect
to a dense PyTorch matmul.

```bash
nix develop --command pdm run build-ext
nix develop --command pdm run test --week 2 --day 3 -- -k partial_tiles -q
```

## Task 3: Integrate Matrix Prefill

The `simd-matmul` checkpoint inherits bounded caches and packed projections.
Pass its matrix-schedule flag into every packed weight, including the tied
vocabulary projection. Tests use ten input tokens so matrix-shaped work
actually exceeds the decode matvec threshold, then verify one-token decode.

```bash
nix develop --command pdm run test --week 2 --day 3 -- -q
nix develop --command pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint simd-matmul --model Qwen/Qwen3-0.6B --max-tokens 32
```

## Measure the Matched Product

```bash
nix develop --command pdm run bench-week2-cuda --solution tiny_llm \
  --checkpoint quantized-matvec --checkpoint simd-matmul \
  --prefill-tokens 128 --decode-tokens 16 --prefill-logits all --repeats 3 \
  --profile /tmp/week2-day3.json
```

The PyTorch profiler records actual native CUDA launches. Compare prefill and
decode separately; the decode schedule is intentionally unchanged. A faster
projection alone does not prove a faster request. These are teaching kernels,
not a claim to beat cuBLAS or production quantized libraries.

Continue with [Day 4](./week2-04-fused-model-kernels.md).

## Profile Before and After

Follow [Find the Next Bottleneck](./week2-profile-guided.md): save the
`quantized-matvec` baseline before editing, state a mechanism, run correctness
checks, then compare `simd-matmul` under the same request workload.
Retain regressions and mark keep/reject/inconclusive; an operator share is
not a predicted request speedup.

```bash
nix develop --command pdm run profile-week2-kernels --solution tiny_llm \
  --model Qwen/Qwen3-0.6B --case quantized-matvec:decode:128 \
  --case simd-matmul:decode:128 --warmup 4 --iterations 12 \
  --json-output simd-matmul-decode.replay.json
```

For prefill changes, repeat both cases with `prefill` instead of `decode`.
The CUDA request benchmark also reports peak allocated tensor bytes and full
request wall time. Preserve their recorded scopes and source/workload identity.

{{#include copyright.md}}
