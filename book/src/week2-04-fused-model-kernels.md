# Week 2 Day 4: Fused Model Primitives

Complete [Day 3](./week2-03-simd-matrix-prefill.md) first. Today integrates
three native CUDA primitives into the cached, packed model in order:
`rmsnorm`, `rope`, then `swiglu`. Keep `simd-matmul` as the pre-edit control.

Work in `src/tiny_llm/week2_kernels.py` and `src/cuda/student.cu`. The supplied
`cuda_ops.py` validates tensors; your Python wrappers and CUDA kernels remain
learner-owned. Build after native edits:

```bash
nix develop --command pdm run build-ext
```

## Why Fusion Helps

Separate elementwise operations read/write intermediate tensors and launch
multiple kernels. A fused operation retains intermediates in registers or
shared memory and writes the final result once. Synchronize only where data
is shared across threads. Measure launch and memory costs before deciding
which fusion matters for a particular model shape.

## Task 1: RMSNorm

For each vector of width D, compute `x / sqrt(mean(x*x) + eps)` and multiply
by learned scale. Reduce squares in FP32. Match Week 1 rounding: cast the
normalized activation to the input dtype **before** multiplying by weight.

A block owns a row. Threads stride over D, reduce shared partial sums, then
write normalized values. Cache row values in registers for D <= 4096; use a
wider-row reread fallback to bound per-thread register storage. Update
`dispatch_counts` under the existing `register_cached` and
`fixed_width_fallback` keys in `FastRMSNorm`.

Use the new norm for input/post-attention norms, Q/K norms and final model
normalization at `rmsnorm` and later checkpoints. Preserve separate Q and K
learned weights. Tests cover FP32/FP16/BF16 and widths 128 through 4352.

```bash
nix develop --command pdm run test --week 2 --day 4 -- -k task_1 -q
```

## Task 2: RoPE

`FastRoPE` receives `[B,L,H,D]` with full even rotary dimension D, an integer
offset or one offset per batch row, and a frequency base. Create a contiguous
int64 `[B]` offsets tensor on the same CUDA device. Reject incompatible
shapes rather than interpreting an offset vector as a scalar.

At sequence index l and pair j, the angle is
`(offset[b] + l) * base ** (-2*j/D)`. Qwen's normal layout pairs coordinate j
with j+D/2; traditional layout pairs 2j with 2j+1. Compute angles in FP32,
cast sin/cos to input dtype and preserve the readable multiplication/addition
rounding. One CUDA thread can compute a pair and write both coordinates.

The `rope` checkpoint inherits fast RMSNorm. Apply rotations to Q/K before
the head/sequence transpose and cache update. Offsets must advance during
decode; repeatedly rotating position zero is a semantic bug.

```bash
nix develop --command pdm run test --week 2 --day 4 -- -k task_2 -q
```

## Task 3: SwiGLU

Fuse `silu(gate) * up` in one CUDA kernel, with matching shapes/dtypes and
FP32 internal SiLU arithmetic. Use the stable expression based on
`exp(-abs(gate))` to avoid overflow on large negative inputs. Match the
readable control's cast of SiLU back to the input dtype before its product
with `up`. The `swiglu` checkpoint uses this operation in every MLP.

```bash
nix develop --command pdm run test --week 2 --day 4 -- -k task_3 -q
```

## Task 4: Verify the Cumulative Model

```bash
nix develop --command pdm run test --week 2 --day 4 -- -q
nix develop --command pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint swiglu --model Qwen/Qwen3-0.6B --max-tokens 32
```

Model tests compare each checkpoint with the same `simd-matmul` model and
weights, for both prefill and decode. Passing isolated primitive tests is
necessary but does not establish model wiring.

## Measure the Cumulative Product

```bash
nix develop --command pdm run bench-week2-cuda --solution tiny_llm \
  --checkpoint simd-matmul --checkpoint rmsnorm --checkpoint rope \
  --checkpoint swiglu --prefill-tokens 128 --decode-tokens 16 \
  --prefill-logits all --repeats 3 --profile /tmp/week2-day4.json
```

Compare prefill and decode separately and inspect kernel attribution. Do not
reuse historical Apple timings as CUDA evidence. Continue with
[Day 5](./week2-05-tiled-prefill-attention.md).

## Profile Before and After

Follow [Find the Next Bottleneck](./week2-profile-guided.md): save the
`simd-matmul` baseline before editing, state a mechanism, run correctness
checks, then compare `swiglu` under the same request workload.
Retain regressions and mark keep/reject/inconclusive; an operator share is
not a predicted request speedup.

```bash
nix develop --command pdm run profile-week2-kernels --solution tiny_llm \
  --model Qwen/Qwen3-0.6B --case simd-matmul:decode:128 \
  --case swiglu:decode:128 --warmup 4 --iterations 12 \
  --json-output swiglu-decode.replay.json
```

For prefill changes, repeat both cases with `prefill` instead of `decode`.
The CUDA request benchmark also reports peak allocated tensor bytes and full
request wall time. Preserve their recorded scopes and source/workload identity.

{{#include copyright.md}}
