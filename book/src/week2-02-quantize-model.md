# Week 2 Day 2: Keep W4 Packed

Complete [Day 1](./week2-01-kv-cache.md) first. Today keeps the capacity-bounded
model but replaces dense projection weights with group-affine four-bit
weights. The active backend is native C++/CUDA with PyTorch tensors.

## Build and Debug CUDA

```bash
nix develop --command pdm run build-ext
nix develop --command pdm run build-ext-ref
nix develop --command pdm run build-ext-test
```

`src/cuda/student.cu` contains learner seams. `reference.cu` is the separate
reference implementation. `bindings.cpp` exports the interfaces for both,
but each build links only its own implementation. Student kernel stubs fail
explicitly; they never invoke reference kernels. Builds use one compiler
worker and live in the external cache described in [CUDA.md](../../CUDA.md).
The build smoke check loads bindings without executing unfinished kernels.

For a launch failure, reduce to a tiny shape, use `CUDA_LAUNCH_BLOCKING=1`,
and synchronize after the suspect operation. Use NVIDIA Compute Sanitizer
when available to investigate invalid reads/writes. Never treat a skipped
CUDA case as validation of a native kernel.

## Represent Weights With Fewer Bits

For every group of 128 weights, store BF16 `scale` and `bias`. Each weight is
approximated as `q * scale + bias`, where integer code `q` lies in `[0,15]`.
The supplied `tiny_llm_cuda.packing.pack_weights` converts Hugging Face weight tensors
into this format. It rounds the scale and bias to their stored precision
before computing codes, so the readable control and kernel reconstruct the
same values.

Eight four-bit codes fit in one 32-bit word. PyTorch stores that bit pattern
as `torch.int32`; interpret the word as unsigned in CUDA before shifting.
For logical column `k`, word index is `k // 8`, shift is `4 * (k % 8)`, and
code is `(word >> shift) & 15`. Scale/bias index is `k // 128` within the
weight row. Shape `[N,K]` becomes `[N,K/8]` packed words and `[N,K/128]`
scales/biases. K must be divisible by 128 on this route.

## Task 1: Implement Quantized Linear and Embedding

Work in `src/tiny_llm/quantize.py` and `quantized_embedding.py`.
`QuantizedWeights` and checkpoint packing are supplied configuration.
Implement `dequantize_weights` with FP32 reconstruction and a cast to the
scale dtype. For `QuantizedEmbedding.__call__`, select token rows in packed
weights, scales and biases **before** dequantization. Do not expand the
whole vocabulary table for a handful of tokens.

Implement `torch_quantized_linear` as a readable control that reconstructs
the weight matrix and calls `torch.nn.functional.linear`. This control is
for correctness comparisons, not the custom packed path.

```bash
nix develop --command pdm run test --week 2 --day 2 -- -k task_1 -q
```

## Task 2: Define the Quantized Matmul Primitive

`cuda_ops.py` supplies checked native boundaries. Native W4 uses BF16
activations/scales/biases, int32 packed words, and `transpose_b=True`:
`[... ,K] @ [N,K]^T -> [... ,N]`. Validate rank, shape, device, bit width,
group size and contiguity before launching. The native kernels assume this
checked contract. `bits != 4`, unsupported transpose and historical Split-K
requests must fail instead of silently falling back.

Flatten activation leading dimensions into M rows, pass contiguous tensors,
then restore leading dimensions in the returned output.

## Task 3: Implement CUDA Matrix Products

Implement `w4_cuda` and its helpers in `src/cuda/student.cu`:

- Mode 0 is a readable scalar control: one thread owns an output and loops
  over K, unpacking weights directly. Accumulate in FP32 and store BF16.
- Mode 1 is decode matvec: one warp owns an output, lanes stride over K,
  then reduce with shuffle instructions. Multiple warps handle neighboring
  outputs. Guard partial output tiles and support M from 1 through 8.
- Mode 2 is the cooperative matrix schedule added on Day 3.

Round each reconstructed weight to BF16 before multiplication, matching the
readable control's stored matrix. Neither custom mode may allocate a full
dequantized weight matrix. Launch on PyTorch's current CUDA stream and guard
the input device with `CUDAGuard`. Check launch errors without forcing a
synchronization inside every operator.

Wire `quantized_matmul_vanilla` to mode 0 and `quantized_matvec_custom` to
mode 1. Keep the scalar operator callable for small inputs too.

```bash
nix develop --command pdm run test --week 2 --day 2 -- -k task_3 -q
```

## Task 4: Integrate Before Continuing

`quantized_linear` selects warp matvec for flattened `M <= 8`, otherwise
scalar matrix control until Day 3. Route all projection weights and the
vocabulary output head through this function, including tied embeddings.
Implement `QuantizedEmbedding.as_linear` through the packed operator.
The `quantized-matvec` model checkpoint must contain `QuantizedWeights`, not
quietly retain dense tensors. Norms remain BF16 and unquantized.

```bash
nix develop --command pdm run test --week 2 --day 2 -- -q
nix develop --command pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint quantized-matvec --model Qwen/Qwen3-0.6B --max-tokens 32
```

## Verify Quantization in the Complete Model

Compare a custom packed model with the same packed weights reconstructed by
the readable PyTorch control, then compare the product against Day 1.
Quantization changes model weights, so do not demand exact equality with the
original BF16 model. Tests use deterministic tiny Hugging Face fixtures with
both tied and untied vocabulary heads.

```bash
nix develop --command pdm run bench-week2-cuda --solution tiny_llm \
  --checkpoint capacity-cache --checkpoint quantized-matvec \
  --prefill-tokens 128 --decode-tokens 16 --prefill-logits all --repeats 3
```

Measure rather than assuming fewer weight bytes imply faster inference.
The readable scalar kernel deliberately provides a simple control, not an
optimized GEMM. Continue with [Day 3](./week2-03-simd-matrix-prefill.md).

## Profile Before and After

Follow [Find the Next Bottleneck](./week2-profile-guided.md): save the
`capacity-cache` baseline before editing, state a mechanism, run correctness
checks, then compare `quantized-matvec` under the same request workload.
Retain regressions and mark keep/reject/inconclusive; an operator share is
not a predicted request speedup.

```bash
nix develop --command pdm run profile-week2-kernels --solution tiny_llm \
  --model Qwen/Qwen3-0.6B --case capacity-cache:decode:128 \
  --case quantized-matvec:decode:128 --warmup 4 --iterations 12 \
  --json-output quantized-matvec-decode.replay.json
```

For prefill changes, repeat both cases with `prefill` instead of `decode`.
The CUDA request benchmark also reports peak allocated tensor bytes and full
request wall time. Preserve their recorded scopes and source/workload identity.

{{#include copyright.md}}
