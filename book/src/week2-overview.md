# Week 2: A Faster Single Request

The active Days 1–5 use PyTorch and native C++/CUDA on Linux, continuing the
Week 1 environment. Student exercises stay unfinished; reference solutions,
deterministic checks, model loading and measurement tools are supplied.

| Day | Objective | Cumulative checkpoints |
|---|---|---|
| [1: Cache and Measure](./week2-01-kv-cache.md) | Reuse K/V; bound storage by the request limit | `kv-cache`, `capacity-cache` |
| [2: Keep W4 Packed](./week2-02-quantize-model.md) | Selected-row embedding and native packed projection | `quantized-matvec` |
| [3: Tensor-Core Matrix Prefill](./week2-03-simd-matrix-prefill.md) | Cooperatively load BF16/W4 tiles with CUDA WMMA | `simd-matmul` |
| [4: Fused Model Primitives](./week2-04-fused-model-kernels.md) | Native RMSNorm, RoPE and SwiGLU | `rmsnorm`, `rope`, `swiglu` |
| [5: Tiled Dense Prefill Attention](./week2-05-tiled-prefill-attention.md) | Online softmax with GQA and mask preservation | `tiled-prefill`, `selected` |

Start with the same cached `Qwen/Qwen3-0.6B` Hugging Face checkpoint used in
Week 1. Do not download MLX-quantized checkpoints for this runtime. Supplied
packing converts BF16 model weights into group-affine W4 on CUDA.

After Day 1, read [Find the Next Bottleneck](./week2-profile-guided.md).
Each later day now starts from a saved baseline and profile, states a mechanism,
then compares correctness, full request latency and peak allocation under one
matched workload. Keep regressions and a decision record.

Day 1 is Python-only and needs no extension build to collect or run tests.
Before Day 2, build learner and reference CUDA extensions:

```bash
nix develop --command pdm run build-ext
nix develop --command pdm run build-ext-ref
nix develop --command pdm run build-ext-test
```

Use `pdm run test --week 2 --day D` for student checks and `test-refsol` for
reference controls. Always start with the day's smallest relevant task, then
run its full suite and a cached 0.6B product check. A reference pass does not
complete your learner TODOs. Native tests require a CUDA GPU; skipped native
tests are not completion evidence.

The CLI selects the CUDA path with `--loader week2`. The checkpoint still
named `simd-matmul` uses CUDA WMMA tensor cores, and its legacy Python flag
`use_simdgroup_matmul` is retained for checkpoint continuity. Attention uses
a separate BQ4/BK16 shared-memory schedule. The active chapters describe
these CUDA schedules rather than the old Metal fragment layout. Reference
`selected` additionally enables stride-aware online/split decode and guarded
Q/K normalization–RoPE fusion. The `tiled-prefill` teaching checkpoint retains
its readable short-query path. MLX's submit-at-layer-18 optimization is not
carried over: CUDA already submits eagerly on its current stream.

`pdm run bench-week2-cuda` compares synchronized prefill/decode timings,
full request wall time, peak allocated CUDA tensor bytes and cache byte counters.
`pdm run profile-week2-kernels` replays public operators in the CUDA backend;
reference `selected` scheduling is reported as incomplete coverage rather than
misattributed to the teaching operators. Use `--profile` for PyTorch CUDA traces. Keep workload
and logit mode matched and do not assume a teaching kernel is faster than
optimized PyTorch operators. The earlier Apple-oriented benchmark/capture
commands remain historical tools, not part of the CUDA route.

See [CUDA setup](../../CUDA.md) for prerequisites and scratch paths. Optional profiler traces and optional capture evidence do not gate progress.
Earlier seven-day pages and the [performance appendix](./appendix-performance.md)
are historical Apple material. Week 3 and later remain MLX/Metal and require
a separate port.

{{#include copyright.md}}
