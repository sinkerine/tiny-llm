# Week 2 CUDA port validation

Verified on kazuki, 2026-11-15 (PST, UTC−08:00), using NVIDIA GeForce RTX
5070 Ti, driver 595.112.07, the repository Nix development shell, and the
existing cached `Qwen/Qwen3-0.6B` model. No model downloads were performed.

## Results

- Learner and reference native extensions built successfully. Learner bindings
  loaded without implementing any exercise kernels.
- Week 2 reference suites: Day 1 **8 passed**, Day 2 **11 passed**, Day 3
  **5 passed**, Day 4 **20 passed**, Day 5 **14 passed**: **58 total**.
- Final combined Week 1 reference, Week 2 reference, and Week 1 student run:
  **622 passed, 4 skipped**. The four skips are the two optional larger model
  checkpoints in each Week 1 suite. Required Qwen3-0.6B checks passed.
- All nine Week 2 checkpoints completed one warmed, fixed-count 0.6B
  prefill/decode workload (10 input tokens, 2 decode tokens, all logits).
- The final `selected` reference model also streamed a response through the
  public CLI. Its 48-token generation limit can truncate the final sentence.
- Week 2 student exercises remain unfinished. Reference verification does not
  constitute completion of those exercises.

Raw evidence: [regression](regression.log), [checkpoint smoke timings](checkpoint-smoke.jsonl),
[CLI generation](generation.log).

The checkpoint timings use one measured repetition and a small input shape.
They establish execution, not a robust performance comparison. In particular,
the teaching WMMA prefill path was slower than the readable controls at this
shape; no speedup over optimized PyTorch is claimed.

## Reproduction

```bash
nix develop --command pdm run build-ext-ref
nix develop --command pdm run build-ext
nix develop --command pdm run build-ext-test
nix develop --command python -m pytest \
  tests_refsol/test_week_1_day_*.py tests_refsol/test_week_2_day_*.py \
  tests/test_week_1_day_*.py -q --tb=short
HF_HUB_OFFLINE=1 nix develop --command pdm run main \
  --solution tiny_llm_ref --loader week2 --week2-checkpoint selected \
  --model Qwen/Qwen3-0.6B --temperature 0 --max-tokens 48 \
  --prompt 'In two sentences, explain how caching past attention keys and values speeds up language model decoding.'
HF_HUB_OFFLINE=1 nix develop --command pdm run bench-week2-cuda \
  --solution tiny_llm_ref --checkpoint kv-cache --checkpoint capacity-cache \
  --checkpoint quantized-matvec --checkpoint simd-matmul --checkpoint rmsnorm \
  --checkpoint rope --checkpoint swiglu --checkpoint tiled-prefill \
  --checkpoint selected --prefill-tokens 10 --decode-tokens 2 \
  --prefill-logits all --repeats 1
```

The regression initially exposed pre-existing Week 1 reference precision
mismatches with Hugging Face BF16. Reference SiLU now computes in FP32 and
casts back, RoPE casts sin/cos to activation precision, and model attention
uses native-dtype PyTorch SDPA with GQA. Student Week 1 code was not changed.

The active CUDA port includes no MLX imports or Metal builds. Historical Apple
sources remain separate; Weeks 3–4 are outside this validation. Tests include
cache append/rewind/reset/overflow, cached-vs-full/HF model behavior, generation
prefill/decode/release, selected-row W4 embeddings, native scalar/warp/WMMA
projections and partial tiles, current CUDA stream handling, FP32/FP16/BF16
normalization, per-batch RoPE offsets, SwiGLU, online-softmax GQA with causal
and additive masks, fully masked rows, and cumulative checkpoint dispatch.
