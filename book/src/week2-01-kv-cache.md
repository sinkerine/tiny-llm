# Week 2 Day 1: Reuse the Prefix, Then Bound the Cache

This chapter uses PyTorch on Linux/CUDA. Complete Week 1 first. The active
Week 2 path loads the same Hugging Face Qwen3 checkpoint; it needs neither
MLX nor Xcode. See [CUDA setup](../../CUDA.md) for the native build environment.

Today you own `src/tiny_llm/kv_cache.py`, `qwen3_week2.py`, and
`generate_week2.py`. Keep the reference solution separate from your exercises.
The checkpoint names are `kv-cache` and `capacity-cache`.

## Task 1: Implement the Key-Value Cache

Attention projects each new hidden state into a key and value. Once a token
has been processed, its key and value do not change during ordinary decoding.
Retain them rather than projecting the entire prefix again.

`TinyKvFullCache.update_and_fetch` receives K/V chunks shaped `[B,Hkv,S,D]`.
Append on dimension 2. Return `(keys, values, offset, mask)`, where `offset`
is the total logical length **after** the append. Pass the mask through.
For example, append a three-token chunk and then a one-token chunk: the
returned lengths are 3 and 4. Queries contain only the new tokens; attention
keys and values contain all four tokens.

Start with the unbounded cache: retain the first chunk, then use `torch.cat`
on subsequent appends. Preserve dtype/device. Track bytes copied from the
old prefix in `logical_copy_bytes`, `physical_growth_copy_bytes`, and
`growth_copy_bytes`. PyTorch is eager, so there is no `mx.eval` step.

```bash
nix develop --command pdm run test --week 2 --day 1 -- -k full_cache -q
```

## Task 2: Build the Cached Week 2 Model

Port your Week 1 transformer assembly into the Week 2 learner seams. The
constructor receives a Hugging Face model: configuration is `hf_model.config`
and parameters are under `hf_model.model`. Read `head_dim` explicitly.
Initialize `embedding`, `layers_inner`, final normalization, and an optional
untied output head. Preserve the pre-normalization residual ordering.

Every attention call receives `x [B,L,hidden]`, an offset, and a per-layer
cache. Project and normalize Q/K, apply RoPE at positions `offset..offset+L`,
reshape to `[B,H,L,D]`, append new K/V, and attend to the returned prefix.
For causal cached chunks, query row `i` can see key positions through
`S-L+i`. With one new token, every retained key is visible.

Keep `kv-cache` readable: use ordinary PyTorch linear layers, norms, RoPE,
and attention. CUDA's `scaled_dot_product_attention` supports GQA with
`enable_gqa=True`; provide an explicit bottom-right causal mask for rectangular
cached chunks. Its `is_causal=True` convention alone is not that mask.

Once the readable model and serving loop work, run the first live checkpoint:

```bash
nix develop --command pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint kv-cache --model Qwen/Qwen3-0.6B --max-tokens 32
```

## Task 3: Create Request-Scoped Caches

`create_kv_cache(capacity=None)` creates one cache per transformer layer.
Validate that all layer offsets agree with the requested model offset before
running the layers. `logits_to_keep=1` selects the final hidden position
before final normalization and vocabulary projection; `None` retains all
positions. Never reuse a previous request's cache accidentally.

The deterministic model tests compare chunked decoding with a complete
sequence and with the Hugging Face model, without downloading weights.

```bash
nix develop --command pdm run test --week 2 --day 1 -- -k 'cached_decode' -q
```

## Task 4: Connect the Serving Loop

Implement `cached_generate` separately from your completed Week 1 generator.
Encode the formatted prompt without adding special tokens, prefill once,
then send `[1,1]` containing only the newly sampled token on each decode call.
Advance the offset by the number of input tokens actually consumed. Stop on a
new EOS or the generation limit, stream CPU tokens, and release every cache
in a `finally` block. Reject empty prompts and invalid limits; zero tokens
requires no model call.

## Second checkpoint: bound the request cache

For `capacity-cache`, allocate K/V storage shaped `[B,Hkv,capacity,D]` once.
Capacity comes from prompt length plus the generation limit. Slice-write the
new chunk into `[offset:end]`, then return views ending at the **logical**
length. Uninitialized capacity must never reach attention.

Check overflow before writing or changing offsets. `reset` sets the logical
length to zero while retaining bounded storage. `rewind(n)` removes the
newest `n` logical tokens; later writes replace that suffix. In unbounded
mode, discard or trim the physical suffix as appropriate.

Track `slice_write_bytes` for new K/V writes. Bounded appends should not copy
the old prefix or grow physical storage. Counters explain the mechanism;
whole-request timing determines whether it helps.

```bash
nix develop --command pdm run test --week 2 --day 1 -- -q
nix develop --command pdm run main --solution tiny_llm --loader week2 \
  --week2-checkpoint capacity-cache --model Qwen/Qwen3-0.6B --max-tokens 32
```

Nsight captures, screenshots and device-specific counters are optional
evidence; they are not acceptance gates. Required checks are the deterministic
tests and the live cached model path described above.

## Measure the First Cache Change

Use the CUDA benchmark, which requires the model to be cached already:

```bash
nix develop --command pdm run bench-week2-cuda --solution tiny_llm \
  --checkpoint week1 --checkpoint kv-cache --checkpoint capacity-cache \
  --prefill-tokens 128 --decode-tokens 16 --prefill-logits all --repeats 3
```

The runner warms each model, synchronizes CUDA at timing boundaries, creates
fresh caches for every request, and reports prefill/decode separately. It
uses equal synthetic input lengths and fixed decode counts, avoiding EOS
length differences. `all` matches Week 1's full-logit work; `last` measures
Week 2's serving optimization and must be labeled separately. No speedup is
guaranteed by the checkpoint name. Repeat runs and change order to investigate
noise. Add `--profile /tmp/week2.json` for per-checkpoint PyTorch CUDA traces.

The Day 1 reference tests can run without compiling extensions. Build native
code before starting [Day 2](./week2-02-quantize-model.md).

## Profile Before and After

Follow [Find the Next Bottleneck](./week2-profile-guided.md): save the
`kv-cache` baseline before editing, state a mechanism, run correctness
checks, then compare `capacity-cache` under the same request workload.
Retain regressions and mark keep/reject/inconclusive; an operator share is
not a predicted request speedup.

```bash
nix develop --command pdm run profile-week2-kernels --solution tiny_llm \
  --model Qwen/Qwen3-0.6B --case kv-cache:decode:128 \
  --case capacity-cache:decode:128 --warmup 4 --iterations 12 \
  --json-output capacity-cache-decode.replay.json
```

For prefill changes, repeat both cases with `prefill` instead of `decode`.
The CUDA request benchmark also reports peak allocated tensor bytes and full
request wall time. Preserve their recorded scopes and source/workload identity.

{{#include copyright.md}}
