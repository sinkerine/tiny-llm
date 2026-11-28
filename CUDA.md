# Weeks 1–3 on CUDA

Weeks 1–3 use PyTorch tensors on Linux with an NVIDIA GPU. Student tasks in
`src/tiny_llm` remain exercises; completed reference implementations are in
`src/tiny_llm_ref`. Weeks 2–3 use native C++/CUDA kernels. Week 4 remains
historical MLX/Metal and is outside this port. CUDA package imports
do not import MLX or eagerly build native extensions.

## Environment

```sh
nix develop
uv sync --locked
pdm run check-installation
```

The flake pins Python 3.12, uv, PDM, Pyright, Ruff, direnv, nix-direnv, and
the Hugging Face CLI (`hf`). It also supplies the CUDA toolkit and build tools.
Both `nix develop` and direnv activate `.venv` when it exists.
`uv.lock` pins Python dependencies, including the CUDA 12.8 PyTorch wheels,
which support Blackwell GPUs such as the RTX 5070 Ti. The wheels include the
CUDA runtime libraries; Week 1 does not compile custom CUDA extensions.
A working host NVIDIA driver is required. On NixOS the shell adds
`/run/opengl-driver/lib` to the library search path.

For automatic activation, enable your shell's direnv hook once (for Bash,
`eval "$(direnv hook bash)"` in your shell configuration), then run:

```sh
direnv allow
uv sync --locked
```

Re-run `uv sync --locked` after dependency updates. Dependency installation is
explicit so entering the directory does not initiate large downloads.

## Tests

Use the same day-by-day workflow as the book:

```sh
pdm run test-refsol --week 1 --day 1 -- -q
pdm run test --week 1 --day 1 -- -k softmax
```

Run every Week 1 reference checkpoint:

```sh
python -m pytest tests_refsol/test_week_1_day_*.py -q
```

Tests cover CPU and CUDA with FP32, FP16, and BF16. CUDA cases skip when no
GPU is available. Three optional model tests use locally cached Hugging Face
weights and skip when those weights are absent. Tiny random Qwen3 models
verify attention, transformer blocks, and tied/untied output heads without
network access. Student tests fail for tasks that have not been implemented.

## Neovim completion

Start Neovim from the activated shell, for example `nix develop --command nvim`.
Configure your Python LSP client to run `pyright-langserver --stdio` (the flake
provides it). The `[tool.pyright]` section in `pyproject.toml` points at the
project's `.venv` and `src`, so completion resolves torch, transformers, numpy,
and project modules from the same environment used for tests.

For Neovim 0.11 with nvim-lspconfig:

```lua
vim.lsp.enable('pyright')
```

If your existing setup uses basedpyright, it can also use `.venv/bin/python`;
set its Python interpreter to that path. Restart the Python LSP after the
initial dependency installation. No global Neovim configuration is modified.

## Model loading and generation

Download the required Day 5 checkpoint into the project model cache:

```sh
hf download Qwen/Qwen3-0.6B --cache-dir ~/.cache/tiny-llm
# Without automatic direnv activation:
nix develop --command hf download Qwen/Qwen3-0.6B --cache-dir ~/.cache/tiny-llm
```

Downloads are explicit; entering the development shell does not download models.
The Nix shell sets `HF_HUB_CACHE=~/.cache/tiny-llm` so the CLI and Python
model loader use the same project cache. Day 5 tests explicitly use this path. Optional
larger checkpoints are `Qwen/Qwen3-1.7B` and `Qwen/Qwen3-4B`.

Use original Hugging Face Qwen3 checkpoints, such as `Qwen/Qwen3-0.6B`, rather
than MLX quantized checkpoints. `Qwen3ModelWeek1(hf_model)` reads configuration
from `hf_model.config` and weights from PyTorch modules' `.weight` tensors.
No dequantization is needed. `torch.Tensor` replaces `mx.array`; common API
changes are `.to(dtype)`, `.permute(...)`, `dim=`, and `keepdim=`. Tensors and
weights must be on the same device. Model execution is eager, so `mx.eval`
has no replacement. CUDA timing requires `torch.cuda.synchronize()`.

```sh
pdm run main-week1 --solution ref --max-tokens 64 --prompt 'What is attention?'
# Equivalent:
pdm run main --loader week1 --solution ref --max-tokens 64
```

The CLI downloads the requested model if necessary. Add `--device cpu` for
CPU execution. `simple_generate` uses Transformers' `TextStreamer` and accepts
a Hugging Face tokenizer; sampling returns `torch.long` token IDs.

## Week 2 native kernels

Week 2 Day 1 runs without native extensions. Days 2–5 require BF16 tensor-core
support (compute capability 8.0+) and the supplied CUDA toolkit, host C++ compiler,
Ninja, and PyTorch development headers. The tested shell builds with CUDA 13
and the pinned PyTorch CUDA 12.8 wheels. This combination was verified on the
local RTX 5070 Ti; other toolkit/driver combinations need their own validation.
No Xcode, Metal, MLX or Triton dependency is needed.

```sh
nix develop --command pdm run build-ext-ref
nix develop --command pdm run build-ext
nix develop --command pdm run build-ext-test
nix develop --command pdm run test-refsol --week 2 --day 1 -- -q
# Repeat reference checks for Days 2–5 after the native build.
```

`src/cuda/bindings.cpp` exposes separate learner/reference implementations
from `student.cu` and `reference.cu`. Learner stubs raise errors until implemented.
`build-ext-test` checks that their bindings load, without claiming unfinished
kernels are correct. Python wrappers validate shapes, dtypes and devices.
Native launches use PyTorch's current CUDA stream and device guard.

Build outputs live under
`${XDG_CACHE_HOME:-$HOME/.cache}/dev-scratch/tiny-llm/cuda-week2/reference/`
and the sibling `student/` directory. `MAX_JOBS` defaults to 1. PyTorch selects
the visible GPU architecture; `TORCH_CUDA_ARCH_LIST` can select another supported
target explicitly. To clean/rebuild, first verify no process uses the relevant
build directory, then remove only that directory and rerun its build command.
Keep source files and diagnostic reports outside this disposable cache.

Day 2 packs Hugging Face BF16 weights into int32 W4 words with BF16 scale/bias
per group of 128. The reference custom operators unpack inside CUDA; only the
explicit readable PyTorch control dequantizes full matrices. Day 3 uses WMMA
16x16 tensor-core tiles. Day 4 adds RMSNorm, RoPE and SwiGLU kernels. Day 5 uses
BQ4/BK16 shared-memory online-softmax attention with BF16/D128, GQA and masks.
Short-query decode retains PyTorch attention in teaching `tiled-prefill`.
Reference `selected` additionally uses stride-aware native online decode,
256-token split partitions (with stable merging and zero fully masked rows),
long-context GQA4 tile sharing, and guarded explicit Q/K RMSNorm–RoPE fusion.
The fusion acts on raw activations because PyTorch tensors are already materialized;
`_qk_norm_rope4` on normalized tensors preserves the ordinary RoPE fallback.
`use_native_online_decode=False` and `use_fused_qk_norm_rope4=False` disable
these reference controls. Upstream's MLX `submit18` flush is inapplicable:
CUDA launches eagerly; `use_decode_submit18=True` is rejected rather than
introducing a synchronization at layer 18. These educational kernels have
no guaranteed latency advantage over optimized libraries.

```sh
HF_HUB_OFFLINE=1 nix develop --command pdm run main --loader week2 \
  --solution tiny_llm_ref --week2-checkpoint selected \
  --model Qwen/Qwen3-0.6B --max-tokens 32
nix develop --command pdm run bench-week2-cuda --solution tiny_llm_ref \
  --checkpoint kv-cache --checkpoint capacity-cache --checkpoint selected \
  --prefill-tokens 128 --decode-tokens 16 --prefill-logits all --repeats 3
```

The CUDA benchmark uses cached weights, warmup, fresh request caches and explicit
CUDA synchronization. It also records full request wall time and per-request
PyTorch CUDA allocator peak allocated bytes, resetting peaks after warmup and
before each request. The scope includes resident source HF and active model
tensors; it is not incremental memory, reserved bytes, or physical GPU usage. `--profile /tmp/week2.json` exports per-checkpoint traces.
`all` is the matched full-logit comparison with Week 1; `last` reduces Week 2
prefill projection work and should be reported separately. Results are JSON lines
on stdout; redirect to a durable report when needed. `pdm run profile-week2-kernels` now runs upstream's public-operator replay
contract on CUDA, with synchronized group medians, cache counters, workload and
source identities, and optional decision records. Reference `selected` reports
incomplete public replay coverage because its extra scheduling is not represented
by those teaching operators. See [the profiling chapter](book/src/week2-profile-guided.md).
Historical MLX capture/private Apple decoder scripts remain separate;
`source-costs-week2-metal` is an optional macOS-only tool.

## Week 3 CUDA serving & paged attention

Week 3 runs on the same PyTorch/CUDA runtime and native extension (`src/cuda/bindings.cpp`, `src/cuda/reference.cu`, `src/cuda/student.cu`):

- **Days 1–2**: Continuous batching (`BatchingKvCache`, `Request`, `batch_generate`) and chunked prefill on PyTorch/CUDA with Hugging Face tokenizers.
- **Day 3**: Paged KV cache (`TinyKvPagedPool`, `TinyKvPagedCache`, `PagedKvMetadata`) backed by `paged_cache_update_cuda` (in-place slice write into `[P, Hkv, page_size, D]` float32/BF16 page storage).
- **Days 4–5**: Direct and tiled FlashAttention paged attention (`paged_attention_cuda`) behind the `paged_attention` boundary:
  - Generic/decode path (`paged_attention_direct_kernel`): warp online-softmax over arbitrary `D` in FP32/BF16 with GQA and bottom-right causal masking.
  - Long-prefill path (`paged_attention_tiled_bf16_d128_kernel`): BQ4/BK16 shared-memory tile staging across logical page boundaries for BF16 `D=128`, `L>=9`.
- **Days 6–7 (Optional)**: Grouped W4 MoE (`grouped_expert_linear`, `route_topk`, `Moe`, supporting 3D packed `[E, N, K/8]` expert weights) and speculative decoding (`speculative_generate` with cache `rewind`).

```sh
nix develop --command pdm run test-refsol --week 3 --day 1 -- -q
# Repeat for Days 2–7.
nix develop --command pdm run main --loader week3 --solution ref \
  --model Qwen/Qwen3-0.6B --max-seq-len 64 --batch-size 1
```

The original Apple CLI remains in `scripts/main_mlx.py` for Week 4. Preserved `*_mlx.py` modules support that historical backend.
