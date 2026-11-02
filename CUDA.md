# Week 1 on CUDA

Week 1 uses PyTorch tensors on Linux with an NVIDIA GPU. Student tasks in
`src/tiny_llm` remain exercises; completed implementations are in
`src/tiny_llm_ref`. Weeks 2–4 still contain MLX/Metal code and are outside this
CUDA port. They are not imported by the Week 1 packages.

## Environment

```sh
nix develop
uv sync --locked
pdm run check-installation
```

The flake pins Python 3.12, uv, PDM, Pyright, Ruff, direnv, and nix-direnv.
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

The original CLI for later weeks is retained in `scripts/main_mlx.py` and is
selected by `main.py --loader week2` or `--loader week3`. Those weeks require
the original Apple/MLX dependencies and extensions in a separate environment.
