# Setting Up the Environment

This checkout uses Nix flakes and direnv to provide a reproducible development
shell on NixOS. The target accelerator is an NVIDIA GPU with CUDA, not Apple
silicon. The project uses PDM for Python dependency management inside the Nix
shell.

Before starting, configure the NVIDIA driver in NixOS and verify it with:

```bash
nvidia-smi
```

Your NixOS configuration should enable flakes and direnv. A typical setup
includes `programs.direnv.enable = true;` and
`programs.direnv.nix-direnv.enable = true;`.

## Clone the Repository

```bash
git clone https://github.com/skyzh/tiny-llm
```

The repository is organized as follows:

```
src/tiny_llm/ -- your implementation
src/tiny_llm_ref/ -- the reference implementation
tests/ -- unit tests for your implementation
tests_refsol/ -- unit tests for the reference implementation
book/ -- the book source
```

Reference implementations are available if you get stuck during the course.

## Install Dependencies

```bash
cd tiny-llm
# Allow direnv to load flake.nix. This enters the CUDA development shell.
direnv allow
# This creates .venv and installs all Python dependencies.
pdm install -v
# Activate the newly-created .venv in the direnv shell.
direnv reload
```

## Check the Installation

```bash
pdm run check-installation
```

## Build the Native Extensions Before Tests

Week 1 test collection reaches the learner and reference native extensions
through package imports, even though the Day 1 attention exercise is in
Python. Before running either test command on a fresh checkout, prepare full
Xcode, its Metal compiler, and CMake 3.27 or newer; the
[toolchain steps in Week 1 Day 7](./week1-07-sampling-prepare.md#task-2-prepare-for-week-2)
give the installation checks. Then, from the repository root, build both
extensions:

```bash
pdm run build-ext
pdm run build-ext-ref
```

The reference build supplies its native import; it does not implement the
learner TODOs in `src/tiny_llm`.

Check the completed Day 1 reference exercise before working on your starter:

```bash
pdm run test-refsol --week 1 --day 1 -- -k task_1
# The reference solution should pass all Week 1 tests.
pdm run test-refsol -- -- -k week_1
```

This command runs a small PyTorch CUDA matmul smoke check. Course tests and
model-loading instructions still target the legacy MLX/Metal code; defer them
until the CUDA rewrite is complete.

## Run Unit Tests (after CUDA rewrite)

Your code is in `src/tiny_llm`. You can run the unit tests with:

```bash
pdm run test
```

## Download the Model Parameters (legacy, deferred)

The original course uses official 4-bit Qwen3 MLX model files. These files and
commands belong to the legacy MLX implementation; skip this section while the
CUDA model loader is being rewritten. No MLX package is installed by this
environment.

The legacy default model is `Qwen/Qwen3-0.6B-MLX-4bit`, which is small enough
for the original dequantized Python implementation in Week 1.

Follow the [Hugging Face CLI guide](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli) to install the `hf`
command-line tool.

The model parameters are hosted on Hugging Face. After authenticating the CLI with your credentials, download them with:

```bash
hf auth login
hf download Qwen/Qwen3-0.6B-MLX-4bit
# Optional larger models:
hf download Qwen/Qwen3-1.7B-MLX-4bit
hf download Qwen/Qwen3-4B-MLX-4bit
```

Legacy path only (not currently supported by the CUDA environment):

```bash
pdm run main --solution ref --loader week1
```

The command should load the reference model and print generated text.

The Nix shell supplies CUDA 13.0, CMake, Ninja, and the compiler toolchain.
The checked-in Python and custom-extension implementations still target MLX
and Metal; their CUDA rewrites are pending.

{{#include copyright.md}}
