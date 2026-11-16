# Setting Up the Environment

This checkout runs Weeks 1–2 on Linux with PyTorch and NVIDIA CUDA. Week 2
adds native C++/CUDA extensions. Weeks 3–4 remain the historical MLX/Metal
route. See [CUDA.md](../../CUDA.md) for detailed environment notes.

## Clone the Repository

```bash
git clone https://github.com/sinkerine/tiny-llm
cd tiny-llm
```

Student implementations live in `src/tiny_llm`, separate references in
`src/tiny_llm_ref`, learner tests in `tests`, and reference tests in
`tests_refsol`. Book sources live in `book` and native CUDA code in `src/cuda`.
Reference code provides controls; do not fill your learner exercises from it.

## Install Dependencies

Configure a working host NVIDIA driver and enable Nix flakes. Verify the
GPU, enter the development shell and install the locked Python environment:

```bash
nvidia-smi
nix develop
uv sync --locked
pdm run check-installation
```

The shell supplies Python 3.12, PDM, uv, PyTorch-compatible CUDA tooling,
CMake, Ninja and the compiler. Optionally use direnv with `direnv allow`;
it activates the same environment. No Xcode, Metal or MLX package is required
for the active Weeks 1–2 route.

## Run Unit Tests

Start with a day's focused task, then its complete suite:

```bash
pdm run test-refsol --week 1 --day 1 -- -k task_1 -q
pdm run test --week 1 --day 1 -- -k task_1 -q
```

Weeks 1 and Week 2 Day 1 need no native build for collection. Before Week 2
Day 2, compile both separate extensions:

```bash
pdm run build-ext
pdm run build-ext-ref
pdm run build-ext-test
```

The build smoke check confirms bindings load; learner kernels remain
unfinished and raise errors until implemented. Builds use external scratch
paths documented in [CUDA.md](../../CUDA.md). Rebuild after native edits.
Tests use deterministic tiny Hugging Face models without downloads. Native
checks require a CUDA GPU with BF16 tensor-core support; a skip is not a pass.

## Download the Model Parameters

Weeks 1–2 load the same original Hugging Face model. Download explicitly to
the project cache; the Nix shell configures the loader to use it:

```bash
hf download Qwen/Qwen3-0.6B --cache-dir ~/.cache/tiny-llm
pdm run main --solution tiny_llm_ref --loader week1 --max-tokens 64
# After building the reference CUDA extension:
pdm run main --solution tiny_llm_ref --loader week2 \
  --week2-checkpoint selected --max-tokens 64
```

Week 2 packs BF16 checkpoint weights into its teaching W4 format locally;
MLX-quantized checkpoints are not compatible with this loader. Larger models
are optional and must use the same cache directory. Network downloads never
happen just from entering the development shell.

{{#include copyright.md}}
