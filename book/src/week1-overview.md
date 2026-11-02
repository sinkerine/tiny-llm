# Week 1: From Matmul to Text

> This branch uses PyTorch/CUDA for Week 1. See [the CUDA setup guide](https://github.com/sinkerine/tiny-llm/blob/week1/CUDA.md) for tensor API changes, environment setup, and tests. Student tasks remain unfinished. Historical MLX links below are supplemental references.

This week, we will start with basic array and matrix operations and use them to turn Qwen3 model parameters into a model
that generates text. We will implement the neural network layers used by Qwen3 with PyTorch tensor operations on CUDA.

We will use `Qwen/Qwen3-0.6B`. The course model uses BF16 weights and
activations by default, so start with the 0.6B model before trying larger Qwen3
models. The required model path runs on the GPU. Small operator fixtures may
use a different dtype as a readable correctness reference; they do not define
the model-storage dtype.

Numerically sensitive operations may promote arithmetic to FP32 and cast the
result back to BF16. Week 1 favors readable array expressions, even when that
means materializing an FP32 intermediate. Week 2 replaces those full-tensor
promotions with kernels that keep model-sized storage in BF16 and accumulate in
FP32 registers.

## What We Will Cover

- Attention, multi-head attention, grouped-query attention, and multi-query attention
- Positional encodings and RoPE
- Using `torch.nn.functional.rms_norm` for Qwen3's per-head Q/K normalization, then implementing RMSNorm ourselves
- Implementing the MLP, combining the attention components, and building the complete Transformer model
- Loading Qwen3 model parameters and generating text

## What We Will Not Cover

To make the journey as interesting as possible, we will skip a few things for now:

- Quantization internals. Week 1 loads the original Hugging Face weights directly as BF16 tensors.
- Low-level implementations of operations such as softmax, exponentiation, and logarithms. These operations are simple
  enough that using the PyTorch versions does not detract from the learning objectives.
- Tokenization. We use the `transformers` tokenizer rather than implementing one from scratch.
- Decoding model-weight files. We use `transformers` to load the model, then transfer its weights into our layer implementations.

## Basic Matrix APIs

PyTorch's tensor API is designed to be familiar to NumPy users. If you are new to array programming, start with
[NumPy: the absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html).

You can also refer to the [MLX Operations API](https://ml-explore.github.io/mlx/build/html/python/ops.html#operations)
for more details.

## Qwen3 Models

You can run Qwen3 with MLX or vLLM. The readings below provide context for what we will build. By the end of the week,
you will be able to use Qwen3 as a causal language model to generate text.

Reference implementations of Qwen3 are available in Hugging Face Transformers, vLLM, and mlx-lm. Use them to explore
the model's internals and compare them with this week's implementation.

**📚 Readings**

- [Qwen3: Think Deeper, Act Faster](https://qwenlm.github.io/blog/qwen3/)
- [Hugging Face Transformers — Qwen3](https://github.com/huggingface/transformers/tree/main/src/transformers/models/qwen3)
- [vLLM Qwen3](https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3.py)
- [mlx-lm Qwen3](https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/qwen3.py)
- [Qwen3 Technical Report](https://arxiv.org/abs/2505.09388)

{{#include copyright.md}}
