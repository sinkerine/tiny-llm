"""PyTorch/CUDA course runtime for Weeks 1 and 2."""

from .attention import (
    scaled_dot_product_attention_simple,
    scaled_dot_product_attention_grouped,
    causal_mask,
    SimpleMultiHeadAttention,
)
from .basics import softmax, linear, silu
from .embedding import Embedding
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from .qwen3_week1 import Qwen3ModelWeek1
from .generate import simple_generate
from .sampler import make_sampler
from .models import dispatch_model
from .qwen3_week2 import Qwen3ModelWeek2, WEEK2_CHECKPOINTS
from .kv_cache import TinyKvCache, TinyKvFullCache
from .quantized_embedding import QuantizedEmbedding
from .quantize import (
    QuantizedWeights,
    dequantize_weights,
    quantized_linear,
    quantized_matmul,
    quantized_matmul_vanilla,
    quantized_matvec_custom,
)
from .week2_kernels import FastRMSNorm, FastRoPE, swiglu, dense_prefill_attention_mma
from .generate_week2 import cached_generate
