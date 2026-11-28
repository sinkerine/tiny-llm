"""PyTorch/CUDA course runtime for Weeks 1, 2, and 3."""

from .attention import (
    scaled_dot_product_attention_simple,
    scaled_dot_product_attention_grouped,
    causal_mask,
    paged_attention,
    SimpleMultiHeadAttention,
)
from .basics import softmax, linear, silu
from .embedding import Embedding
from .layer_norm import RMSNorm
from .positional_encoding import RoPE
from .qwen3_week1 import Qwen3ModelWeek1
from .generate import simple_generate, speculative_generate
from .sampler import make_sampler
from .models import dispatch_model, dispatch_week3_batch_model
from .qwen3_week2 import Qwen3ModelWeek2, WEEK2_CHECKPOINTS
from .kv_cache import BatchingKvCache, TinyKvCache, TinyKvFullCache
from .paged_kv_cache import PagedKvMetadata, TinyKvPagedCache, TinyKvPagedPool
from .batch import Request, batch_generate
from .moe import Moe, grouped_expert_linear, route_topk
from .qwen3_week3 import Qwen3ModelWeek3, is_qwen3_moe_sparse_layer
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
