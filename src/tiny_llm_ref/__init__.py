"""Week 1 PyTorch/CUDA exercises and layers.

Later weeks still target MLX/Metal and are not imported by the CUDA runtime.
"""

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
