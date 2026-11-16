"""PyTorch/CUDA runtime bindings for upstream's public-operator replay contract."""

from dataclasses import asdict
import importlib
import torch
from .week2_profile_adapter import (
    ModelShape,
    ReplayPlan,
    PublicBindings,
    PublicReplay,
    choose_operator,
)


def make_cuda_replay(
    implementation, source, case, *, prefill_logits="last", cache_capacity=None
):
    args = source.config
    shape = ModelShape(
        args.num_hidden_layers,
        args.hidden_size,
        args.intermediate_size,
        args.num_attention_heads,
        args.num_key_value_heads,
        args.head_dim,
        args.vocab_size,
    )
    features = asdict(implementation.features[case.checkpoint])
    # Validate coverage before preparing weights or allocating fixtures.
    plan = ReplayPlan(
        case, shape, features, implementation.name, prefill_logits, cache_capacity
    )
    m = implementation.modules
    quantize, kernels = m["quantize"], m["week2_kernels"]
    device = source.device

    def weight(layer):
        if features["quantized_weights"]:
            return quantize.QuantizedWeights.from_layer(
                layer, use_simdgroup_matmul=features["simdgroup_matmul"]
            )
        return layer.weight.detach().to(torch.bfloat16)

    weights, norms = {}, {}
    norm_type = choose_operator(
        features, "fast_rms_norm", m["layer_norm"].RMSNorm, kernels.FastRMSNorm
    )
    for i, layer in enumerate(source.model.layers):
        for role, original in (
            ("q", layer.self_attn.q_proj),
            ("k", layer.self_attn.k_proj),
            ("v", layer.self_attn.v_proj),
            ("o", layer.self_attn.o_proj),
            ("gate", layer.mlp.gate_proj),
            ("up", layer.mlp.up_proj),
            ("down", layer.mlp.down_proj),
        ):
            weights[i, role] = weight(original)
        for role, width, original in (
            ("input_norm", shape.hidden, layer.input_layernorm.weight),
            ("post_norm", shape.hidden, layer.post_attention_layernorm.weight),
            ("q_norm", shape.head_dim, layer.self_attn.q_norm.weight),
            ("k_norm", shape.head_dim, layer.self_attn.k_norm.weight),
        ):
            norms[i, role] = norm_type(width, original.detach(), eps=args.rms_norm_eps)
    if len(source.model.layers) != shape.layers:
        raise ValueError("Qwen layer count disagrees with configuration")
    embedding_type = (
        importlib.import_module(
            implementation.name + ".quantized_embedding"
        ).QuantizedEmbedding
        if features["quantized_weights"]
        else m["embedding"].Embedding
    )
    embedding = embedding_type(
        shape.vocab, shape.hidden, weight(source.model.embed_tokens)
    )
    if not args.tie_word_embeddings:
        weights[None, "head"] = weight(source.lm_head)
    norms[None, "final_norm"] = norm_type(
        shape.hidden, source.model.norm.weight.detach(), eps=args.rms_norm_eps
    )
    # Match the readable model's CUDA precision rather than historical MLX arithmetic.
    if features["fast_rope"]:
        rope = kernels.FastRoPE(
            shape.head_dim, args.max_position_embeddings, args.rope_theta
        )
    else:
        # The student's Week 1 RoPE is a public operator; use its slice interface.
        rope = m["positional_encoding"].RoPE(
            shape.head_dim, args.max_position_embeddings, args.rope_theta
        )
    generator = torch.Generator(device=device).manual_seed(0)

    def array(dims, role, dtype):
        if dtype == "int32":
            return torch.zeros(dims, device=device, dtype=torch.long)
        return torch.randn(
            dims, device=device, dtype=torch.bfloat16, generator=generator
        )

    def project(layer, role, value):
        if role == "head" and args.tie_word_embeddings:
            return embedding.as_linear(value)
        op = (
            quantize.quantized_linear
            if features["quantized_weights"]
            else m["basics"].linear
        )
        result = op(value, weights[layer, role])
        if result is None:
            raise RuntimeError(f"implement public projection {role}")
        return result

    def pointwise(layer, role, *values, offset=0):
        if role == "embedding":
            result = embedding(values[0])
        elif role == "rope":
            result = rope(values[0], offset=offset)
        elif role == "swiglu":
            result = (
                kernels.swiglu(*values)
                if features["fast_swiglu"]
                else m["basics"].silu(values[0]) * values[1]
            )
        elif role == "residual":
            result = values[0] + values[1]
        else:
            result = norms[layer, role](values[0])
        if result is None:
            raise RuntimeError(f"implement public operator {role}")
        return result

    def prepare_offset(offset, batch):
        if features["fast_rope"]:
            return torch.full((batch,), offset, device=device, dtype=torch.long)
        return slice(offset, offset + plan.rows)

    def attention(q, k, v, *, scale, mask):
        if (
            features["tiled_prefill_attention"]
            and plan.rows >= 9
            and shape.head_dim == 128
        ):
            return implementation.tiled_attention(q, k, v, scale=scale, mask=mask)
        if implementation.name == "tiny_llm_ref":
            return kernels.scaled_dot_product_attention(q, k, v, scale, mask)
        return m["attention"].scaled_dot_product_attention_grouped(q, k, v, scale, mask)

    def evaluate(outputs):
        torch.cuda.synchronize(device)

    bindings = PublicBindings(
        array,
        project,
        pointwise,
        attention,
        lambda capacity: m["kv_cache"].TinyKvFullCache(capacity=capacity),
        evaluate,
        prepare_offset,
    )
    return PublicReplay(plan, bindings)
