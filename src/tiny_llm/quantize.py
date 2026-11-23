from typing import Any

import torch
from tiny_llm_cuda.packing import pack_weights

from . import cuda_ops


def dequantize_linear(layer: Any) -> torch.Tensor:
    return layer.weight.detach()


class QuantizedWeights:
    def __init__(
        self,
        scales: torch.Tensor,
        biases: torch.Tensor,
        group_size: int,
        bits: int,
        weight: torch.Tensor,
        use_simdgroup_matmul: bool = False,
        use_simdgroup_matvec: bool = True,
        use_split_k_matmul: bool = False,
        use_torch_quantized_linear: bool = False,
    ):
        self.scales = scales
        self.biases = biases
        self.group_size = group_size
        self.bits = bits
        self.weight = weight
        self.use_simdgroup_matmul = use_simdgroup_matmul
        self.use_simdgroup_matvec = use_simdgroup_matvec
        self.use_split_k_matmul = use_split_k_matmul
        self.use_torch_quantized_linear = use_torch_quantized_linear

    @staticmethod
    def from_layer(layer, **kwargs):
        packed, scales, biases = pack_weights(layer.weight)
        return QuantizedWeights(scales, biases, 128, 4, packed, **kwargs)


def torch_quantized_linear(
    x: torch.Tensor,
    w: QuantizedWeights,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    dense_w = dequantize_weights(w.weight, w.scales, w.biases, w.group_size, w.bits)
    return torch.nn.functional.linear(x, dense_w, bias)


def quantized_matmul(
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int,
    bits: int,
    a: torch.Tensor,
    b: torch.Tensor,
    transpose_b: bool = False,
    use_simdgroup: bool = False,
    use_split_k: bool = False,
) -> torch.Tensor:
    if bits != 4:
        raise ValueError("only 4-bit quantized matmul is supported")
    if not transpose_b:
        raise ValueError("quantized matmul requires transpose_b=True")
    if use_split_k:
        raise ValueError("split-K quantized matmul is not supported on CUDA")
    if biases is None:
        raise ValueError("quantized matmul requires biases")
    collapsed_a = a.reshape(-1, a.shape[-1]).contiguous()
    result = cuda_ops.w4(
        collapsed_a,
        b.contiguous(),
        scales.contiguous(),
        biases.contiguous(),
        group_size,
        2 if use_simdgroup else 0,
    )
    return result.reshape(*a.shape[:-1], b.shape[0])


def dequantize_weights(
    weight: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor | None,
    group_size: int,
    bits: int,
) -> torch.Tensor:
    if bits != 4:
        raise ValueError()
    k = weight.shape[-1] * (32 // bits)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=weight.device)
    q = (weight.unsqueeze(-1) >> shifts) & 0xF
    q = q.reshape(*weight.shape[:-1], k // group_size, group_size).float()
    q *= scales.float().unsqueeze(-1)
    if biases is not None:
        q += biases.float().unsqueeze(-1)
    return q.reshape(*weight.shape[:-1], k).to(scales.dtype)


def quantized_matvec_custom(
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int,
    bits: int,
    a: torch.Tensor,
    b: torch.Tensor,
    transpose_b: bool = False,
) -> torch.Tensor:
    if bits != 4:
        raise ValueError("only 4-bit quantized matmul is supported")
    if not transpose_b:
        raise ValueError("quantized matmul requires transpose_b=True")
    if biases is None:
        raise ValueError("quantized matmul requires biases")
    collapsed_a = a.reshape(-1, a.shape[-1]).contiguous()
    if collapsed_a.shape[0] > 8:
        raise ValueError("quantized_matvec_custom supports flattened M <= 8")
    result = cuda_ops.w4(
        collapsed_a,
        b.contiguous(),
        scales.contiguous(),
        biases.contiguous(),
        group_size,
        1,
    )
    return result.reshape(*a.shape[:-1], b.shape[0])


def quantized_matmul_vanilla(
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int,
    bits: int,
    a: torch.Tensor,
    b: torch.Tensor,
    transpose_b: bool = False,
) -> torch.Tensor:
    return quantized_matmul(scales, biases, group_size, bits, a, b, transpose_b)


def quantized_linear(
    x: torch.Tensor,
    w: QuantizedWeights,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    if w.use_torch_quantized_linear:
        return torch_quantized_linear(x, w, bias)
    m = x.numel() // x.shape[-1]
    if w.use_simdgroup_matvec and m <= 8:
        out = quantized_matvec_custom(
            w.scales,
            w.biases,
            w.group_size,
            w.bits,
            x,
            w.weight,
            transpose_b=True,
        )
    else:
        out = quantized_matmul(
            w.scales,
            w.biases,
            w.group_size,
            w.bits,
            x,
            w.weight,
            transpose_b=True,
            use_simdgroup=w.use_simdgroup_matmul,
            use_split_k=w.use_split_k_matmul,
        )
    if bias is not None:
        out = out + bias
    return out
