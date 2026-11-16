from typing import Any

import torch
from tiny_llm_cuda.packing import pack_weights


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
    pass


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
    pass


def dequantize_weights(
    weight: torch.Tensor,
    scales: torch.Tensor,
    biases: torch.Tensor | None,
    group_size: int,
    bits: int,
) -> torch.Tensor:
    pass


def quantized_matvec_custom(
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int,
    bits: int,
    a: torch.Tensor,
    b: torch.Tensor,
    transpose_b: bool = False,
) -> torch.Tensor:
    pass


def quantized_matmul_vanilla(
    scales: torch.Tensor,
    biases: torch.Tensor,
    group_size: int,
    bits: int,
    a: torch.Tensor,
    b: torch.Tensor,
    transpose_b: bool = False,
) -> torch.Tensor:
    pass


def quantized_linear(
    x: torch.Tensor,
    w: QuantizedWeights,
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    pass
