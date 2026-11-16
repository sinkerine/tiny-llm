"""Group-affine W4 storage and native CUDA projection dispatch."""

from dataclasses import dataclass
import torch
from . import cuda_ops


from tiny_llm_cuda.packing import pack_weights


@dataclass
class QuantizedWeights:
    scales: torch.Tensor
    biases: torch.Tensor
    group_size: int
    bits: int
    weight: torch.Tensor
    use_simdgroup_matmul: bool = False
    use_simdgroup_matvec: bool = True
    use_split_k_matmul: bool = False
    use_torch_quantized_linear: bool = False

    @staticmethod
    def from_layer(layer, **kwargs):
        packed, scales, biases = pack_weights(layer.weight)
        return QuantizedWeights(scales, biases, 128, 4, packed, **kwargs)


def dequantize_weights(weight, scales, biases, group_size, bits):
    if (
        bits != 4
        or group_size <= 0
        or weight.shape[-1] * 8 != scales.shape[-1] * group_size
    ):
        raise ValueError("incompatible packed W4 layout")
    shifts = torch.arange(0, 32, 4, device=weight.device)
    q = (
        ((weight.long()[..., None] >> shifts) & 15)
        .reshape(*weight.shape[:-1], -1)
        .float()
    )
    scale = scales.repeat_interleave(group_size, -1).float()
    result = q * scale
    if biases is not None:
        result = result + biases.repeat_interleave(group_size, -1).float()
    return result.to(scales.dtype)


def dequantize_linear(layer):
    return layer.weight.detach()


def torch_quantized_linear(x, w, bias=None):
    dense = dequantize_weights(w.weight, w.scales, w.biases, w.group_size, w.bits)
    result = torch.nn.functional.linear(x, dense)
    return result if bias is None else result + bias


def quantized_matmul(
    scales,
    biases,
    group_size,
    bits,
    a,
    b,
    transpose_b=False,
    use_simdgroup=False,
    use_split_k=False,
):
    if bits != 4 or not transpose_b or use_split_k:
        raise ValueError(
            "CUDA Week 2 supports W4, transpose_b=True, and no historical Split-K"
        )
    return cuda_ops.w4(
        a.contiguous(),
        b.contiguous(),
        scales.contiguous(),
        biases.contiguous(),
        group_size,
        2 if use_simdgroup else 0,
    )


def quantized_matvec_custom(scales, biases, group_size, bits, a, b, transpose_b=False):
    if a.numel() // a.shape[-1] > 8:
        raise ValueError("quantized_matvec_custom supports at most 8 rows")
    if bits != 4 or not transpose_b:
        raise ValueError("CUDA matvec requires W4 and transpose_b=True")
    return cuda_ops.w4(
        a.contiguous(),
        b.contiguous(),
        scales.contiguous(),
        biases.contiguous(),
        group_size,
        1,
    )


def quantized_matmul_vanilla(scales, biases, group_size, bits, a, b, transpose_b=False):
    return quantized_matmul(scales, biases, group_size, bits, a, b, transpose_b)


def quantized_linear(x, w, bias=None):
    if w.use_torch_quantized_linear:
        return torch_quantized_linear(x, w, bias)
    rows = x.numel() // x.shape[-1]
    if rows <= 8 and w.use_simdgroup_matvec:
        result = quantized_matvec_custom(
            w.scales, w.biases, w.group_size, w.bits, x, w.weight, True
        )
    else:
        result = quantized_matmul(
            w.scales,
            w.biases,
            w.group_size,
            w.bits,
            x,
            w.weight,
            True,
            w.use_simdgroup_matmul,
            w.use_split_k_matmul,
        )
    return result if bias is None else result + bias
