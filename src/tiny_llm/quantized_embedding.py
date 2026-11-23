"""Week 2 Day 2: selected-row W4 embedding exercise."""

from .quantize import QuantizedWeights, dequantize_weights, quantized_linear


class QuantizedEmbedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: QuantizedWeights):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x):
        idx = x.long()
        weight = self.weight.weight[idx]
        scales = self.weight.scales[idx]
        biases = self.weight.biases[idx] if self.weight.biases is not None else None
        dense_weight = dequantize_weights(
            weight, scales, biases, self.weight.group_size, bits=self.weight.bits
        )
        return dense_weight

    def as_linear(self, x):
        return quantized_linear(x, self.weight)
