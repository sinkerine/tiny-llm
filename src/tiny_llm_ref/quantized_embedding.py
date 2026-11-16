"""Dequantize selected embedding rows only; project through packed W4."""

from .quantize import dequantize_weights, quantized_linear


class QuantizedEmbedding:
    def __init__(self, vocab_size, embedding_dim, weight):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x):
        w = self.weight
        return dequantize_weights(
            w.weight[x.long()],
            w.scales[x.long()],
            w.biases[x.long()],
            w.group_size,
            w.bits,
        )

    def as_linear(self, x):
        return quantized_linear(x, self.weight)
