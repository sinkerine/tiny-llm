"""Week 2 Day 2: selected-row W4 embedding exercise."""

from .quantize import QuantizedWeights, dequantize_weights, quantized_linear


class QuantizedEmbedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: QuantizedWeights):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight

    def __call__(self, x):
        # TODO: dequantize only the rows selected by token IDs.
        raise NotImplementedError("Week 2 Day 2: selected-row embedding")

    def as_linear(self, x):
        # TODO: project hidden vectors through the packed embedding weights.
        raise NotImplementedError("Week 2 Day 2: packed vocabulary projection")
