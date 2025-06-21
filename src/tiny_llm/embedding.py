import mlx.core as mx
from .basics import linear

class Embedding:
    def __init__(self, vocab_size: int, embedding_dim: int, weight: mx.array):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.weight = weight # (vocab_size, embedding_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.weight[x] # (batch_size, sequence_length, embedding_dim)

    def as_linear(self, x: mx.array) -> mx.array:
        # Input: N.. x embedding_dim
        # Output: N.. x vocab_size
        return linear(x, self.weight) # (N.., vocab_size)