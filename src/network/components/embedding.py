import numpy as np
from tensors.tensor import Tensor


class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.W = Tensor(np.random.randn(vocab_size, embed_dim) * 0.01)

    def forward(self, token_ids):
        return self.W[token_ids]

    def parameters(self):
        return [self.W]
