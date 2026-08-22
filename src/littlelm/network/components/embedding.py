from littlelm.backend import xp
from littlelm.tensor import Tensor


class Embedding:
    def __init__(self, vocab_size, embed_dim):
        self.W = Tensor(xp.random.randn(vocab_size, embed_dim) * 0.01)

    def forward(self, token_ids):
        return self.W[token_ids]

    def parameters(self):
        return [self.W]

    def state_dict(self):
        return {"W": self.W.data}

    def load_state_dict(self, state):
        assert self.W.data.shape == state["W"].shape, (
            f"Embedding.W shape mismatch: {self.W.data.shape} vs {state['W'].shape}"
        )
        self.W.data = state["W"]
