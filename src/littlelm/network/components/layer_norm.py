import numpy as np
from littlelm.tensor import Tensor


class LayerNorm:
    def __init__(self, d_model, eps=1e-5):
        self.gamma = Tensor(np.ones(d_model))
        self.beta = Tensor(np.zeros(d_model))
        self.eps = eps

    def forward(self, x: Tensor):
        mean = x.mean(axis=-1, keepdims=True)

        centered = x - mean
        var = (centered**2).mean(axis=-1, keepdims=True)

        normalized = centered / (var + self.eps).sqrt()
        return normalized * self.gamma + self.beta

    def parameters(self):
        return [self.gamma, self.beta]

    def state_dict(self):
        return {"gamma": self.gamma.data, "beta": self.beta.data}

    def load_state_dict(self, state):
        for name, tensor in [("gamma", self.gamma), ("beta", self.beta)]:
            assert tensor.data.shape == state[name].shape, (
                f"LayerNorm.{name} shape mismatch: {tensor.data.shape} vs {state[name].shape}"
            )
            tensor.data = state[name]
