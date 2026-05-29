import numpy as np

from tensors.tensor import Tensor


class MLP:
    def __init__(self, d_model: int):
        self.W_1 = Tensor(np.random.randn(d_model, d_model * 4) * 0.01)
        self.b_1 = Tensor(np.zeros(d_model * 4))
        self.W_2 = Tensor(np.random.randn(d_model * 4, d_model) * 0.01)
        self.b_2 = Tensor(np.zeros(d_model))

    def forward(self, x: Tensor):
        z1 = x @ self.W_1 + self.b_1
        z1 = z1.relu()
        z2 = z1 @ self.W_2 + self.b_2
        return z2

    def parameters(self):
        return [self.W_1, self.b_1, self.W_2, self.b_2]
