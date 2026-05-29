from littlelm.tensor import Tensor

import numpy as np


class SGD:
    def __init__(self, params: list[Tensor], lr: float = 0.01):
        self.params = params
        self.lr = lr

    def step(self):
        for param in self.params:
            param.data -= self.lr * param.grad

    def zero_grad(self):
        for param in self.params:
            param.grad = np.zeros_like(param.data)
