from littlelm.backend import xp

from littlelm.tensor import Tensor


class MLP:
    def __init__(self, d_model: int):
        self.W_1 = Tensor(xp.random.randn(d_model, d_model * 4) * 0.01)
        self.b_1 = Tensor(xp.zeros(d_model * 4))
        self.W_2 = Tensor(xp.random.randn(d_model * 4, d_model) * 0.01)
        self.b_2 = Tensor(xp.zeros(d_model))

    def forward(self, x: Tensor):
        z1 = x @ self.W_1 + self.b_1
        z1 = z1.relu()
        z2 = z1 @ self.W_2 + self.b_2
        return z2

    def parameters(self):
        return [self.W_1, self.b_1, self.W_2, self.b_2]

    def state_dict(self):
        return {
            "W_1": self.W_1.data,
            "b_1": self.b_1.data,
            "W_2": self.W_2.data,
            "b_2": self.b_2.data,
        }

    def load_state_dict(self, state):
        for name, tensor in [
            ("W_1", self.W_1),
            ("b_1", self.b_1),
            ("W_2", self.W_2),
            ("b_2", self.b_2),
        ]:
            assert tensor.data.shape == state[name].shape, (
                f"MLP.{name} shape mismatch: {tensor.data.shape} vs {state[name].shape}"
            )
            tensor.data = state[name]
