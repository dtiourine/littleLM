import numpy as np
from tensors.tensor import Tensor


class MultiheadAttention:
    def __init__(self, d_model: int, n_heads: int = 8):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = Tensor(np.random.randn(d_model, d_model) * 0.01)
        self.W_k = Tensor(np.random.randn(d_model, d_model) * 0.01)
        self.W_v = Tensor(np.random.randn(d_model, d_model) * 0.01)

        self.b_q = Tensor(np.zeros(d_model))
        self.b_k = Tensor(np.zeros(d_model))
        self.b_v = Tensor(np.zeros(d_model))

        self.W_o = Tensor(np.random.randn(d_model, d_model) * 0.01)
        self.b_o = Tensor(np.zeros(d_model))

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        Q = x @ self.W_q + self.b_q
        K = x @ self.W_k + self.b_k
        V = x @ self.W_v + self.b_v

        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k)

        Q = Q.transpose(axes=(0, 2, 1, 3))
        K = K.transpose(axes=(0, 2, 1, 3))
        V = V.transpose(axes=(0, 2, 1, 3))

        attn_scores = (Q @ K.transpose(axes=(0, 1, 3, 2))) / np.sqrt(self.d_k)

        mask = np.tril(np.ones((seq_len, seq_len)))
        mask_value = (1 - mask) * -1e9
        attn_scores = attn_scores + Tensor(mask_value)

        attn_weights = attn_scores.softmax(axis=-1)
        attended = attn_weights @ V

        attended = attended.transpose(axes=(0, 2, 1, 3))
        attended = attended.reshape(batch_size, seq_len, self.n_heads * self.d_k)

        output = attended @ self.W_o + self.b_o
        return output

    def parameters(self):
        return [
            self.W_q,
            self.b_q,
            self.W_k,
            self.b_k,
            self.W_v,
            self.b_v,
            self.W_o,
            self.b_o,
        ]
