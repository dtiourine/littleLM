import numpy as np
from tensor import Tensor, stack


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

        Q, K = self._encode_position(Q, K, seq_len, batch_size)

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

    def state_dict(self):
        return {
            "W_q": self.W_q.data,
            "b_q": self.b_q.data,
            "W_k": self.W_k.data,
            "b_k": self.b_k.data,
            "W_v": self.W_v.data,
            "b_v": self.b_v.data,
            "W_o": self.W_o.data,
            "b_o": self.b_o.data,
        }

    def load_state_dict(self, state):
        for name, tensor in [
            ("W_q", self.W_q),
            ("b_q", self.b_q),
            ("W_k", self.W_k),
            ("b_k", self.b_k),
            ("W_v", self.W_v),
            ("b_v", self.b_v),
            ("W_o", self.W_o),
            ("b_o", self.b_o),
        ]:
            assert tensor.data.shape == state[name].shape, (
                f"MultiheadAttention.{name} shape mismatch: "
                f"{tensor.data.shape} vs {state[name].shape}"
            )
            tensor.data = state[name]

    # RoPE
    def _encode_position(self, Q, K, seq_len, batch_size):
        k = np.arange(self.d_k // 2)[None, :]
        angles = 10000 ** (-2 * k / self.d_k)

        n = np.arange(seq_len)[:, None]
        m_theta = n * angles

        cos_half = Tensor(np.cos(m_theta)[None, None, :, :])
        sin_half = Tensor(np.sin(m_theta)[None, None, :, :])

        Q_even, Q_odd = Q[:, :, :, 0::2], Q[:, :, :, 1::2]
        K_even, K_odd = K[:, :, :, 0::2], K[:, :, :, 1::2]

        Q_even_rot = Q_even * cos_half - Q_odd * sin_half
        Q_odd_rot = Q_even * sin_half + Q_odd * cos_half

        K_even_rot = K_even * cos_half - K_odd * sin_half
        K_odd_rot = K_even * sin_half + K_odd * cos_half

        Q_stacked = stack([Q_even_rot, Q_odd_rot], axis=-1)
        K_stacked = stack([K_even_rot, K_odd_rot], axis=-1)

        Q = Q_stacked.reshape(batch_size, self.n_heads, seq_len, self.d_k)
        K = K_stacked.reshape(batch_size, self.n_heads, seq_len, self.d_k)

        return Q, K
