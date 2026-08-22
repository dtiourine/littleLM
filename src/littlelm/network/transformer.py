import json
from pathlib import Path

import numpy as np

from littlelm.backend import to_cpu, xp

from littlelm.loss import cross_entropy
from littlelm.network.components import Embedding, MLP, MultiheadAttention, LayerNorm
from littlelm.tensor import Tensor


class TransformerBlock:
    def __init__(self, d_model: int, n_heads: int):
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiheadAttention(d_model, n_heads)
        self.ln2 = LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.attn.forward(self.ln1.forward(x))
        x = x + self.mlp.forward(self.ln2.forward(x))

        return x

    def parameters(self):
        return (
            self.ln1.parameters()
            + self.attn.parameters()
            + self.ln2.parameters()
            + self.mlp.parameters()
        )

    def _children(self):
        return [
            ("ln1", self.ln1),
            ("attn", self.attn),
            ("ln2", self.ln2),
            ("mlp", self.mlp),
        ]

    def state_dict(self):
        out = {}
        for name, child in self._children():
            for k, v in child.state_dict().items():
                out[f"{name}.{k}"] = v
        return out

    def load_state_dict(self, state):
        for name, child in self._children():
            prefix = f"{name}."
            sub = {
                k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)
            }
            child.load_state_dict(sub)


class LittleLM:
    def __init__(self, d_model: int, vocab_size: int, n_heads: int, n_layers: int):
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.n_layers = n_layers

        self.embedding = Embedding(vocab_size, d_model)
        self.blocks = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        self.final_ln = LayerNorm(d_model)
        self.lm_head = Tensor(xp.random.randn(d_model, vocab_size) * 0.01)

    def forward(self, x, targets=None):
        x = self.embedding.forward(x)

        for block in self.blocks:
            x = block.forward(x)
        x = self.final_ln.forward(x)

        if targets is None:
            # Inference
            last = x[:, -1:, :]
            logits = last @ self.lm_head
            return logits, None
        else:
            # Training
            logits = x @ self.lm_head
            loss = cross_entropy(logits, targets)
            return logits, loss

    def parameters(self):
        params = self.embedding.parameters()
        for block in self.blocks:
            params += block.parameters()
        params += self.final_ln.parameters()
        params.append(self.lm_head)
        return params

    def config(self):
        return {
            "d_model": self.d_model,
            "vocab_size": self.vocab_size,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
        }

    def state_dict(self):
        out = {}
        for k, v in self.embedding.state_dict().items():
            out[f"embedding.{k}"] = v
        for i, block in enumerate(self.blocks):
            for k, v in block.state_dict().items():
                out[f"blocks.{i}.{k}"] = v
        for k, v in self.final_ln.state_dict().items():
            out[f"final_ln.{k}"] = v
        out["lm_head"] = self.lm_head.data
        return out

    def load_state_dict(self, state):
        def strip(prefix):
            return {
                k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)
            }

        self.embedding.load_state_dict(strip("embedding."))
        for i, block in enumerate(self.blocks):
            block.load_state_dict(strip(f"blocks.{i}."))
        self.final_ln.load_state_dict(strip("final_ln."))

        assert (
            self.lm_head.data.shape == state["lm_head"].shape
        ), f"lm_head shape mismatch: {self.lm_head.data.shape} vs {state['lm_head'].shape}"
        self.lm_head.data = state["lm_head"]

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "config.json", "w") as f:
            json.dump(self.config(), f, indent=2)

        state = {key: to_cpu(value) for key, value in self.state_dict().items()}
        np.savez(path / "weights.npz", **state)

    @classmethod
    def from_pretrained(cls, path):
        path = Path(path)

        with open(path / "config.json") as f:
            config = json.load(f)

        model = cls(**config)

        with np.load(path / "weights.npz") as weights:
            state = {k: weights[k] for k in weights.files}
        model.load_state_dict(state)

        return model
