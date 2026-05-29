from network.components import Embedding, MLP, MultiheadAttention, LayerNorm
from tensors.tensor import Tensor
import numpy as np
import json


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
        return {
            "ln1": self.ln1.parameters(),
            "attn": self.attn.parameters(),
            "ln2": self.ln2.parameters(),
            "mlp": self.mlp.parameters(),
        }


class Transformer:
    def __init__(self, d_model: int, vocab_size: int, n_heads: int, n_layers: int):
        self.embedding = Embedding(vocab_size, d_model)
        self.blocks = [TransformerBlock(d_model, n_heads) for _ in range(n_layers)]
        self.final_ln = LayerNorm(d_model)
        self.lm_head = Tensor(np.random.randn(d_model, vocab_size) * 0.01)

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
        params = {"embedding": self.embedding.parameters()}
        for i, block in enumerate(self.blocks):
            params[f"block.{i}"] = block.parameters()
        params["final_ln"] = self.final_ln.parameters()
        params["lm_head"] = [self.lm_head]
        return params

    def save_model(self, path):
        params = self.parameters()

        if path.is_dir():
            path = path.joinpath("params.json")

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as outfile:
            json.dump(params, outfile, indent=4)
