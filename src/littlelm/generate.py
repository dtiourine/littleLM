import numpy as np

from littlelm.backend import xp

from littlelm.network.transformer import LittleLM
from littlelm.tokenizer import Tokenizer


def generate(
    model: LittleLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    context_len: int,
    seed: int | None = None,
) -> str:
    rng = np.random.default_rng(seed)
    ids = list(tokenizer.encode(prompt))

    for _ in range(max_new_tokens):
        context = ids[-context_len:]
        logits, _ = model.forward(xp.asarray([context]))
        next_logits = logits.data[0, -1, :] / temperature

        cutoff = xp.partition(next_logits, -top_k)[-top_k]
        next_logits = xp.where(next_logits >= cutoff, next_logits, -xp.inf)

        next_logits -= next_logits.max()
        probs = xp.exp(next_logits)
        probs /= probs.sum()
        next_id = int(rng.choice(len(probs), p=np.asarray(probs)))
        ids.append(next_id)

    return tokenizer.decode(ids)
