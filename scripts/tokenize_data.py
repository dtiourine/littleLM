from pathlib import Path

import numpy as np

from littlelm.config import DATA_DIR, DataConfig, ModelConfig
from littlelm.tokenizer import Tokenizer


def tokenize(text_path: Path, data_dest: Path, tokenizer_dest: Path, vocab_size: int):
    if data_dest.exists() and tokenizer_dest.exists():
        print(
            f"Already tokenized: {data_dest} ({data_dest.stat().st_size:,} bytes) "
            f"and {tokenizer_dest}"
        )
        return

    with open(text_path) as f:
        text = f.read()

    print(f"Training tokenizer on {text_path} (vocab_size={vocab_size})")
    tokenizer = Tokenizer(vocab_size=vocab_size)
    tokenizer.train(text)

    print(f"Encoding {text_path} → {data_dest}")
    tokenized = tokenizer.encode(text)

    data_dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(data_dest, np.array(tokenized, dtype=np.int32))
    print(f"Saved {data_dest} ({data_dest.stat().st_size:,} bytes, {len(tokenized):,} tokens)")

    tokenizer.save(tokenizer_dest)
    print(f"Saved tokenizer to {tokenizer_dest}")


if __name__ == "__main__":
    data_cfg = DataConfig()
    model_cfg = ModelConfig()
    tokenize(
        text_path=DATA_DIR / data_cfg.text_file,
        data_dest=DATA_DIR / data_cfg.tokens_file,
        tokenizer_dest=DATA_DIR / data_cfg.tokenizer_file,
        vocab_size=model_cfg.vocab_size,
    )
