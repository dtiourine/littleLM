from pathlib import Path

import numpy as np

from littlelm.tokenizer import Tokenizer


VOCAB_SIZE = 1024


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
    data_path = Path(__file__).resolve().parent.parent / "data"
    text_path = data_path / "tinyshakespeare.txt"
    data_dest = data_path / "data.npy"
    tokenizer_dest = data_path / "tokenizer.json"
    tokenize(text_path, data_dest, tokenizer_dest, vocab_size=VOCAB_SIZE)
