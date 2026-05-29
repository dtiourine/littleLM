from pathlib import Path
import numpy as np
from src.tokenizer import tokenizer


def tokenize(text_path: Path, dest: Path):
    if dest.exists():
        print(f"Already downloaded: {dest} ({dest.stat().st_size:,} bytes)")
        return

    with open(text_path, "r") as f:
        text = f.read()

    print(f"Tokenizing {text_path} → {dest}")
    tokenizer.train(text)
    tokenized = tokenizer.encode(text)

    dest.parent.mkdir(parents=True, exist_ok=True)
    np.save(dest, tokenized)
    print(f"Saved {dest} ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parent.parent / "data"
    text_path = data_path / "tinyshakespeare.txt"
    dest = data_path / "data.npy"
    tokenize(text_path, dest)
