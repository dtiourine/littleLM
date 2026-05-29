from pathlib import Path

import requests

from littlelm.config import DATA_DIR, MODEL_DIR

HF_REPO = "dtiourine/littleLM"
HF_BASE = f"https://huggingface.co/{HF_REPO}/resolve/main"


FILES = [
    ("config.json", MODEL_DIR),
    ("weights.npz", MODEL_DIR),
    ("tokenizer.json", DATA_DIR),
]


def download(url: str, dest: Path) -> None:
    if dest.exists():
        print(f"Already downloaded: {dest} ({dest.stat().st_size:,} bytes)")
        return

    print(f"Downloading {url} → {dest}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    print(f"Saved {dest} ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    for filename, dest_dir in FILES:
        download(f"{HF_BASE}/{filename}", dest_dir / filename)
