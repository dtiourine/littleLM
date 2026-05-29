from pathlib import Path

import requests


def download(dest: Path) -> None:
    source_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    if dest.exists():
        print(f"Already downloaded: {dest} ({dest.stat().st_size:,} bytes)")
        return

    print(f"Downloading {source_url} → {dest}")
    response = requests.get(source_url, timeout=30)
    response.raise_for_status()

    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(response.content)
    print(f"Saved {dest} ({dest.stat().st_size:,} bytes)")


if __name__ == "__main__":
    dest = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"
    download(dest)
