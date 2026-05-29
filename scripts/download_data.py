from pathlib import Path

import requests

from littlelm.config import DATA_DIR, DataConfig


def download(dest: Path, source_url: str) -> None:
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
    cfg = DataConfig()
    download(DATA_DIR / cfg.text_file, cfg.text_url)
