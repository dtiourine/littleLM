from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
MODEL_DIR = REPO_ROOT / "model"


@dataclass
class ModelConfig:
    d_model: int = 64
    vocab_size: int = 1024
    n_heads: int = 4
    n_layers: int = 2


@dataclass
class TrainConfig:
    batch_size: int = 32
    context_len: int = 128
    lr: float = 1e-3
    max_steps: int = 1000
    log_every: int = 50
    eval_every: int = 100
    eval_batches: int = 20
    val_split: float = 0.1
    seed: int = 42


@dataclass
class DataConfig:
    text_url: str = (
        "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
        "TinyStories-train.txt?download=true"
    )
    text_file: str = "tinystories-train.txt"
    tokens_file: str = "data.npy"
    tokenizer_file: str = "tokenizer.json"
