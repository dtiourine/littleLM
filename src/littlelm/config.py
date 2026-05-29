from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = REPO_ROOT / "data"
CHECKPOINT_DIR = REPO_ROOT / "checkpoints"


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
    epochs: int = 10
    seed: int = 42


@dataclass
class DataConfig:
    text_url: str = (
        "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/"
        "tinyshakespeare/input.txt"
    )
    text_file: str = "tinyshakespeare.txt"
    tokens_file: str = "data.npy"
    tokenizer_file: str = "tokenizer.json"
