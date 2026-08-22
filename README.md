# LittleLM

```text
  _ _ _   _   _     _    __  __ 
 | (_) |_| |_| |___| |  |  \/  |
 | | |  _|  _| / -_) |__| |\/| |
 |_|_|\__|\__|_\___|____|_|  |_|
                                
```

A small decoder-only language model built from scratch in NumPy, all the way from tokenization and autograd to training and inference.


## Quick setup

```bash
# 1. Create env (Python 3.12 recommended)
conda create -n littlelm python=3.12
conda activate littlelm

# 2. Install the package in editable mode
pip install -e ".[dev]"

# 3. Run the test suite to verify everything works
pytest

# 4. Prepare data (downloads TinyShakespeare, trains BPE, encodes corpus)
python scripts/download_data.py
python scripts/tokenize_data.py

# 5. Train the model (writes weights to model/)
python scripts/train_model.py

# 6. Generate text from the trained model
python scripts/generate.py --prompt "ROMEO:"
```

## GPU acceleration with CuPy

CuPy allows the custom autograd and model code to use GPU acceleration while
keeping the NumPy-like array API. Install the optional CUDA dependency, and the
backend will automatically use CuPy when a CUDA-capable GPU is available:

```bash
pip install -e ".[gpu]"
python scripts/train_model.py
```

If CuPy or a CUDA device is unavailable, the backend automatically falls back
to NumPy.

## Use a pretrained checkpoint

Skip training and pull weights + tokenizer from HuggingFace:

```bash
python scripts/download_pretrained.py
python scripts/generate.py --prompt "ROMEO:"
```

## Layout

```
src/littlelm/
├── tensor.py          # custom autograd (numpy-backed Tensor + backward)
├── tokenizer.py       # BPE tokenizer (train/encode/decode/save/load)
├── dataloader.py      # random-sampling LM dataloader
├── loss.py            # cross-entropy with the gather-into-log-softmax trick
├── optimizer.py       # SGD
├── network/
│   ├── transformer.py # LittleLM model + TransformerBlock
│   └── components/    # Embedding, LayerNorm, MLP, MultiheadAttention (RoPE)
├── train.py           # training/eval loops (library)
├── generate.py        # autoregressive sampling (library)
└── config.py          # ModelConfig, TrainConfig, DataConfig dataclasses

scripts/               # one-shot runners (download, tokenize, train, generate)
tests/                 # 130+ tests across tensor, tokenizer, dataloader, model
```
