from dataclasses import asdict

import numpy as np

from littlelm.config import DATA_DIR, MODEL_DIR, DataConfig, ModelConfig, TrainConfig
from littlelm.dataloader import DataLoader
from littlelm.network.transformer import LittleLM
from littlelm.optimizer import SGD
from littlelm.train import train

if __name__ == "__main__":
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    data_cfg = DataConfig()

    print(f"Loading tokenized data from {DATA_DIR / data_cfg.tokens_file}")
    data = np.load(DATA_DIR / data_cfg.tokens_file)
    split = int((1 - train_cfg.val_split) * len(data))
    train_data = data[:split]
    val_data = data[split:]
    print(
        f"Loaded {len(data):,} tokens "
        f"({len(train_data):,} train, {len(val_data):,} val)"
    )

    print("Creating new model")
    model = LittleLM(**asdict(model_cfg))
    optimizer = SGD(model.parameters(), lr=train_cfg.lr)
    train_loader = DataLoader(
        train_data,
        batch_size=train_cfg.batch_size,
        context_len=train_cfg.context_len,
        seed=train_cfg.seed,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=train_cfg.batch_size,
        context_len=train_cfg.context_len,
        seed=train_cfg.seed + 1,
    )

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        max_steps=train_cfg.max_steps,
        log_every=train_cfg.log_every,
        eval_every=train_cfg.eval_every,
        eval_batches=train_cfg.eval_batches,
    )

    print(f"Saving model to {MODEL_DIR}")
    model.save(MODEL_DIR)
    print("Done")
