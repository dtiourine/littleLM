from dataclasses import asdict

from littlelm.config import ModelConfig, TrainConfig
from littlelm.network.transformer import LittleLM
from littlelm.optimizer import SGD


def train(model: LittleLM, dataloader, optimizer: SGD, epochs: int):
    for epoch in range(epochs):
        for batch, targets in dataloader:
            optimizer.zero_grad()
            logits, loss = model.forward(batch, targets=targets)
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    model_cfg = ModelConfig()
    train_cfg = TrainConfig()
    model = LittleLM(**asdict(model_cfg))
