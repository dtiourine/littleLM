from littlelm.network.transformer import LittleLM
from littlelm.optimizer import SGD


def train(model: LittleLM, epochs: int, dataloader, optimizer: SGD):
    for epoch in range(epochs):
        for batch, targets in dataloader:
            logits, loss = model.forward(batch, targets=targets)
            loss.backward()
            optimizer.zero_grad()
            optimizer.step()


if __name__ == "__main__":
    model = LittleLM(d_model=64, vocab_size=50000, n_heads=4, n_layers=2)
