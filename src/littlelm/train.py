from littlelm.dataloader import DataLoader
from littlelm.network.transformer import LittleLM
from littlelm.optimizer import SGD


def evaluate(model: LittleLM, dataloader: DataLoader, num_batches: int) -> float:
    losses = []
    for _ in range(num_batches):
        inputs, targets = next(iter(dataloader))
        _, loss = model.forward(inputs, targets=targets)
        assert loss is not None
        losses.append(float(loss.data))
    return sum(losses) / len(losses)


def train(
    model: LittleLM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: SGD,
    max_steps: int,
    log_every: int,
    eval_every: int,
    eval_batches: int,
):
    for step in range(max_steps):
        inputs, targets = next(iter(train_loader))
        optimizer.zero_grad()
        _, loss = model.forward(inputs, targets=targets)
        assert loss is not None
        loss.backward()
        optimizer.step()

        if step % eval_every == 0:
            val_loss = evaluate(model, val_loader, num_batches=eval_batches)
            print(
                f"step {step:5d} | train_loss {float(loss.data):.4f} "
                f"| val_loss {val_loss:.4f}"
            )
        elif step % log_every == 0:
            print(f"step {step:5d} | train_loss {float(loss.data):.4f}")
