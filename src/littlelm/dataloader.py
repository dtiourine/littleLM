import numpy as np


class DataLoader:
    def __init__(
        self,
        data: np.ndarray,
        batch_size: int,
        context_len: int,
        seed: int | None = None,
    ):
        self.data = data
        self.batch_size = batch_size
        self.context_len = context_len
        self.rng = np.random.default_rng(seed)

    def __iter__(self):
        while True:
            chunks = []
            start_positions = self.rng.integers(
                0, len(self.data) - self.context_len, size=self.batch_size
            )
            for start in start_positions:
                chunks.append(self.data[start : start + self.context_len + 1])
            chunks = np.stack(chunks)

            inputs = chunks[:, :-1]
            targets = chunks[:, 1:]
            yield inputs, targets
