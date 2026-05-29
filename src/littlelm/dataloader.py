class DataLoader:
    def __init__(self, tokenized: list[int], targets, batch_size=32):
        self.data = tokenized
        self.batch_size = batch_size

    def __iter__(self):
        for i in range(0, len(self.data), self.batch_size):
            yield self.data[i : i + self.batch_size]
