import numpy as np


class DataLoader:
    """Fresh, reproducible shuffle on each epoch; separate RNG per loader."""

    def __init__(self, dataset, batch_size=64, shuffle=False, drop_last=False, seed=42):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        n = len(self.dataset)
        return (
            n // self.batch_size if self.drop_last else (n + self.batch_size - 1) // self.batch_size
        )

    def __iter__(self):
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            self.rng.shuffle(indices)
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if self.drop_last and len(batch) < self.batch_size:
                break
            yield self.dataset[batch]
