import numpy as np


class ArrayDataset:
    def __init__(self, x, y):
        self.x, self.y = np.asarray(x), np.asarray(y)
        if len(self.x) != len(self.y):
            raise ValueError("x and y must have equal length")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def train_val_split(dataset, val_fraction=0.1, seed=42):
    if not 0 < val_fraction < 1 or len(dataset) < 2:
        raise ValueError("Need at least 2 samples and 0 < val_fraction < 1")
    indices = np.random.default_rng(seed).permutation(len(dataset))
    count = min(len(dataset) - 1, max(1, int(len(dataset) * val_fraction)))
    val, train = indices[:count], indices[count:]
    return ArrayDataset(*dataset[train]), ArrayDataset(*dataset[val])
