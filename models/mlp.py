import numpy as np

from minidl.nn import Linear, Module, ReLU


class MLP(Module):
    """784 -> hidden -> 10; composition only, layer algorithms remain TODO."""

    def __init__(self, input_size=784, hidden_size=128, num_classes=10, dtype=np.float32):
        super().__init__()
        self.fc1 = Linear(input_size, hidden_size, dtype=dtype)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, num_classes, dtype=dtype)

    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
