import numpy as np

from minidl.nn import Conv2D, Linear, MaxPool2D, Module, ReLU


class CNN(Module):
    """MNIST NCHW: conv3 -> ReLU -> pool2 -> conv3 -> ReLU -> flatten -> linear."""

    def __init__(self, dtype=np.float32):
        super().__init__()
        self.conv1 = Conv2D(1, 4, 3, dtype=dtype)
        self.relu1 = ReLU()
        self.pool = MaxPool2D(2)
        self.conv2 = Conv2D(4, 8, 3, dtype=dtype)
        self.relu2 = ReLU()
        self.fc = Linear(8 * 11 * 11, 10, dtype=dtype)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.relu2(self.conv2(x))
        return self.fc(x.reshape(x.shape[0], -1))
