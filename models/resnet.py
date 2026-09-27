import numpy as np

from minidl.nn import Conv2D, Linear, MaxPool2D, Module, ReLU
from minidl.utils.student import todo


class ResidualBlock(Module):
    """Input/output (N,C,H,W); student implements F(x)+x and skip gradients.

    Identity skip, two padded 3x3 convolutions, ReLU after first convolution.
    Tests: tests/nn/test_resnet.py. No downsampling inside the block.
    """

    def __init__(self, channels, dtype=np.float64):
        super().__init__()
        self.conv1 = Conv2D(channels, channels, 3, padding=1, dtype=dtype)
        self.relu = ReLU()
        self.conv2 = Conv2D(channels, channels, 3, padding=1, dtype=dtype)

    def forward(self, x):
        todo("ResidualBlock.forward", "models/resnet.py", "tests/nn/test_resnet.py")


class SmallResNet(Module):
    def __init__(self, dtype=np.float32):
        super().__init__()
        self.stem = Conv2D(1, 4, 3, padding=1, dtype=dtype)
        self.block = ResidualBlock(4, dtype=dtype)
        self.pool = MaxPool2D(2)
        self.fc = Linear(4 * 14 * 14, 10, dtype=dtype)

    def forward(self, x):
        x = self.pool(self.block(self.stem(x)))
        return self.fc(x.reshape(x.shape[0], -1))
