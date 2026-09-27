import numpy as np

from minidl.utils.student import todo

from .initialization import normal
from .module import Module
from .parameter import Parameter


class Conv2D(Module):
    """Purpose / mathematical definition: 2D cross-correlation, no kernel reversal; learn spatial weight sharing.

    Input: (N,C,H,W).
    Output: (N,O, floor((H+2P-K)/S)+1, floor((W+2P-K)/S)+1).
    Student: forward math, graph connections, local derivatives and edge cases.
    Tests: python -m pytest tests/nn/test_conv.py --run-student -q
    """

    def __init__(
        self, in_channels, out_channels, kernel_size=3, stride=1, padding=0, dtype=np.float64
    ):
        super().__init__()
        if min(in_channels, out_channels, kernel_size, stride) <= 0 or padding < 0:
            raise ValueError("Invalid convolution dimensions")
        self.stride, self.padding = stride, padding
        self.weight = Parameter(
            normal((out_channels, in_channels, kernel_size, kernel_size), dtype=dtype)
        )
        self.bias = Parameter(np.zeros(out_channels, dtype=dtype))

    def forward(self, x):
        # TODO(student): implement this lesson without detaching Tensor data.
        todo("Conv2D.forward", "src/minidl/nn/conv.py", "tests/nn/test_conv.py")
