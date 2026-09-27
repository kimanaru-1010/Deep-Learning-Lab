from minidl.utils.student import todo

from .module import Module


class MaxPool2D(Module):
    """Purpose / mathematical definition: Maximum per window; route gradients to recorded maxima, first index on ties.

    Input: (N,C,H,W).
    Output: (N,C, floor((H-K)/S)+1, floor((W-K)/S)+1).
    Student: forward math, graph connections, local derivatives and edge cases.
    Tests: python -m pytest tests/nn/test_pooling.py --run-student -q
    """

    def __init__(self, kernel_size=2, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = kernel_size if stride is None else stride

    def forward(self, x):
        # TODO(student): implement this lesson without detaching Tensor data.
        todo("MaxPool2D.forward", "src/minidl/nn/pooling.py", "tests/nn/test_pooling.py")
