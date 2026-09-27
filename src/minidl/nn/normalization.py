import numpy as np

from minidl.utils.student import todo

from .module import Module
from .parameter import Parameter


class BatchNorm(Module):
    """Purpose / mathematical definition: gamma * standardized(x) + beta; training statistics vs running eval statistics.

    Input: (N,C) or (N,C,H,W).
    Output: same shape.
    Student: forward math, graph connections, local derivatives and edge cases.
    Tests: python -m pytest tests/nn/test_normalization.py --run-student -q
    """

    def __init__(self, features, eps=1e-5, momentum=0.1, dtype=np.float64):
        super().__init__()
        self.eps, self.momentum = eps, momentum
        self.weight = Parameter(np.ones(features, dtype=dtype))
        self.bias = Parameter(np.zeros(features, dtype=dtype))
        self.register_buffer("running_mean", np.zeros(features, dtype=dtype))
        self.register_buffer("running_var", np.ones(features, dtype=dtype))

    def forward(self, x):
        # TODO(student): implement this lesson without detaching Tensor data.
        todo(
            "BatchNorm.forward", "src/minidl/nn/normalization.py", "tests/nn/test_normalization.py"
        )


class LayerNorm(Module):
    """Purpose / mathematical definition: Normalize final feature axis per example; gamma and beta affine transform.

    Input: (...,features).
    Output: same shape.
    Student: forward math, graph connections, local derivatives and edge cases.
    Tests: python -m pytest tests/nn/test_normalization.py --run-student -q
    """

    def __init__(self, features, eps=1e-5, dtype=np.float64):
        super().__init__()
        self.eps = eps
        self.weight = Parameter(np.ones(features, dtype=dtype))
        self.bias = Parameter(np.zeros(features, dtype=dtype))

    def forward(self, x):
        # TODO(student): implement this lesson without detaching Tensor data.
        todo(
            "LayerNorm.forward", "src/minidl/nn/normalization.py", "tests/nn/test_normalization.py"
        )
