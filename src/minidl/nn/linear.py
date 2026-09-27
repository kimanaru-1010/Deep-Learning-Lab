import numpy as np

from minidl.utils.student import todo

from .initialization import normal
from .module import Module
from .parameter import Parameter


class Linear(Module):
    """Purpose: affine projection, Y = XW + b.

    Input: (batch, in_features). Output: (batch, out_features).
    Weight: (in_features, out_features); bias: (out_features,).
    Student: differentiable forward and correct broadcast gradients via Tensor ops.
    Tests: python -m pytest tests/nn/test_linear.py --run-student -q
    """

    def __init__(self, in_features, out_features, bias=True, dtype=np.float64):
        super().__init__()
        if min(in_features, out_features) <= 0:
            raise ValueError("Feature counts must be positive")
        self.weight = Parameter(normal((in_features, out_features), dtype=dtype))
        self.bias = Parameter(np.zeros(out_features, dtype=dtype)) if bias else None

    def forward(self, x):
        # TODO(student): preserve the computation graph.
        todo("Linear.forward", "src/minidl/nn/linear.py", "tests/nn/test_linear.py")
