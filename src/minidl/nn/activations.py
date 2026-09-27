from minidl.utils.student import todo

from .module import Module


class ReLU(Module):
    """Purpose / mathematical definition: Rectification: max(0, x); choose derivative 0 at x=0.

    Input: any shape.
    Output: same shape.
    Student: forward math, graph connections, local derivatives and edge cases.
    Tests: python -m pytest tests/nn/test_activations.py --run-student -q
    """

    def __init__(self):
        super().__init__()
        # No trainable parameters.

    def forward(self, x):
        # TODO(student): implement this lesson without detaching Tensor data.
        todo("ReLU.forward", "src/minidl/nn/activations.py", "tests/nn/test_activations.py")


class Sigmoid(Module):
    """Purpose / mathematical definition: Logistic activation: 1 / (1 + exp(-x)); handle large magnitudes.

    Input: any shape.
    Output: same shape.
    Student: forward math, graph connections, local derivatives and edge cases.
    Tests: python -m pytest tests/nn/test_activations.py --run-student -q
    """

    def __init__(self):
        super().__init__()
        # No trainable parameters.

    def forward(self, x):
        # TODO(student): implement this lesson without detaching Tensor data.
        todo("Sigmoid.forward", "src/minidl/nn/activations.py", "tests/nn/test_activations.py")


class Tanh(Module):
    """Purpose / mathematical definition: Hyperbolic tangent; avoid overflow.

    Input: any shape.
    Output: same shape.
    Student: forward math, graph connections, local derivatives and edge cases.
    Tests: python -m pytest tests/nn/test_activations.py --run-student -q
    """

    def __init__(self):
        super().__init__()
        # No trainable parameters.

    def forward(self, x):
        # TODO(student): implement this lesson without detaching Tensor data.
        todo("Tanh.forward", "src/minidl/nn/activations.py", "tests/nn/test_activations.py")
