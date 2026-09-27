"""Lesson 01: scalar graph skeleton. No copied or hidden autograd solution."""

from minidl.utils.student import todo


class Value:
    def __init__(self, data, parents=(), operation="leaf"):
        self.data = float(data)
        self.grad = 0.0
        self.parents = tuple(parents)
        self.operation = operation
        self.local_backward = None

    def __add__(self, other):
        todo("Value addition", "experiments/scalar.py", "tests/core/test_scalar.py")

    __radd__ = __add__

    def __mul__(self, other):
        todo("Value multiplication", "experiments/scalar.py", "tests/core/test_scalar.py")

    __rmul__ = __mul__

    def __pow__(self, exponent):
        todo("Value power", "experiments/scalar.py", "tests/core/test_scalar.py")

    def backward(self):
        # TODO(student): topological ordering, reverse traversal, gradient accumulation.
        todo("Value.backward", "experiments/scalar.py", "tests/core/test_scalar.py")

    def zero_grad(self):
        self.grad = 0.0
