"""Tensor storage and public API. All differentiable operations are student work."""

import numpy as np

from minidl.utils.student import todo


class Tensor:
    def __init__(self, data, requires_grad=False, *, parents=(), operation="leaf"):
        self.data = np.asarray(data)
        if self.data.dtype.kind not in "fc":
            self.data = self.data.astype(np.float64)
        self.requires_grad = requires_grad
        self.grad = None
        self.parents = tuple(parents)
        self.operation = operation
        self.context = {}

    @property
    def shape(self):
        return self.data.shape

    def zero_grad(self):
        self.grad = None

    def backward(self, gradient=None):
        """Seed scalar output with 1, or accept a matching upstream array.

        TODO(student): traversal, propagation, shared-node accumulation.
        Leaf gradients accumulate across backward calls until zero_grad().
        Rebuild the graph for each new forward pass.
        """
        from .autograd import backward

        return backward(self, gradient)

    def __add__(self, other):
        from .ops import add

        return add(self, other)

    __radd__ = __add__

    def __mul__(self, other):
        from .ops import multiply

        return multiply(self, other)

    __rmul__ = __mul__

    def __neg__(self):
        from .ops import negative

        return negative(self)

    def __sub__(self, other):
        from .ops import subtract

        return subtract(self, other)

    def __rsub__(self, other):
        from .ops import subtract

        return subtract(other, self)

    def __truediv__(self, other):
        from .ops import divide

        return divide(self, other)

    def __rtruediv__(self, other):
        from .ops import divide

        return divide(other, self)

    def exp(self):
        from .ops import exp

        return exp(self)

    def log(self):
        from .ops import log

        return log(self)

    def __matmul__(self, other):
        from .ops import matmul

        return matmul(self, other)

    def __pow__(self, exponent):
        from .ops import power

        return power(self, exponent)

    def sum(self, axis=None, keepdims=False):
        from .ops import reduce_sum

        return reduce_sum(self, axis, keepdims)

    def mean(self, axis=None, keepdims=False):
        from .ops import reduce_mean

        return reduce_mean(self, axis, keepdims)

    def reshape(self, *shape):
        from .ops import reshape

        return reshape(self, shape)

    def transpose(self, *axes):
        from .ops import transpose

        return transpose(self, axes or None)

    def __getitem__(self, index):
        # TODO(student): include accumulation for repeated indices.
        todo("Tensor indexing", "src/minidl/core/tensor.py", "tests/core")

    def __repr__(self):
        return f"Tensor(shape={self.shape}, requires_grad={self.requires_grad})"
