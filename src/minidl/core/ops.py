"""Primitive contracts. All graph creation and derivative rules are student work."""

from minidl.utils.student import todo


def negative(a):
    """Elementwise sign inversion. TODO(student): graph and local derivative."""
    todo("negative", "src/minidl/core/ops.py", "tests/core/test_tensor.py")


def subtract(a, b):
    """Broadcast subtraction. TODO(student): graph and local derivatives."""
    todo("subtract", "src/minidl/core/ops.py", "tests/core/test_tensor.py")


def divide(a, b):
    """Broadcast division. TODO(student): graph and local derivatives."""
    todo("divide", "src/minidl/core/ops.py", "tests/core/test_tensor.py")


def add(a, b):
    """Elementwise addition with NumPy broadcasting. Tests: tests/core/test_tensor.py."""
    # TODO(student): implement numerical forward and local gradient rules.
    todo("add", "src/minidl/core/ops.py", "tests/core/test_tensor.py")


def multiply(a, b):
    """Elementwise product with broadcast gradient reduction. Tests: tests/core/test_tensor.py."""
    # TODO(student): implement numerical forward and local gradient rules.
    todo("multiply", "src/minidl/core/ops.py", "tests/core/test_tensor.py")


def matmul(a, b):
    """Matrix multiplication; batched trailing matrix axes. Tests: tests/core/test_tensor.py."""
    # TODO(student): implement numerical forward and local gradient rules.
    todo("matmul", "src/minidl/core/ops.py", "tests/core/test_tensor.py")


def power(a, exponent):
    """Scalar constant exponent. Tests: tests/core/test_tensor.py."""
    # TODO(student): implement numerical forward and local gradient rules.
    todo("power", "src/minidl/core/ops.py", "tests/core/test_tensor.py")


def reduce_sum(a, axis=None, keepdims=False):
    """Sum reduction and restored gradient shape. Tests: tests/core/test_tensor.py."""
    # TODO(student): implement numerical forward and local gradient rules.
    todo("reduce_sum", "src/minidl/core/ops.py", "tests/core/test_tensor.py")


def reduce_mean(a, axis=None, keepdims=False):
    """Mean over exactly the reduced elements. Tests: tests/core/test_tensor.py."""
    # TODO(student): implement numerical forward and local gradient rules.
    todo("reduce_mean", "src/minidl/core/ops.py", "tests/core/test_tensor.py")


def reshape(a, shape):
    """Reshape without detaching the graph. Tests: tests/core/test_tensor.py."""
    # TODO(student): implement numerical forward and local gradient rules.
    todo("reshape", "src/minidl/core/ops.py", "tests/core/test_tensor.py")


def transpose(a, axes=None):
    """Axis permutation and inverse permutation in backward. Tests: tests/core/test_tensor.py."""
    # TODO(student): implement numerical forward and local gradient rules.
    todo("transpose", "src/minidl/core/ops.py", "tests/core/test_tensor.py")


def exp(a):
    """Elementwise exponential. Tests: tests/core/test_tensor.py."""
    # TODO(student): implement numerical forward and local gradient rules.
    todo("exp", "src/minidl/core/ops.py", "tests/core/test_tensor.py")


def log(a):
    """Elementwise natural logarithm for positive input. Tests: tests/core/test_tensor.py."""
    # TODO(student): implement numerical forward and local gradient rules.
    todo("log", "src/minidl/core/ops.py", "tests/core/test_tensor.py")


def softmax(a, axis=-1):
    """Stable normalized exponentials along axis. Tests: tests/core/test_tensor.py."""
    # TODO(student): implement numerical forward and local gradient rules.
    todo("softmax", "src/minidl/core/ops.py", "tests/core/test_tensor.py")


def concatenate(tensors, axis=0):
    """Join tensors; preserve all parents. Tests: tests/core/test_tensor.py."""
    # TODO(student): implement numerical forward and local gradient rules.
    todo("concatenate", "src/minidl/core/ops.py", "tests/core/test_tensor.py")


def stack(tensors, axis=0):
    """Insert an axis; preserve all parents. Tests: tests/core/test_tensor.py."""
    # TODO(student): implement numerical forward and local gradient rules.
    todo("stack", "src/minidl/core/ops.py", "tests/core/test_tensor.py")


def unbroadcast(gradient, shape):
    """Reduce upstream gradient back to the original operand shape. Tests: tests/core/test_tensor.py."""
    # TODO(student): implement numerical forward and local gradient rules.
    todo("unbroadcast", "src/minidl/core/ops.py", "tests/core/test_tensor.py")
