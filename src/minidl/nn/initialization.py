"""Basic storage initialization; fan-aware initializers are exercises."""

import numpy as np

from minidl.utils.student import todo


def normal(shape, scale=0.02, dtype=np.float64):
    return np.random.normal(0, scale, shape).astype(dtype)


def uniform(shape, low=-0.1, high=0.1, dtype=np.float64):
    return np.random.uniform(low, high, shape).astype(dtype)


def zeros(shape, dtype=np.float64):
    return np.zeros(shape, dtype=dtype)


def xavier(shape, dtype=np.float64):
    """TODO(student): fan-in/fan-out scaling, document your weight convention."""
    todo("Xavier initialization", "src/minidl/nn/initialization.py", "tests/nn")


def he(shape, dtype=np.float64):
    """TODO(student): variance scaling appropriate for ReLU."""
    todo("He initialization", "src/minidl/nn/initialization.py", "tests/nn")
