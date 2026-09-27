"""Lesson 00: ndarray only, no autograd. Implement forward and manual derivatives."""

import numpy as np

from minidl.utils.student import todo


class Linear:
    """Y=XW+b. Input (N,I), output (N,O); backward returns dX and stores dW/db."""

    def __init__(self, input_size, output_size):
        self.weight = np.random.normal(0, 0.1, (input_size, output_size))
        self.bias = np.zeros(output_size)
        self.dweight = None
        self.dbias = None
        self.cache = None

    def forward(self, x):
        todo("manual Linear.forward", "experiments/manual.py", "tests/core/test_manual.py")

    def backward(self, upstream):
        todo("manual Linear.backward", "experiments/manual.py", "tests/core/test_manual.py")


class ReLU:
    """max(0,x); same input/output shape; backward returns dX, derivative at 0 is 0."""

    def forward(self, x):
        todo("manual ReLU.forward", "experiments/manual.py", "tests/core/test_manual.py")

    def backward(self, upstream):
        todo("manual ReLU.backward", "experiments/manual.py", "tests/core/test_manual.py")


class MSE:
    """Mean squared error over all elements; backward returns prediction gradient."""

    def forward(self, prediction, target):
        todo("manual MSE.forward", "experiments/manual.py", "tests/core/test_manual.py")

    def backward(self):
        todo("manual MSE.backward", "experiments/manual.py", "tests/core/test_manual.py")


class CrossEntropy:
    """Mean CE from logits (N,C), labels (N,); backward returns logit gradient."""

    def forward(self, logits, labels):
        todo("manual CrossEntropy.forward", "experiments/manual.py", "tests/core/test_manual.py")

    def backward(self):
        todo("manual CrossEntropy.backward", "experiments/manual.py", "tests/core/test_manual.py")


class MLP:
    """Two affine layers with ReLU; cache values and explicitly chain backward calls."""

    def __init__(self, input_size=3, hidden_size=4, output_size=2):
        self.fc1 = Linear(input_size, hidden_size)
        self.relu = ReLU()
        self.fc2 = Linear(hidden_size, output_size)

    def forward(self, x):
        todo("manual MLP.forward", "experiments/manual.py", "tests/core/test_manual.py")

    def backward(self, upstream):
        todo("manual MLP.backward", "experiments/manual.py", "tests/core/test_manual.py")
