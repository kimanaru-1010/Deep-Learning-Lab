import numpy as np
import pytest

from minidl.core import Tensor
from minidl.nn import ReLU, Sigmoid, Tanh
from tests.helpers import check_layer

pytestmark = pytest.mark.student


@pytest.mark.parametrize(
    "kind, expected",
    [
        (ReLU, [0.0, 0.0, 1.0]),
        (Sigmoid, [0.2689414214, 0.5, 0.7310585786]),
        (Tanh, [-0.761594156, 0.0, 0.761594156]),
    ],
)
def test_known_activation(kind, expected):
    np.testing.assert_allclose(kind()(Tensor([-1.0, 0.0, 1.0])).data, expected, atol=1e-8)


@pytest.mark.parametrize("kind", [ReLU, Sigmoid, Tanh])
def test_activation_gradients(kind):
    check_layer(kind(), [np.array([[-0.7, 0.2, 1.1]])])


def test_relu_zero_convention():
    x = Tensor([0.0], True)
    ReLU()(x).sum().backward()
    np.testing.assert_array_equal(x.grad, [0.0])
