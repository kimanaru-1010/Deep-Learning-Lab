import numpy as np
import pytest

from minidl.core import Tensor
from minidl.nn import Linear
from tests.helpers import check_layer

pytestmark = pytest.mark.student


def test_linear_known_values_and_gradients():
    layer = Linear(3, 2)
    layer.weight.data[...] = [[1, 0], [0, 2], [-1, 1]]
    layer.bias.data[...] = [0.5, -0.5]
    x = np.array([[1.0, 2.0, 3.0], [-1.0, 0.0, 2.0]])
    np.testing.assert_allclose(layer(Tensor(x)).data, [[-1.5, 6.5], [-2.5, 1.5]])
    check_layer(layer, [x])


def test_linear_no_bias():
    layer = Linear(2, 1, bias=False)
    assert layer.bias is None
    assert layer(Tensor([[1.0, 2.0]])).shape == (1, 1)
