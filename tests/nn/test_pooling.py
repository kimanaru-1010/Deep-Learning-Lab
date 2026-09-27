import numpy as np
import pytest

from minidl.core import Tensor
from minidl.nn import MaxPool2D
from tests.helpers import check_layer

pytestmark = pytest.mark.student


def test_pool_known_value_and_unique_max_gradient():
    layer = MaxPool2D(2, 1)
    x = np.array([[[[1.0, 3.0, 2.0], [4.0, 8.0, 6.0], [7.0, 5.0, 9.0]]]])
    np.testing.assert_allclose(layer(Tensor(x)).data, [[[[8.0, 8.0], [8.0, 9.0]]]])
    check_layer(layer, [x])


def test_pool_tie_first_index():
    x = Tensor(np.ones((1, 1, 2, 2)), True)
    MaxPool2D(2)(x).sum().backward()
    np.testing.assert_array_equal(x.grad, [[[[1.0, 0.0], [0.0, 0.0]]]])
