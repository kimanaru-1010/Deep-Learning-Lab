import numpy as np
import pytest

from minidl.core import Tensor
from minidl.nn import Conv2D
from tests.helpers import check_layer

pytestmark = pytest.mark.student


def test_conv_known_values():
    layer = Conv2D(1, 1, 2)
    layer.weight.data[...] = 1
    layer.bias.data[...] = 0
    x = Tensor(np.arange(1.0, 10.0).reshape(1, 1, 3, 3))
    np.testing.assert_allclose(layer(x).data, [[[[12.0, 16.0], [24.0, 28.0]]]])


@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 1)])
def test_conv_gradient(stride, padding):
    rng = np.random.default_rng(4)
    check_layer(Conv2D(1, 2, 2, stride=stride, padding=padding), [rng.normal(size=(1, 1, 3, 3))])
