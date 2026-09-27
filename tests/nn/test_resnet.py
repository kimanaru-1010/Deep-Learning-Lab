import numpy as np
import pytest

from minidl.core import Tensor
from models.resnet import ResidualBlock
from tests.helpers import check_layer

pytestmark = pytest.mark.student


def test_identity_skip_and_gradient():
    layer = ResidualBlock(1)
    for parameter in layer.parameters():
        parameter.data[...] = 0
    x = np.array([[[[0.2, 0.5], [-0.3, 0.1]]]])
    np.testing.assert_allclose(layer(Tensor(x)).data, x)
    # Inputs of zero-weight convolutions are at ReLU kinks; check skip input only.
    tensor = Tensor(x, True)
    layer(tensor).sum().backward()
    np.testing.assert_allclose(tensor.grad, np.ones_like(x))


def test_residual_general_gradient():
    layer = ResidualBlock(1)
    layer.conv1.bias.data[...] = 0.3
    check_layer(layer, [np.array([[[[0.2, 0.5], [-0.3, 0.1]]]])])
