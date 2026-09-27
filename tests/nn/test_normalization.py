import numpy as np
import pytest

from minidl.core import Tensor
from minidl.nn import BatchNorm, LayerNorm
from tests.helpers import check_layer

pytestmark = pytest.mark.student


@pytest.mark.parametrize(
    "layer,shape", [(LayerNorm(3), (2, 3)), (BatchNorm(2), (3, 2)), (BatchNorm(2), (2, 2, 2, 2))]
)
def test_normalization_gradients(layer, shape):
    check_layer(layer, [np.random.default_rng(6).normal(size=shape)])


def test_layernorm_known_values():
    layer = LayerNorm(2, eps=0)
    np.testing.assert_allclose(layer(Tensor([[1.0, 3.0]])).data, [[-1.0, 1.0]])


def test_batchnorm_train_eval_running_stats():
    layer = BatchNorm(2, momentum=1.0)
    layer(Tensor([[1.0, 2.0], [3.0, 6.0]]))
    np.testing.assert_allclose(layer.running_mean, [2.0, 4.0])
    before = layer.running_mean.copy()
    layer.eval()
    layer(Tensor([[50.0, 60.0]]))
    np.testing.assert_array_equal(layer.running_mean, before)
