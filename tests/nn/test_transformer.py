import numpy as np
import pytest

from minidl.core import Tensor
from minidl.nn import PositionalEncoding, TransformerBlock
from models.transformer import TinyTransformer
from tests.helpers import check_layer

pytestmark = pytest.mark.student


def test_block_zero_residual_is_identity():
    layer = TransformerBlock(2, 1, 3)
    for parameter in layer.parameters():
        parameter.data[...] = 0
    x = Tensor([[[1.0, 3.0], [2.0, 4.0]]])
    np.testing.assert_allclose(layer(x).data, x.data)


def test_block_shape_and_gradients():
    layer = TransformerBlock(2, 1, 3)
    x = np.random.default_rng(2).normal(size=(1, 2, 2))
    assert layer(Tensor(x)).shape == x.shape
    check_layer(layer, [x], atol=1e-4, rtol=2e-3)


def test_position_changes_timesteps():
    output = PositionalEncoding(4)(Tensor(np.zeros((1, 3, 4))))
    assert output.shape == (1, 3, 4)
    assert not np.array_equal(output.data[:, 0], output.data[:, 1])


def test_causality():
    model = TinyTransformer(5, model_dim=4, num_heads=2, max_length=4, dtype=np.float64)
    model.eval()
    a = model(np.array([[0, 1, 2, 3]])).data
    b = model(np.array([[0, 1, 4, 4]])).data
    assert a.shape == (1, 4, 5)
    np.testing.assert_allclose(a[:, :2], b[:, :2], atol=1e-8)
