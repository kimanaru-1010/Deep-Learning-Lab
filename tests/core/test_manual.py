import numpy as np
import pytest

from experiments.manual import MLP, MSE, CrossEntropy, Linear, ReLU
from minidl.utils.gradcheck import check_gradient

pytestmark = pytest.mark.student


def test_manual_linear_known_forward_and_all_gradients():
    layer = Linear(2, 1)
    layer.weight[...] = [[2], [3]]
    layer.bias[...] = 1
    x = np.array([[1.0, 2.0], [-1.0, 0.5]])
    np.testing.assert_allclose(layer.forward(x), [[9], [0.5]])
    upstream = np.array([[0.7], [-0.2]])
    dx = layer.backward(upstream)
    dw, db = layer.dweight.copy(), layer.dbias.copy()
    assert check_gradient(lambda z: np.sum(layer.forward(z) * upstream), x, dx).passed
    for name, grad in [("weight", dw), ("bias", db)]:
        original = getattr(layer, name).copy()

        def function(value):
            getattr(layer, name)[...] = value
            return np.sum(layer.forward(x) * upstream)

        try:
            assert check_gradient(function, original, grad).passed
        finally:
            getattr(layer, name)[...] = original


def test_manual_relu():
    layer = ReLU()
    np.testing.assert_array_equal(layer.forward(np.array([-2.0, 0.0, 3.0])), [0, 0, 3])
    np.testing.assert_array_equal(layer.backward(np.ones(3)), [0, 0, 1])


def test_manual_losses():
    mse = MSE()
    assert mse.forward(np.array([1.0, 3.0]), np.array([0.0, 1.0])) == 2.5
    np.testing.assert_allclose(mse.backward(), [1, 2])
    ce = CrossEntropy()
    assert ce.forward(np.zeros((2, 2)), np.array([0, 1])) == pytest.approx(np.log(2))
    np.testing.assert_allclose(ce.backward(), [[-0.25, 0.25], [0.25, -0.25]])


def test_manual_mlp_gradient():
    np.random.seed(3)
    model = MLP()
    x = np.array([[0.2, -0.5, 0.8]])
    output = model.forward(x)
    assert output.shape == (1, 2)
    gradient = model.backward(np.ones_like(output))
    assert check_gradient(lambda z: np.sum(model.forward(z)), x, gradient).passed
