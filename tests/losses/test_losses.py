import numpy as np
import pytest

from minidl.core import Tensor
from minidl.losses import CrossEntropyLoss, MSELoss
from minidl.utils.gradcheck import check_gradient

pytestmark = pytest.mark.student


def test_mse_known_value_and_gradient():
    x = Tensor([1.0, 3.0], True)
    target = np.array([0.0, 1.0])
    loss = MSELoss()(x, target)
    assert loss.shape == () and float(loss.data) == 2.5
    loss.backward()
    np.testing.assert_allclose(x.grad, [1, 2])
    assert check_gradient(lambda z: MSELoss()(Tensor(z), target).data, x.data, x.grad).passed


def test_cross_entropy_stability_and_gradient():
    x = Tensor(np.full((2, 2), 1000.0), True)
    labels = np.array([0, 1])
    loss = CrossEntropyLoss()(x, labels)
    assert float(loss.data) == pytest.approx(np.log(2))
    loss.backward()
    np.testing.assert_allclose(x.grad, [[-0.25, 0.25], [0.25, -0.25]])
    assert check_gradient(
        lambda z: CrossEntropyLoss()(Tensor(z), labels).data, x.data, x.grad
    ).passed


def test_cross_entropy_rejects_invalid_labels():
    with pytest.raises(ValueError):
        CrossEntropyLoss()(Tensor([[1.0, 2.0]]), np.array([2]))
