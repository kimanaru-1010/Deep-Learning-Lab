import numpy as np
import pytest

from minidl.nn import Parameter
from minidl.optim import SGD, Adam, Momentum

pytestmark = pytest.mark.student


@pytest.mark.parametrize(
    "kind, expected", [(SGD, [1.95, 1.90]), (Momentum, [1.95, 1.855]), (Adam, [1.9, 1.8])]
)
def test_known_updates(kind, expected):
    weight = Parameter([2.0])
    unused = Parameter([7.0])
    optimizer = kind([weight, unused], lr=0.1)
    for value in expected:
        weight.grad = np.array([0.5])
        optimizer.step()
        np.testing.assert_allclose(weight.data, [value], atol=1e-7)
        np.testing.assert_allclose(unused.data, [7])
    optimizer.zero_grad()
    assert weight.grad is None
