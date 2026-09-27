import numpy as np
import pytest

from minidl.nn import Embedding
from minidl.utils.gradcheck import check_gradient

pytestmark = pytest.mark.student


def test_embedding_repeated_ids_gradient():
    layer = Embedding(3, 2)
    layer.weight.data[...] = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    ids = np.array([[0, 2, 0]])
    np.testing.assert_array_equal(layer(ids).data, [[[1.0, 2.0], [5.0, 6.0], [1.0, 2.0]]])
    layer(ids).sum().backward()
    np.testing.assert_allclose(layer.weight.grad, [[2.0, 2.0], [0.0, 0.0], [1.0, 1.0]])
    gradient = layer.weight.grad.copy()
    original = layer.weight.data.copy()

    def function(value):
        layer.weight.data[...] = value
        return layer(ids).data.sum()

    try:
        assert check_gradient(function, original, gradient).passed
    finally:
        layer.weight.data[...] = original


def test_embedding_bounds():
    with pytest.raises((ValueError, IndexError)):
        Embedding(2, 3)(np.array([[2]]))
