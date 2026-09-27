import numpy as np
import pytest

from minidl.core import Tensor
from minidl.nn import MultiHeadAttention, ScaledDotProductAttention
from tests.helpers import check_layer

pytestmark = pytest.mark.student


def test_multihead_uniform_projection_values():
    layer = MultiHeadAttention(2, 2)
    for parameter in layer.parameters():
        parameter.data[...] = 0
    layer.value.weight.data[...] = np.eye(2)
    layer.output.weight.data[...] = np.eye(2)
    np.testing.assert_allclose(
        layer(Tensor([[[2.0, 4.0], [6.0, 8.0]]])).data, [[[4.0, 6.0], [4.0, 6.0]]]
    )


def test_uniform_scores_and_causal_mask():
    q, k = Tensor(np.zeros((1, 2, 2))), Tensor(np.zeros((1, 2, 2)))
    v = Tensor(np.array([[[2.0, 4.0], [6.0, 8.0]]]))
    layer = ScaledDotProductAttention()
    np.testing.assert_allclose(layer(q, k, v).data, [[[4.0, 6.0], [4.0, 6.0]]])
    np.testing.assert_allclose(
        layer(q, k, v, np.tril(np.ones((2, 2), dtype=bool))).data, [[[2.0, 4.0], [4.0, 6.0]]]
    )
    with pytest.raises(ValueError):
        layer(q, k, v, np.zeros((2, 2), dtype=bool))


def test_attention_all_inputs_gradient():
    rng = np.random.default_rng(4)
    check_layer(ScaledDotProductAttention(), [rng.normal(size=(1, 2, 2)) for _ in range(3)])


def test_multihead_gradients():
    check_layer(MultiHeadAttention(4, 2), [np.random.default_rng(4).normal(size=(1, 2, 4))])
