import numpy as np
import pytest

from minidl.core import Tensor
from minidl.nn import LSTM, RNN, LSTMCell, Module, RNNCell
from tests.helpers import check_layer

pytestmark = pytest.mark.student


def test_lstm_cell_known_gate_values():
    cell = LSTMCell(1, 1)
    for parameter in cell.parameters():
        parameter.data[...] = 0
    hidden, memory = cell(Tensor([[0.0]]), (Tensor([[0.0]]), Tensor([[2.0]])))
    np.testing.assert_allclose(memory.data, [[1.0]])
    np.testing.assert_allclose(hidden.data, [[0.3807970779778824]])


def test_rnn_cell_known_nonzero_value():
    cell = RNNCell(1, 1)
    cell.weight_ih.data[...] = 2
    cell.weight_hh.data[...] = 1
    cell.bias.data[...] = 0
    np.testing.assert_allclose(cell(Tensor([[0.25]]), Tensor([[0.5]])).data, [[0.7615941559557649]])


def test_rnn_cell_zero_and_gradients():
    cell = RNNCell(2, 2)
    np.testing.assert_allclose(cell(Tensor(np.zeros((1, 2))), Tensor(np.zeros((1, 2)))).data, 0.0)
    check_layer(cell, [np.array([[0.3, -0.2]]), np.array([[0.2, 0.4]])])


class LSTMAdapter(Module):
    def __init__(self):
        super().__init__()
        self.cell = LSTMCell(2, 2)

    def forward(self, x, h, c):
        return self.cell(x, (h, c))


def test_lstm_cell_all_state_gradients():
    layer = LSTMAdapter()
    check_layer(layer, [np.array([[0.3, -0.2]]), np.array([[0.2, 0.4]]), np.array([[-0.1, 0.2]])])


@pytest.mark.parametrize("kind", [RNN, LSTM])
def test_sequence_shape_and_bptt(kind):
    layer = kind(2, 3)
    x = Tensor(np.random.default_rng(3).normal(size=(1, 3, 2)), True)
    output, state = layer(x)
    assert output.shape == (1, 3, 3)
    output.sum().backward()
    assert x.grad.shape == x.shape
    assert all(p.grad is not None and np.isfinite(p.grad).all() for p in layer.parameters())
