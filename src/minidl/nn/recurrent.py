import numpy as np

from minidl.utils.student import todo

from .initialization import normal
from .module import Module
from .parameter import Parameter


class RNNCell(Module):
    """Purpose / mathematical definition: h_t = tanh(x_t W_x + h_prev W_h + b); shared weights through time.

    Input: x:(N,I), hidden:(N,H).
    Output: hidden:(N,H).
    Student: forward math, graph connections, local derivatives and edge cases.
    Tests: python -m pytest tests/nn/test_recurrent.py --run-student -q
    """

    def __init__(self, input_size, hidden_size, dtype=np.float64):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(normal((input_size, hidden_size), dtype=dtype))
        self.weight_hh = Parameter(normal((hidden_size, hidden_size), dtype=dtype))
        self.bias = Parameter(np.zeros(hidden_size, dtype=dtype))

    def forward(self, x, hidden):
        # TODO(student): implement this lesson without detaching Tensor data.
        todo("RNNCell.forward", "src/minidl/nn/recurrent.py", "tests/nn/test_recurrent.py")


class RNN(Module):
    """Purpose / mathematical definition: Unroll one shared cell; BPTT must accumulate each timestep contribution.

    Input: (N,T,I), optional (N,H).
    Output: (outputs:(N,T,H), final_hidden:(N,H)).
    Student: forward math, graph connections, local derivatives and edge cases.
    Tests: python -m pytest tests/nn/test_recurrent.py --run-student -q
    """

    def __init__(self, input_size, hidden_size, dtype=np.float64):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = RNNCell(input_size, hidden_size, dtype=dtype)

    def forward(self, x, hidden=None):
        # TODO(student): implement this lesson without detaching Tensor data.
        todo("RNN.forward", "src/minidl/nn/recurrent.py", "tests/nn/test_recurrent.py")


class LSTMCell(Module):
    """Purpose / mathematical definition: Input, forget, candidate, output gates (i,f,g,o); cell and hidden recurrence.

    Input: x:(N,I), state:(h,c) each (N,H).
    Output: (hidden,cell) each (N,H).
    Student: forward math, graph connections, local derivatives and edge cases.
    Tests: python -m pytest tests/nn/test_recurrent.py --run-student -q
    """

    def __init__(self, input_size, hidden_size, dtype=np.float64):
        super().__init__()
        self.hidden_size = hidden_size
        self.weight_ih = Parameter(normal((input_size, 4 * hidden_size), dtype=dtype))
        self.weight_hh = Parameter(normal((hidden_size, 4 * hidden_size), dtype=dtype))
        self.bias = Parameter(np.zeros(4 * hidden_size, dtype=dtype))

    def forward(self, x, state):
        # TODO(student): implement this lesson without detaching Tensor data.
        todo("LSTMCell.forward", "src/minidl/nn/recurrent.py", "tests/nn/test_recurrent.py")


class LSTM(Module):
    """Purpose / mathematical definition: Unroll LSTMCell with shared parameters; retain both state paths.

    Input: (N,T,I), optional (h,c).
    Output: (outputs:(N,T,H), (h,c)).
    Student: forward math, graph connections, local derivatives and edge cases.
    Tests: python -m pytest tests/nn/test_recurrent.py --run-student -q
    """

    def __init__(self, input_size, hidden_size, dtype=np.float64):
        super().__init__()
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size, dtype=dtype)

    def forward(self, x, state=None):
        # TODO(student): implement this lesson without detaching Tensor data.
        todo("LSTM.forward", "src/minidl/nn/recurrent.py", "tests/nn/test_recurrent.py")
