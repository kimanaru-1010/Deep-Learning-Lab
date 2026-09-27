import numpy as np

from minidl.nn import RNN, Embedding, Linear, Module
from minidl.utils.student import todo


class RNNLanguageModel(Module):
    """IDs (N,T) -> logits (N,T,V); student composes embedding, RNN, output head."""

    def __init__(self, vocabulary_size, model_dim=32, dtype=np.float32):
        super().__init__()
        self.embedding = Embedding(vocabulary_size, model_dim, dtype=dtype)
        self.recurrent = RNN(model_dim, model_dim, dtype=dtype)
        self.head = Linear(model_dim, vocabulary_size, dtype=dtype)

    def forward(self, tokens):
        todo("RNNLanguageModel.forward", "models/rnn.py", "tests/nn/test_recurrent.py")
