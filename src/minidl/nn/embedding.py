import numpy as np

from minidl.utils.student import todo

from .initialization import normal
from .module import Module
from .parameter import Parameter


class Embedding(Module):
    """Purpose / mathematical definition: Lookup weight rows; accumulate gradients for repeated token IDs.

    Input: integer IDs (N,T).
    Output: (N,T,D).
    Student: forward math, graph connections, local derivatives and edge cases.
    Tests: python -m pytest tests/nn/test_embedding.py --run-student -q
    """

    def __init__(self, vocabulary_size, embedding_dim, dtype=np.float64):
        super().__init__()
        self.weight = Parameter(normal((vocabulary_size, embedding_dim), dtype=dtype))

    def forward(self, token_ids):
        # TODO(student): implement this lesson without detaching Tensor data.
        todo("Embedding.forward", "src/minidl/nn/embedding.py", "tests/nn/test_embedding.py")
