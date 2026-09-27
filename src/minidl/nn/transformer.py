import numpy as np

from minidl.utils.student import todo

from .module import Module


class TransformerBlock(Module):
    """Purpose / mathematical definition: Pre-norm attention and feed-forward sublayers with residual connections.

    Input: (N,T,D).
    Output: (N,T,D).
    Student: forward math, graph connections, local derivatives and edge cases.
    Tests: python -m pytest tests/nn/test_transformer.py --run-student -q
    """

    def __init__(self, model_dim, num_heads, ff_dim, dtype=np.float64):
        super().__init__()
        from .attention import MultiHeadAttention
        from .linear import Linear
        from .normalization import LayerNorm

        self.attention = MultiHeadAttention(model_dim, num_heads, dtype=dtype)
        self.norm1 = LayerNorm(model_dim, dtype=dtype)
        self.norm2 = LayerNorm(model_dim, dtype=dtype)
        self.fc1 = Linear(model_dim, ff_dim, dtype=dtype)
        self.fc2 = Linear(ff_dim, model_dim, dtype=dtype)

    def forward(self, x, mask=None):
        # TODO(student): implement this lesson without detaching Tensor data.
        todo(
            "TransformerBlock.forward",
            "src/minidl/nn/transformer.py",
            "tests/nn/test_transformer.py",
        )


class PositionalEncoding(Module):
    """Purpose / mathematical definition: Provide positional information; choose and document sinusoidal encoding.

    Input: (N,T,D), T <= max_length.
    Output: (N,T,D).
    Student: forward math, graph connections, local derivatives and edge cases.
    Tests: python -m pytest tests/nn/test_transformer.py --run-student -q
    """

    def __init__(self, model_dim, max_length=128):
        super().__init__()
        self.model_dim, self.max_length = model_dim, max_length

    def forward(self, x):
        # TODO(student): implement this lesson without detaching Tensor data.
        todo(
            "PositionalEncoding.forward",
            "src/minidl/nn/transformer.py",
            "tests/nn/test_transformer.py",
        )
