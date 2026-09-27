import numpy as np

from minidl.utils.student import todo

from .module import Module


class ScaledDotProductAttention(Module):
    """Purpose / mathematical definition: softmax(Q K^T / sqrt(d_k) + mask) V; True mask entries are allowed.

    Input: Q:(N,T,D), K:(N,S,D), V:(N,S,V), boolean mask:(T,S).
    Output: (N,T,V); reject fully masked rows.
    Student: forward math, graph connections, local derivatives and edge cases.
    Tests: python -m pytest tests/nn/test_attention.py --run-student -q
    """

    def __init__(self):
        super().__init__()
        # No trainable parameters.

    def forward(self, query, key, value, mask=None):
        # TODO(student): implement this lesson without detaching Tensor data.
        todo(
            "ScaledDotProductAttention.forward",
            "src/minidl/nn/attention.py",
            "tests/nn/test_attention.py",
        )


class MultiHeadAttention(Module):
    """Purpose / mathematical definition: Project Q/K/V, split heads, attend, concatenate and project.

    Input: (N,T,D) and optional allowed-position boolean mask:(T,T).
    Output: (N,T,D).
    Student: forward math, graph connections, local derivatives and edge cases.
    Tests: python -m pytest tests/nn/test_attention.py --run-student -q
    """

    def __init__(self, model_dim, num_heads, dtype=np.float64):
        super().__init__()
        from .linear import Linear

        if model_dim % num_heads:
            raise ValueError("model_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.query = Linear(model_dim, model_dim, dtype=dtype)
        self.key = Linear(model_dim, model_dim, dtype=dtype)
        self.value = Linear(model_dim, model_dim, dtype=dtype)
        self.output = Linear(model_dim, model_dim, dtype=dtype)

    def forward(self, x, mask=None):
        # TODO(student): implement this lesson without detaching Tensor data.
        todo(
            "MultiHeadAttention.forward", "src/minidl/nn/attention.py", "tests/nn/test_attention.py"
        )
