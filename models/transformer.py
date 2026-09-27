import numpy as np

from minidl.nn import Embedding, LayerNorm, Linear, Module, PositionalEncoding, TransformerBlock
from minidl.utils.student import todo


class TinyTransformer(Module):
    """Causal character LM: IDs (N,T) -> logits (N,T,V).

    Student: embedding + position + causal mask + blocks + final norm/head.
    A True mask entry means attention is allowed. Tests: tests/nn/test_transformer.py.
    """

    def __init__(self, vocabulary_size, model_dim=32, num_heads=2, max_length=64, dtype=np.float32):
        super().__init__()
        self.embedding = Embedding(vocabulary_size, model_dim, dtype=dtype)
        self.position = PositionalEncoding(model_dim, max_length)
        self.block = TransformerBlock(model_dim, num_heads, 4 * model_dim, dtype=dtype)
        self.norm = LayerNorm(model_dim, dtype=dtype)
        self.head = Linear(model_dim, vocabulary_size, dtype=dtype)

    def forward(self, tokens):
        todo("TinyTransformer.forward", "models/transformer.py", "tests/nn/test_transformer.py")
