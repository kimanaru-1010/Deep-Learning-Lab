from minidl.nn.module import Module
from minidl.utils.student import todo


class CrossEntropyLoss(Module):
    """Stable mean negative log likelihood from logits (N,C) and integer labels (N,).

    Output: scalar Tensor. Student: stable forward and correct gradient scaling.
    Tests: python -m pytest tests/losses --run-student -q
    """

    def forward(self, prediction, target):
        # TODO(student): core loss mathematics.
        todo("CrossEntropyLoss.forward", "src/minidl/losses/cross_entropy.py", "tests/losses")
