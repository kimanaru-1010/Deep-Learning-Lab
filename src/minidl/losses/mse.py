from minidl.nn.module import Module
from minidl.utils.student import todo


class MSELoss(Module):
    """Mean squared error over all elements; predictions and targets same shape.

    Output: scalar Tensor. Student: stable forward and correct gradient scaling.
    Tests: python -m pytest tests/losses --run-student -q
    """

    def forward(self, prediction, target):
        # TODO(student): core loss mathematics.
        todo("MSELoss.forward", "src/minidl/losses/mse.py", "tests/losses")
