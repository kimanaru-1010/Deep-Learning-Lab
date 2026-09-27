from minidl.utils.student import todo

from .optimizer import Optimizer


class SGD(Optimizer):
    """theta <- theta - lr * gradient. Skip parameters with grad=None.

    Input: iterable of Parameters with same-shaped gradients. Output: in-place update.
    Student: update rule and persistent state (include counter in state for Adam).
    Tests: python -m pytest tests/optim --run-student -q
    """

    def __init__(self, parameters, lr=0.01):
        super().__init__(parameters, lr)

    def step(self):
        # TODO(student): never use finite differences to compute training gradients.
        todo("SGD.step", "src/minidl/optim/sgd.py", "tests/optim")
