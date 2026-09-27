from minidl.utils.student import todo

from .optimizer import Optimizer


class Momentum(Optimizer):
    """Velocity v <- momentum*v + gradient; theta <- theta - lr*v.

    Input: iterable of Parameters with same-shaped gradients. Output: in-place update.
    Student: update rule and persistent state (include counter in state for Adam).
    Tests: python -m pytest tests/optim --run-student -q
    """

    def __init__(self, parameters, lr=0.01, momentum=0.9):
        super().__init__(parameters, lr)
        self.momentum = momentum

    def step(self):
        # TODO(student): never use finite differences to compute training gradients.
        todo("Momentum.step", "src/minidl/optim/momentum.py", "tests/optim")
