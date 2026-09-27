from minidl.utils.student import todo

from .optimizer import Optimizer


class Adam(Optimizer):
    """First/second moments, step counter, bias correction, epsilon after square root.

    Input: iterable of Parameters with same-shaped gradients. Output: in-place update.
    Student: update rule and persistent state (include counter in state for Adam).
    Tests: python -m pytest tests/optim --run-student -q
    """

    def __init__(self, parameters, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__(parameters, lr)
        self.beta1, self.beta2, self.eps = beta1, beta2, eps

    def step(self):
        # TODO(student): never use finite differences to compute training gradients.
        todo("Adam.step", "src/minidl/optim/adam.py", "tests/optim")
