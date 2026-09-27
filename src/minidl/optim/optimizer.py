import math


class Optimizer:
    """State is a flat mapping string -> numeric scalar/ndarray for checkpoints."""

    def __init__(self, parameters, lr=0.01):
        if not math.isfinite(lr) or lr <= 0:
            raise ValueError("lr must be positive")
        self.parameters = list(dict.fromkeys(parameters))
        self.lr = lr
        self.state = {}

    def zero_grad(self):
        for parameter in self.parameters:
            parameter.zero_grad()

    def step(self):
        raise NotImplementedError("Subclass must implement step")
