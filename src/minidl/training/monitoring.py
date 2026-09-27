import warnings
from dataclasses import dataclass

import numpy as np


def gradient_norm(parameters):
    gradients = [p.grad for p in parameters if p.grad is not None]
    if not gradients:
        return None
    return float(np.sqrt(sum(np.sum(np.asarray(g, dtype=np.float64) ** 2) for g in gradients)))


def parameter_norm(parameters):
    return float(np.sqrt(sum(np.sum(p.data.astype(np.float64) ** 2) for p in parameters)))


def detect_nonfinite(model, loss=None, step=None):
    messages = []
    if loss is not None and not np.isfinite(loss).all():
        messages.append(f"Non-finite loss at step {step}")
    for name, parameter in model.named_parameters():
        for kind, array in (("parameter", parameter.data), ("gradient", parameter.grad)):
            if array is not None and not np.isfinite(array).all():
                messages.append(f"Non-finite {kind} detected in {name} at step {step}")
    for message in messages:
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    return messages


@dataclass
class ParameterWatch:
    name: str = "fc1.weight"
    index: tuple = (0, 0)
    every: int = 10

    def __post_init__(self):
        if self.every <= 0:
            raise ValueError("every must be positive")

    def capture(self, model, step):
        if step % self.every:
            return None
        parameters = dict(model.named_parameters())
        if self.name not in parameters:
            raise ValueError(f"Unknown parameter {self.name}. Choose from {list(parameters)}")
        parameter = parameters[self.name]
        if len(self.index) != parameter.data.ndim or any(
            index < -size or index >= size for index, size in zip(self.index, parameter.shape)
        ):
            raise ValueError(
                f"Watch index {self.index} does not select one value in {self.name} {parameter.shape}"
            )
        return {
            "before": float(parameter.data[self.index]),
            "grad": None if parameter.grad is None else float(parameter.grad[self.index]),
            "grad_norm": gradient_norm([parameter]),
        }

    def report(self, model, snapshot):
        if snapshot is None:
            return None
        parameter = dict(model.named_parameters())[self.name]
        return {
            "name": self.name,
            "index": self.index,
            **snapshot,
            "after": float(parameter.data[self.index]),
            "param_norm": parameter_norm([parameter]),
        }
