"""Evaluation helpers. The visible training loop lives in experiments/mnist.py."""

import numpy as np

from minidl.core import Tensor
from minidl.training.metrics import accuracy


def evaluate(model, loader, criterion, sequence=False):
    previous_mode = model.training
    model.eval()
    total_loss, total_correct, count = 0.0, 0.0, 0
    predictions, targets = [], []
    try:
        for x, y in loader:
            logits = model(x if sequence else Tensor(x))
            if sequence:
                logits = logits.reshape(-1, logits.shape[-1])
                y = y.reshape(-1)
            loss = criterion(logits, y)
            n = y.size
            total_loss += float(loss.data) * n
            total_correct += accuracy(logits.data, y) * n
            count += n
            predictions.append(np.argmax(logits.data, axis=-1))
            targets.append(y)
    finally:
        model.train(previous_mode)
    if not count:
        raise ValueError("Cannot evaluate an empty loader")
    return {
        "loss": total_loss / count,
        "accuracy": total_correct / count,
        "predictions": np.concatenate(predictions),
        "targets": np.concatenate(targets),
    }
