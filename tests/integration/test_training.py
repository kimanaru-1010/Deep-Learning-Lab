import numpy as np
import pytest

from minidl.core import Tensor
from minidl.data import ArrayDataset, DataLoader
from minidl.losses import CrossEntropyLoss
from minidl.optim import SGD
from minidl.training.trainer import evaluate
from minidl.utils.seed import set_seed
from models.mlp import MLP

pytestmark = pytest.mark.student


def test_real_training_updates_and_eval_does_not_mutate():
    set_seed(42)
    model = MLP(2, 8, 2, dtype=np.float64)
    x = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, -1.0]])
    y = np.array([0, 1, 0, 1])
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters(), lr=0.1)
    initial = [p.data.copy() for p in model.parameters()]
    losses = []
    for _ in range(30):
        loss = criterion(model(Tensor(x)), y)
        losses.append(float(loss.data))
        optimizer.zero_grad()
        loss.backward()
        assert all(p.grad is not None for p in model.parameters())
        optimizer.step()
    assert losses[-1] < losses[0]
    assert any(not np.array_equal(a, p.data) for a, p in zip(initial, model.parameters()))
    saved = [(p.data.copy(), p.grad.copy()) for p in model.parameters()]
    evaluate(model, DataLoader(ArrayDataset(x, y), 2), criterion)
    for p, (data, gradient) in zip(model.parameters(), saved):
        np.testing.assert_array_equal(p.data, data)
        np.testing.assert_array_equal(p.grad, gradient)
