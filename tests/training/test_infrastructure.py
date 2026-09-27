import json

import numpy as np
import pytest

from minidl.nn import Module, Parameter, count_parameters, model_summary
from minidl.optim import SGD
from minidl.training.history import History
from minidl.training.metrics import accuracy, confusion_matrix
from minidl.training.monitoring import (
    ParameterWatch,
    detect_nonfinite,
    gradient_norm,
    parameter_norm,
)
from minidl.utils.checkpoint import load_checkpoint, save_checkpoint
from minidl.utils.logging import create_run
from minidl.utils.visualization import plot_confusion_matrix, plot_history


def model():
    module = Module()
    module.weight = Parameter(np.array([[3.0, 4.0]]))
    return module


def test_module_nested_shared_modes_and_summary():
    parent = Module()
    child = model()
    parent.children = [child, {"shared": child.weight}]
    parent.cycle = parent
    assert len(parent.parameters()) == 1
    assert count_parameters(parent) == 2
    assert "children.0.weight" in model_summary(parent)
    parent.eval()
    assert not child.training
    parent.train()
    assert child.training
    child.weight.grad = np.ones((1, 2))
    parent.zero_grad()
    assert child.weight.grad is None


def test_monitoring_and_watch():
    module = model()
    assert parameter_norm(module.parameters()) == 5
    assert gradient_norm(module.parameters()) is None
    module.weight.grad = np.array([[0.3, 0.4]])
    assert gradient_norm(module.parameters()) == pytest.approx(0.5)
    watch = ParameterWatch("weight", (0, 0), every=2)
    before = watch.capture(module, 0)
    module.weight.data[0, 0] = 2.0  # Simulate an external update, not an optimizer.
    report = watch.report(module, before)
    assert report["before"] == 3 and report["after"] == 2 and report["grad"] == 0.3
    assert watch.capture(module, 1) is None
    module.weight.grad[0, 0] = np.inf
    with pytest.warns(RuntimeWarning, match="weight at step 7"):
        assert detect_nonfinite(module, step=7)


def test_metrics_known_values():
    assert accuracy([[0, 2], [3, 0]], [1, 1]) == 0.5
    np.testing.assert_array_equal(confusion_matrix([0, 1, 1], [0, 0, 1], 2), [[1, 0], [1, 1]])
    with pytest.raises(ValueError):
        confusion_matrix([2], [0], 2)


def test_checkpoint_roundtrip_and_atomic_validation(tmp_path):
    module = model()
    module.register_buffer("running_mean", np.array([4.0, 5.0]))
    optimizer = SGD(module.parameters(), 0.2)
    optimizer.state = {"velocity/0": np.array([[1.0, 2.0]]), "step": 3}
    path = tmp_path / "checkpoint.npz"
    save_checkpoint(path, module, optimizer, epoch=2, step=9, metadata={"note": "test"})
    module.weight.data[...] = -1
    module.running_mean[...] = -1
    optimizer.state.clear()
    optimizer.lr = 0.9
    info = load_checkpoint(path, module, optimizer)
    np.testing.assert_array_equal(module.weight.data, [[3, 4]])
    np.testing.assert_array_equal(module.running_mean, [4, 5])
    assert optimizer.lr == 0.2 and int(optimizer.state["step"]) == 3
    assert info["epoch"] == 2 and info["metadata"]["note"] == "test"
    bad = model()
    bad.extra = Parameter([0.0])
    with pytest.raises(ValueError):
        load_checkpoint(path, bad)


def test_run_history_and_all_plots(tmp_path):
    path = create_run(tmp_path, "example", {"seed": 42})
    assert json.loads((path / "config.json").read_text())["seed"] == 42
    with pytest.raises(FileExistsError):
        create_run(tmp_path, "example")
    with pytest.raises(ValueError):
        create_run(tmp_path, "../escape")
    history = History(path / "metrics.csv")
    history.append(
        epoch=1, step=0, train_loss=2.3, train_acc=0.1, grad_norm=0.5, param_norm=2, lr=0.01
    )
    history.append(epoch=1, step=0, val_loss=2.2, val_acc=0.2)
    loaded = History.read(path / "metrics.csv")
    assert loaded.rows[1]["train_loss"] is None
    plot_history(loaded, path)
    plot_confusion_matrix(np.eye(2), path / "confusion_matrix.png")
    assert len(list(path.glob("*.png"))) == 6
    for png in path.glob("*.png"):
        assert png.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")


def test_evaluation_weighting_partial_batch_and_mode_restoration():
    from minidl.core import Tensor
    from minidl.data import ArrayDataset, DataLoader
    from minidl.training.trainer import evaluate

    class EchoScores(Module):
        """Test double for evaluation plumbing only; not a trainable DL model."""

        def forward(self, x):
            assert not self.training
            return x

    scores = np.array([[3.0, 0.0], [0.0, 2.0], [1.0, 0.0]])
    dataset = ArrayDataset(scores, np.array([0, 1, 1]))
    module = EchoScores()
    # A known batch scalar tests weighting without providing any DL loss solution.
    result = evaluate(module, DataLoader(dataset, 2), lambda x, y: Tensor(x.data[:, 0].mean()))
    assert result["loss"] == pytest.approx(4 / 3)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert module.training
    with pytest.raises(ValueError, match="empty"):
        evaluate(module, DataLoader(ArrayDataset(scores[:0], np.array([], dtype=int))), None)
    assert module.training


def test_invalid_watch_index_is_actionable():
    with pytest.raises(ValueError, match="Watch index"):
        ParameterWatch("weight", (4, 5)).capture(model(), 0)
