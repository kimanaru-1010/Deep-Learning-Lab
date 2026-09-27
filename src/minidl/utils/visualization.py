import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / "outputs" / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _save(figure, path, show=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(path)
    if show:
        # Desktop image viewer avoids changing the headless backend mid-run.
        import webbrowser

        webbrowser.open(path.resolve().as_uri())
    plt.close(figure)
    return path


def plot_metrics(history, keys, path, *, axis="step", show=False):
    figure, axes = plt.subplots()
    for key in keys:
        points = [
            (r[axis], r[key])
            for r in history.rows
            if r.get(key) is not None and r.get(axis) is not None
        ]
        if points:
            x, y = zip(*points)
            axes.plot(x, y, label=key)
    axes.set_xlabel(axis)
    axes.set_ylabel(" / ".join(keys))
    if axes.lines:
        axes.legend()
    return _save(figure, path, show)


def plot_loss(history, path="outputs/plots/loss.png", show=False):
    return plot_metrics(history, ["train_loss", "val_loss"], path, show=show)


def plot_accuracy(history, path="outputs/plots/accuracy.png", show=False):
    return plot_metrics(history, ["train_acc", "val_acc"], path, axis="epoch", show=show)


def plot_gradient_norm(history, path="outputs/plots/grad_norm.png", show=False):
    return plot_metrics(history, ["grad_norm"], path, show=show)


def plot_parameter_norm(history, path="outputs/plots/param_norm.png", show=False):
    return plot_metrics(history, ["param_norm"], path, show=show)


def plot_learning_rate(history, path="outputs/plots/lr.png", show=False):
    return plot_metrics(history, ["lr"], path, show=show)


def plot_confusion_matrix(matrix, path="outputs/plots/confusion_matrix.png", show=False):
    figure, axes = plt.subplots()
    axes.imshow(matrix, cmap="Blues")
    axes.set(xlabel="Predicted class", ylabel="True class")
    return _save(figure, path, show)


def plot_history(history, directory, show=False):
    for function, name in [
        (plot_loss, "loss"),
        (plot_accuracy, "accuracy"),
        (plot_gradient_norm, "grad_norm"),
        (plot_parameter_norm, "param_norm"),
        (plot_learning_rate, "lr"),
    ]:
        function(history, Path(directory) / f"{name}.png", show=show)
